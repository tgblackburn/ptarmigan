//! Parse input configuration file

use std::fmt;
use std::error::Error;
use std::path::Path;
use yaml_rust::{YamlLoader, yaml::Yaml};
use meval::Context;

use crate::constants::*;

/// Represents the input configuration, which defines values
/// for simulation parameters, and any automatic values
/// for those parameters.
pub struct Config<'a> {
    input: Yaml,
    ctx: Context<'a>,
}

/// Effectively a triple of (file, section, field) that
/// can be converted into various types.
pub struct Key<'a, 'b> {
    config: &'a Config<'a>,
    section: &'b str,
    field: &'b str,
}

impl<'a, 'b> Key<'a, 'b> {
    fn new(config: &'a Config<'a>, section: &'b str, field: &'b str) -> Self {
        Key {config: config, section: section, field: field}
    }
}

/// The reason for a failure of `Config::read`
#[derive(PartialEq,Copy,Clone)]
pub enum ConfigErrorKind {
    MissingFile,
    MissingSection,
    MissingField,
    ConversionFailure,
}

/// Supplies the cause and origin of a failure of `Config::read`
#[derive(Clone)]
pub struct ConfigError {
    kind: ConfigErrorKind,
    section: String,
    field: String,
}

impl fmt::Debug for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ConfigErrorKind::*;
        let help_msg = "Usage: mpirun -n np ./opal input-file";
        match self.kind {
            MissingFile => write!(f, "Unable to open configuration file.\n{}", help_msg),
            MissingSection => write!(f, "Could not find section \"{}\".\n{}", self.section, help_msg),
            MissingField => write!(f, "Could not find field \"{}\" in section \"{}\".\n{}", self.field, self.section, help_msg),
            ConversionFailure => write!(f, "Could not convert field \"{}\" in section \"{}\" to target type.\n{}", self.field, self.section, help_msg),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ConfigError {
    /// Constructs a new ConfigError
    pub fn raise(kind: ConfigErrorKind, section: &str, field: &str) -> Self {
        ConfigError {kind: kind, section: section.to_owned(), field: field.to_owned()}
    }
}

impl Error for ConfigError {}

use std::convert::{TryFrom};

impl<'a> Config<'a> {
    /// Loads a configuration file.
    /// Fails if the file cannot be opened or if it is not
    /// YAML-formatted.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        use ConfigErrorKind::*;
        let contents = std::fs::read_to_string(path)
            .map_err(|_| ConfigError::raise(MissingFile, "", ""))?;
        Self::from_string(&contents)
    }

    /// Loads a YAML configuration from a string.
    /// Fails if the string is not formatted correctly.
    fn from_string(s: &str) -> Result<Self, ConfigError> {
        use ConfigErrorKind::*;
        let input = YamlLoader::load_from_str(s)
            .map_err(|_| ConfigError::raise(MissingFile, "", ""))?;
        let input = input.first()
            .ok_or(ConfigError::raise(MissingFile, "", ""))?;

        Ok(Config {
            input: input.clone(),
            ctx: Context::new(),
        })
    }

    /// Loads automatic values for constants, special functions
    /// and keywords.
    /// Also loads and evaluates mathematical expressions
    /// that are given in the specified `section`.
    pub fn with_context(&mut self, section: &str) -> &mut Self {
        // Default constants and plasma-related functions

        let gauss_pulse_re = |args: &[f64]| -> f64 {
            let t = args[0];
            let x = args[1];
            let omega = args[2];
            let sigma = args[3];
            let phi = omega * (t - x/SPEED_OF_LIGHT);
            let carrier = phi.sin() + phi * phi.cos() / sigma.powi(2);
            let envelope = (-phi.powi(2) / (2.0 * sigma.powi(2))).exp();
            carrier * envelope
        };

        let gauss_pulse_im = |args: &[f64]| -> f64 {
            let t = args[0];
            let x = args[1];
            let omega = args[2];
            let sigma = args[3];
            let phi = omega * (t - x/SPEED_OF_LIGHT);
            let carrier = phi.cos() - phi * phi.sin() / sigma.powi(2);
            let envelope = (-phi.powi(2) / (2.0 * sigma.powi(2))).exp();
            carrier * envelope
        };

        self.ctx
            .var("m", ELECTRON_MASS)
            .var("me", ELECTRON_MASS)
            .var("mp", PROTON_MASS)
            .var("c", SPEED_OF_LIGHT)
            .var("e", ELEMENTARY_CHARGE)
            .var("eV", ELEMENTARY_CHARGE)
            .var("keV", 1.0e3 * ELEMENTARY_CHARGE)
            .var("MeV", 1.0e6 * ELEMENTARY_CHARGE)
            .var("GeV", 1.0e9 * ELEMENTARY_CHARGE)
            .var("femto", 1.0e-15)
            .var("pico", 1.0e-12)
            .var("nano", 1.0e-9)
            .var("micro", 1.0e-6)
            .var("milli", 1.0e-3)
            .func3("step", |x, min, max| if x >= min && x < max {1.0} else {0.0})
            .func3("gauss", |x, mu, sigma| (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp())
            .func("critical", |omega| VACUUM_PERMITTIVITY * ELECTRON_MASS * omega.powi(2) / ELEMENTARY_CHARGE.powi(2))
            .funcn("gauss_pulse_re", gauss_pulse_re, 4)
            .funcn("gauss_pulse_im", gauss_pulse_im, 4);

        // Read in from 'constants' block if it exists
        if self.input[section].is_badvalue() {
            return self;
        }

        let tmp = self.ctx.clone(); // a constant cannot depend on other constants yet...
        //println!("{:#?}", self.input[section].as_hash());

        for (a, b) in self.input[section].as_hash().unwrap() {
            //println!("{:?} {:?}", a, b);
            match (a, b) {
                (Yaml::String(s), Yaml::Real(v)) => {
                    if let Ok(num) = v.parse::<f64>() {self.ctx.var(s, num);}
                },
                (Yaml::String(s), Yaml::String(v)) => {
                    if let Ok(expr) = v.parse::<meval::Expr>() {
                        if let Ok(num) = expr.eval_with_context(&tmp) {self.ctx.var(s, num);}
                    }
                },
                _ => ()
            }
        }

        self
    }

    /// Test if the file contains a specific section
    pub fn contains(&self, section: &str) -> bool {
        use std::ops::Not;
        self.input[section].is_badvalue().not()
    }

    /// Locates a key-value pair in the configuration file (as specified by
    /// `section` and `field`) and attempts to parse it as the specified type.
    pub fn read<'b, T>(&'a self, section: &'b str, field: &'b str) -> Result<T, ConfigError>
    where T: TryFrom<Key<'a, 'b>> {
        use ConfigErrorKind::*;
        // Does the section exist?
        if self.input[section].is_badvalue() {
            return Err(ConfigError::raise(MissingSection, section, field));
        }
        // Does the field exist?
        if self.input[section][field].is_badvalue() {
            return Err(ConfigError::raise(MissingField, section, field));
        }
        // Now try conversion:
        let key = Key::new(&self, section, field);
        T::try_from(key).map_err(|_| ConfigError::raise(ConversionFailure, section, field))
    }

    /// Like `Config::read`, but parses the value of a key-value pair
    /// as a function of a single variable `arg`.
    pub fn func(&'a self, section: &str, field: &str, arg: &str) -> Result<impl Fn(f64) -> f64 + 'a, ConfigError> {
        use ConfigErrorKind::*;
        // Does the section exist?
        if self.input[section].is_badvalue() {
            return Err(ConfigError::raise(MissingSection, section, field));
        }
        // Does the field exist?
        if self.input[section][field].is_badvalue() {
            return Err(ConfigError::raise(MissingField, section, field));
        }
        // Now try conversion:
        match &self.input[section][field] {
            Yaml::String(s) | Yaml::Real(s) => {
                let expr = s
                    .parse::<meval::Expr>()
                    .map_err(|_| ConfigError::raise(ConversionFailure, section, field))?;
                let func = expr
                    .bind_with_context(&self.ctx, arg)
                    .map_err(|_| ConfigError::raise(ConversionFailure, section, field))?;
                Ok(func)
            },
            _ => Err(ConfigError::raise(ConversionFailure, section, field))
        }
    }

    /// Like `Config::read`, but parses the value of a key-value pair
    /// as a function of two variables.
    pub fn func2(&'a self, section: &str, field: &str, arg: [&str; 2]) -> Result<impl Fn(f64, f64) -> f64 + 'a, ConfigError> {
        use ConfigErrorKind::*;
        // Does the section exist?
        if self.input[section].is_badvalue() {
            return Err(ConfigError::raise(MissingSection, section, field));
        }
        // Does the field exist?
        if self.input[section][field].is_badvalue() {
            return Err(ConfigError::raise(MissingField, section, field));
        }
        // Now try conversion:
        match &self.input[section][field] {
            Yaml::String(s) | Yaml::Real(s) => {
                let expr = s
                    .parse::<meval::Expr>()
                    .map_err(|_| ConfigError::raise(ConversionFailure, section, field))?;
                let func = expr
                    .bind2_with_context(&self.ctx, arg[0], arg[1])
                    .map_err(|_| ConfigError::raise(ConversionFailure, section, field))?;
                Ok(func)
            },
            _ => Err(ConfigError::raise(ConversionFailure, section, field))
        }
    }

    /// Like `Config::read`, but parses the value of a key-value pair
    /// as a function of three variables.
    pub fn func3(&'a self, section: &str, field: &str, arg: [&str; 3]) -> Result<impl Fn(f64, f64, f64) -> f64 + 'a, ConfigError> {
        use ConfigErrorKind::*;
        // Does the section exist?
        if self.input[section].is_badvalue() {
            return Err(ConfigError::raise(MissingSection, section, field));
        }
        // Does the field exist?
        if self.input[section][field].is_badvalue() {
            return Err(ConfigError::raise(MissingField, section, field));
        }
        // Now try conversion:
        match &self.input[section][field] {
            Yaml::String(s) | Yaml::Real(s) => {
                let expr = s
                    .parse::<meval::Expr>()
                    .map_err(|_| ConfigError::raise(ConversionFailure, section, field))?;
                let func = expr
                    .bind3_with_context(&self.ctx, arg[0], arg[1], arg[2])
                    .map_err(|_| ConfigError::raise(ConversionFailure, section, field))?;
                Ok(func)
            },
            _ => Err(ConfigError::raise(ConversionFailure, section, field))
        }
    }
}

impl<'a,'b> TryFrom<Key<'a,'b>> for f64 {
    type Error = ();
    fn try_from(key: Key<'a,'b>) -> Result<Self, Self::Error> {
        match &key.config.input[key.section][key.field] {
            Yaml::Real(s) => {
                s.parse::<f64>().map_err(|_| ())
            },
            Yaml::Integer(i) => {
                Ok(*i as f64)
            },
            Yaml::String(s) => {
                let expr = s.parse::<meval::Expr>().map_err(|_| ())?;
                expr.eval_with_context(&key.config.ctx).map_err(|_| ())
            }
            _ => Err(())
        }
    }
}

impl<'a,'b> TryFrom<Key<'a,'b>> for i64 {
    type Error = ();
    fn try_from(key: Key<'a,'b>) -> Result<Self, Self::Error> {
        match &key.config.input[key.section][key.field] {
            Yaml::Integer(i) => Ok(*i),
            _ => Err(())
        }
    }
}

impl<'a,'b> TryFrom<Key<'a,'b>> for usize {
    type Error = ();
    fn try_from(key: Key<'a,'b>) -> Result<Self, Self::Error> {
        let i = i64::try_from(key)?;
        usize::try_from(i).map_err(|_| ())
    }
}

impl<'a,'b> TryFrom<Key<'a,'b>> for bool {
    type Error = ();
    fn try_from(key: Key<'a,'b>) -> Result<Self, Self::Error> {
        match &key.config.input[key.section][key.field] {
            Yaml::Boolean(b) => Ok(*b),
            _ => Err(())
        }
    }
}

impl<'a,'b> TryFrom<Key<'a,'b>> for Vec<String> {
    type Error = ();
    fn try_from(key: Key<'a,'b>) -> Result<Self, Self::Error> {
        match &key.config.input[key.section][key.field] {
            // turn a single String into a vec of length 1.
            Yaml::String(s) => {
                Ok(vec![s.clone()])
            },
            Yaml::Array(array) => {
                // a is a vec of Vec<Yaml>
                let take_yaml_string = |y: &Yaml| -> Option<String> {
                    match y {
                        Yaml::String(s) => Some(s.clone()),
                        _ => None
                    }
                };
                let got: Vec<String> = array.iter().filter_map(take_yaml_string).collect();
                if got.is_empty() {
                    Err(())
                } else {
                    Ok(got)
                }
            },
            _ => Err(())
        }
    }
}

impl<'a,'b> TryFrom<Key<'a,'b>> for String {
    type Error = ();
    fn try_from(key: Key<'a,'b>) -> Result<Self, Self::Error> {
        match &key.config.input[key.section][key.field] {
            Yaml::String(s) => Ok(s.clone()),
            _ => Err(())
        }
    }
}

/// Estimated time to completion, based on amount of work done
pub fn ettc(start: std::time::Instant, current: usize, total: usize) -> std::time::Duration {
    let rt = start.elapsed().as_secs_f64();
    let ettc = rt * ((total - current) as f64) / (current as f64);
    std::time::Duration::from_secs_f64(ettc)
}

/// Wrapper around std::time::Duration
pub struct PrettyDuration {
    pub duration: std::time::Duration,
}

impl From<std::time::Duration> for PrettyDuration {
    fn from(duration: std::time::Duration) -> PrettyDuration {
        PrettyDuration {duration: duration}
    }
}

impl fmt::Display for PrettyDuration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut t = self.duration.as_secs();
        let s = t % 60;
        t /= 60;
        let min = t % 60;
        t /= 60;
        let hr = t % 24;
        let d = t / 24;
        if d > 0 {
            write!(f, "{}d {:02}:{:02}:{:02}", d, hr, min, s)
        } else {
            write!(f, "{:02}:{:02}:{:02}", hr, min, s)
        }
    }
}

/// Wrapper around the simulation time (in seconds)
pub struct SimulationTime(pub f64);

impl fmt::Display for SimulationTime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // find nearest SI prefix
        let power = 3.0 * ((self.0.abs().log10() + 0.0) / 3.0).floor();
        // and clip to -18 <= x <= 0
        let power = power.min(0.0f64).max(-18.0f64);
        let power = power as i32;
        let (unit, scale) = match power {
            -18 => ("as", 1.0e18),
            -15 => ("fs", 1.0e15),
            -12 => ("ps", 1.0e12),
            -9  => ("ns", 1.0e9),
            -6  => ("\u{03bc}s", 1.0e6),
            -3  => ("ms", 1.0e3),
            _   => (" s", 1.0)
        };
        write!(f, "{: >8.2} {}", scale * self.0, unit)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts;
    use super::*;

    #[test]
    fn config_parser() {
        let text = "---
        control:
          dx: 0.001
          nx: 4000
          ne: sin(a * x)
          ib: a * b^3
        
        extra:
          dx: 160

        constants:
          a: 2.0 * pi
          b: 17.0
        ";

        let mut config = Config::from_string(&text).unwrap();
        config.with_context("constants");

        // Plain f64
        let dx: f64 = config.read("control", "dx").unwrap();
        assert_eq!(dx, 0.001);

        // Plain usize
        let nx: usize = config.read("control", "nx").unwrap();
        assert_eq!(nx, 4000);

        // Evaluates math expr
        let ib: f64 = config.read("control", "ib").unwrap();
        assert_eq!(ib, 2.0 * consts::PI * 17.0f64.powi(3));

        // Implicit onversion from integer to f64
        let dx: f64 = config.read("extra", "dx").unwrap();
        assert_eq!(dx, 160.0);

        // Function of one variable
        let ne = config.func("control", "ne", "x").unwrap();
        assert_eq!(ne(0.6), (2.0 * consts::PI * 0.6).sin());
    }

    #[test]
    fn time_format() {
        let t = 2.6e-4_f64;
        let output = SimulationTime(t).to_string();
        println!("\"{}\" => \"{}\"", t, output);
        assert_eq!(output, "  260.00 \u{03bc}s");
    }
}