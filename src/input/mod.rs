//! Parse input configuration file

// use std::fmt;
// use std::error::Error;
use std::path::Path;
use yaml_rust::{YamlLoader, yaml::Yaml};
use meval::Context;

use crate::constants::*;

mod error;
mod types;
mod timing;

pub use error::*;
use types::*;
pub use timing::*;

/// Represents the input configuration, which defines values
/// for simulation parameters, and any automatic values
/// for those parameters.
pub struct Config<'a> {
    input: Yaml,
    ctx: Context<'a>,
}

impl<'a> Config<'a> {
    /// Loads a configuration file.
    /// Fails if the file cannot be opened or if it is not
    /// YAML-formatted.
    #[allow(unused)]
    pub fn from_file(path: &Path) -> Result<Self, InputError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|_| InputError::file())?;
        Self::from_string(&contents)
    }

    /// Loads a YAML configuration from a string.
    /// Fails if the string is not formatted correctly.
    #[allow(unused)]
    pub fn from_string(s: &str) -> Result<Self, InputError> {
        let input = YamlLoader::load_from_str(s)
            .map_err(|_| InputError::file())?;
        let input = input.first()
            .ok_or(InputError::file())?;

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
            .var("qe", ELECTRON_CHARGE)
            .var("eV", ELEMENTARY_CHARGE)
            .var("keV", 1.0e3 * ELEMENTARY_CHARGE)
            .var("MeV", 1.0e6 * ELEMENTARY_CHARGE)
            .var("GeV", 1.0e9 * ELEMENTARY_CHARGE)
            .var("femto", 1.0e-15)
            .var("pico", 1.0e-12)
            .var("nano", 1.0e-9)
            .var("micro", 1.0e-6)
            .var("milli", 1.0e-3)
            .var("degree", std::f64::consts::PI / 180.0)
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

    /// Locates a key-value pair in the configuration file and attempts the
    /// to parse the value as the specified type.
    /// The path to the key-value pair is specified by a string of colon-separated 
    /// sections, e.g. `'section:subsection:subsubsection:key'`.
    pub fn read<T, S>(&self, path: S) -> Result<T, InputError>
    where
        T: FromYaml,
        S: AsRef<str>,
    {
        let address: Vec<&str> = path.as_ref().split(':').collect();
        let value = address.iter()
          .try_fold(&self.input, |y, s| {
              if y[*s].is_badvalue() {
                  Err(InputError::location(path.as_ref(), s))
              } else {
                  Ok(&y[*s])
              }
          });
        value.and_then(|arg| T::from_yaml(arg.clone(), &self.ctx).map_err(|_| InputError::conversion(path.as_ref(), address.last().unwrap())))
    }

    /// Like `Config::read`, but parses the value of a key-value pair
    /// as a function of a single variable `arg`.
    #[allow(unused)]
    pub fn func<S: AsRef<str>>(&'a self, path: S, arg: S) -> Result<impl Fn(f64) -> f64 + 'a, InputError> {
        // get the field, if it exists
        let s: String = self.read(&path)?;
        // Now try conversion:
        let expr = s
            .parse::<meval::Expr>()
            .map_err(|_| InputError::conversion(path.as_ref(), &s))?;
        let func = expr
            .bind_with_context(&self.ctx, arg.as_ref())
            .map_err(|_| InputError::conversion(path.as_ref(), &s))?;
        Ok(func)
    }

    /// Parses a string argument and evaluates it using the default context. Extends
    /// ```
    /// let arg = "2.0";
    /// let val = arg.parse::<f64>().unwrap();
    /// ```
    /// to handle mathematical expressions, e.g.
    /// ```
    /// let arg = "2.0 / (1.0 + density)";
    /// let val = input.evaluate(arg).unwrap();
    /// ```
    /// where 'density' is specified in the input file.
    #[allow(unused)]
    pub fn evaluate<S: AsRef<str>>(&self, arg: S) -> Option<f64> {
        arg.as_ref()
            .parse::<meval::Expr>()
            .and_then(|expr| expr.eval_with_context(&self.ctx))
            .ok()
    }

    /// Uses 'Config::read' to determine if a0 is to be looped over. If a "start" key is given,
    /// assumes that a0 will be looped over.
    #[allow(unused)]
    pub fn read_loop<S: AsRef<str> + std::fmt::Display>(&self, path: S) -> Result<Vec<f64>, InputError> {
        if self.read::<f64, &str>(format!("{}{}", path, ":start").as_str()).is_err() { 
            let value: f64 = self.read(path)?;                                                  
            let v: Vec<f64> = vec![value; 1];                                                   
            Ok(v)
        }
        else { // 'start' value found
            let start: f64 = self.read(format!("{}{}", path, ":start").as_str())?;
            let stop: f64 = self.read(format!("{}{}", path, ":stop").as_str())?;
            let step: f64 = self.read(format!("{}{}", path, ":step").as_str())?;

            let mut v: Vec<f64> = Vec::new();
            let mut x: f64 = start;
            while x <= stop {
                v.push(x);
                x += step;
            }
            Ok(v)
        }
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
          r: [0.0, b, 1.0, 2.0 * a]
          s: [0.0, none]

        constants:
          a: 2.0 * pi
          b: 17.0
          y: 1.0

        deep:
          nested:
            section:
              key: 1.0
        ";

        let mut config = Config::from_string(&text).unwrap();
        config.with_context("constants");

        // Plain f64
        let dx: f64 = config.read("control:dx").unwrap();
        assert_eq!(dx, 0.001);

        // Plain usize
        let nx: usize = config.read("control:nx").unwrap();
        assert_eq!(nx, 4000);

        // Evaluates math expr
        let ib: f64 = config.read("control:ib").unwrap();
        assert_eq!(ib, 2.0 * consts::PI * 17.0f64.powi(3));

        // Implicit onversion from integer to f64
        let dx: f64 = config.read("extra:dx").unwrap();
        assert_eq!(dx, 160.0);

        // Function of one variable
        let ne = config.func("control:ne", "x").unwrap();
        assert_eq!(ne(0.6), (2.0 * consts::PI * 0.6).sin());

        // array of f64
        let r: Vec<f64> = config.read("extra:r").unwrap();
        assert_eq!(r.len(), 4);
        assert_eq!(r[0], 0.0);
        assert_eq!(r[1], 17.0);
        assert_eq!(r[2], 1.0);
        assert_eq!(r[3], 4.0 * consts::PI);

        let s: Result<Vec<f64>, _> = config.read("extra:s");
        assert!(s.is_err());

        let key: f64 = config.read("deep:nested:section:key").unwrap();
        assert_eq!(key, 1.0);

        // evaluate arb string
        let val = config.evaluate("1.0 / (1.0 + y)").unwrap();
        assert_eq!(val, 1.0 / 2.0);
    }

    #[test]
    fn looper() {
        // Test extraction of single value
        let text: &str = "---
        laser:
            a0: 10.0
        ";
        let mut config = Config::from_string(&text).unwrap();
        //config.with_context("constants");
        let a0_values1 = config.read_loop("laser:a0").unwrap();
        assert_eq!(a0_values1, vec![10.0; 1]);
        
        // Test extraction of looped values
        let text: &str = "---
        laser:
            a0:
                start: 1.0
                stop: 10.0
                step: 2.0
        ";
        config = Config::from_string(&text).unwrap();
        //config.with_context("constants");
        let a0_values = config.read_loop("laser:a0").unwrap();
        assert_eq!(a0_values, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }
    
}