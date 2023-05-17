//! Parse input configuration file

use std::path::Path;
use std::ops::Add;
use yaml_rust::{YamlLoader, yaml::Yaml};
use evalexpr::*;

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
pub struct Config {
    input: Yaml,
    ctx: HashMapContext,
}

impl Config {
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
            ctx: HashMapContext::new(),
        })
    }

    /// Loads automatic values for constants, special functions
    /// and keywords.
    /// Also loads and evaluates mathematical expressions
    /// that are given in the specified `section`.
    pub fn with_context(&mut self, section: &str) -> Result<&mut Self, InputError> {
        use helper::context_function;
        // Default constants and plasma-related functions

        let mut ctx = context_map! {
            "m" => ELECTRON_MASS,
            "me" => ELECTRON_MASS,
            "mp" => PROTON_MASS,
            "c" => SPEED_OF_LIGHT,
            "e" => ELECTRON_CHARGE,
            "qe" => ELECTRON_CHARGE,
            "eV" => ELEMENTARY_CHARGE,
            "keV" => 1.0e3 * ELEMENTARY_CHARGE,
            "MeV" => 1.0e6 * ELEMENTARY_CHARGE,
            "GeV" => 1.0e9 * ELEMENTARY_CHARGE,
            "femto" => 1.0e-15,
            "pico" => 1.0e-12,
            "nano" => 1.0e-9,
            "micro" => 1.0e-6,
            "milli" => 1.0e-3,
            "pi" => std::f64::consts::PI,
            "degree" => std::f64::consts::PI / 180.0,
        }.unwrap();

        context_function!(ctx, "sqrt",   f64::sqrt);
        context_function!(ctx, "cbrt",   f64::cbrt);
        context_function!(ctx, "abs",    f64::abs);
        context_function!(ctx, "exp",    f64::exp);
        context_function!(ctx, "ln",     f64::ln);
        context_function!(ctx, "sin",    f64::sin);
        context_function!(ctx, "cos",    f64::cos);
        context_function!(ctx, "tan",    f64::tan);
        context_function!(ctx, "asin",   f64::asin);
        context_function!(ctx, "acos",   f64::acos);
        context_function!(ctx, "atan",   f64::atan);
        context_function!(ctx, "atan2",  f64::atan2, 2);
        context_function!(ctx, "sinh",   f64::sinh);
        context_function!(ctx, "cosh",   f64::cosh);
        context_function!(ctx, "tanh",   f64::tanh);
        context_function!(ctx, "asinh",  f64::asinh);
        context_function!(ctx, "acosh",  f64::acosh);
        context_function!(ctx, "atanh",  f64::atanh);
        context_function!(ctx, "floor",  f64::floor);
        context_function!(ctx, "ceil",   f64::ceil);
        context_function!(ctx, "round",  f64::round);
        context_function!(ctx, "signum", f64::signum);

        context_function!(ctx, "step",     |x: f64, min: f64, max: f64| {if x >= min && x < max {1.0} else {0.0}}, 3);
        context_function!(ctx, "gauss",    |x: f64, mu: f64, sigma: f64| (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp(), 3);
        context_function!(ctx, "critical", |omega: f64| VACUUM_PERMITTIVITY * ELECTRON_MASS * omega.powi(2) / ELEMENTARY_CHARGE.powi(2));

        self.ctx = ctx;

        // Read in from 'constants' block if it exists
        if self.input[section].is_badvalue() {
            return Ok(self);
        }

        for (a, b) in self.input[section].as_hash().unwrap() {
            // grab the value, if possible
            let (key, value) = match (a, b) {
                (Yaml::String(k), Yaml::Integer(i)) => (Some(k), Some(*i as f64)),
                (Yaml::String(k), Yaml::Real(s)) => (Some(k), s.parse::<f64>().ok()),
                (Yaml::String(k), Yaml::String(s)) => (Some(k), eval_number_with_context(s, &self.ctx).ok()),
                _ => (None, None),
            };

            // insert it into the context so it's available for the next read
            if let Some(v) = value {
                let key = key.unwrap(); // if value.is_some() so is key
                self.ctx.set_value(key.clone(), Value::from(v))
                    .map_err(|_| {
                        eprintln!("Failed to insert {} = {} from constants block into context.", key, v);
                        InputError::conversion(section, key)
                    })?
            } else if let Some(k) = key {
                // found a key, value pair but parsing failed
                Err(InputError::conversion(section, k))?
            }
        }

        Ok(self)
    }

    /// Locates a key-value pair in the configuration file and attempts
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
    pub fn func<'a, S: AsRef<str> + 'a>(&'a self, path: S, arg: S) -> Result<impl Fn(f64) -> f64 + 'a, InputError> {
        // get the field, if it exists
        let s: String = self.read(&path)?;

        let tree = build_operator_tree(&s)
            .map_err(|_| InputError::conversion(path.as_ref(), &s))?;

        // walk tree and verify there are no missing identifiers, apart from 'arg'
        for var in tree.iter_read_variable_identifiers() {
            if var == arg.as_ref() || self.ctx.iter_variable_names().find(|id| var == id).is_some() {
                continue;
            } else {
                return Err(InputError::conversion(path.as_ref(), &s))
            }
        }

        let func = move |x| {
            let name = arg.as_ref().to_owned();
            let mut ctx = self.ctx.clone();
            ctx.set_value(name, Value::from(x)).unwrap();
            tree.eval_number_with_context(&ctx).unwrap()
        };

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
        eval_number_with_context(arg.as_ref(), &self.ctx).ok()
    }

    /// Locates a key-value pair in the configuration file and attempts
    /// to parse it as a looped variable, returning a Vec of the values.
    /// The loop is defined by a `start`, `stop` and `step`:
    ///
    /// ```
    /// let text: &str = "---
    ///     x:
    ///         start: 1.0
    ///         stop: 1.5
    ///         step: 0.1
    /// ";
    ///
    /// let values: Vec<f64> = Config::from_string(&text).unwrap()
    ///     .read_loop("x").unwrap();
    ///
    /// assert_eq!(values, vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5]);
    /// ```
    pub fn read_loop<T, S>(&self, path: S) -> Result<Vec<T>, InputError>
    where
        T: FromYaml + PartialOrd + Add<Output=T> + Copy,
        S: AsRef<str> {
        let key = path.as_ref();

        if self.read::<T, _>(format!("{}{}", key, ":start").as_str()).is_err() {
            let value = self.read(path)?;
            let v = vec![value];
            Ok(v)
        }
        else { // 'start' value found
            let start = self.read(format!("{}{}", key, ":start").as_str())?;
            let stop = self.read(format!("{}{}", key, ":stop").as_str())?;
            let step = self.read(format!("{}{}", key, ":step").as_str())?;

            let mut v: Vec<T> = Vec::new();
            let mut x = start;
            while x <= stop {
                v.push(x);
                x = x + step;
            }
            Ok(v)
        }
    }
}

mod helper {
    macro_rules! context_function {
        ($ctx:expr, $name:literal, $func:expr) => {
            $ctx.set_function(
                $name.to_string(),
                Function::new(|arg| {
                    let x = arg.as_number()?;
                    Ok(Value::Float($func(x)))
                })
            ).unwrap()
        };
        ($ctx:expr, $name:literal, $func:expr, 2) => {
            $ctx.set_function(
                $name.to_string(),
                Function::new(|arg| {
                    let arg = arg.as_fixed_len_tuple(2)?;
                    let x = arg[0].as_number()?;
                    let y = arg[1].as_number()?;
                    Ok(Value::Float($func(x, y)))
                })
            ).unwrap()
        };
        ($ctx:expr, $name:literal, $func:expr, 3) => {
            $ctx.set_function(
                $name.to_string(),
                Function::new(|arg| {
                    let arg = arg.as_fixed_len_tuple(3)?;
                    let x = arg[0].as_number()?;
                    let y = arg[1].as_number()?;
                    let z = arg[2].as_number()?;
                    Ok(Value::Float($func(x, y, z)))
                })
            ).unwrap()
        };
    }

    pub(super) use context_function;
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
          N: 2.0
          a: N * pi
          b: 17.0
          y: 1.0

        deep:
          nested:
            section:
              key: 1.0
        ";

        let mut config = Config::from_string(&text).unwrap();
        config.with_context("constants").unwrap();

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

        let ne = config.func("control:ne", "y");
        assert!(ne.is_err());

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
        let a0_values1: Vec<f64> = config.read_loop("laser:a0").unwrap();
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
        let a0_values: Vec<f64> = config.read_loop("laser:a0").unwrap();
        assert_eq!(a0_values, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }
    
}