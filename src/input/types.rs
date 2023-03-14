
//! YAML-readable types

use std::convert::TryFrom;
use yaml_rust::yaml::Yaml;
use evalexpr::{HashMapContext, eval_number_with_context};
// use meval::Context;

/// Types that can be parsed from a YML-formatted file
pub trait FromYaml: Sized {
    type Error;
    /// Attempt to parse the YML field as the specified type, using the supplied Context for named variables and constants.
    fn from_yaml(arg: Yaml, ctx: &HashMapContext) -> Result<Self, Self::Error>;
}

// Atomic

impl FromYaml for bool {
    type Error = ();
    fn from_yaml(arg: Yaml, _ctx: &HashMapContext) -> Result<Self, Self::Error> {
        match arg {
            Yaml::Boolean(b) => Ok(b),
            _ => Err(())
        }
    }
}

impl FromYaml for String {
    type Error = ();
    fn from_yaml(arg: Yaml, _ctx: &HashMapContext) -> Result<Self, Self::Error> {
        match arg {
            Yaml::String(s) => Ok(s.clone()),
            Yaml::Integer(i) => Ok(i.to_string()),
            Yaml::Real(s) => Ok(s.clone()),
            Yaml::Boolean(b) => Ok(b.to_string()),
            _ => Err(())
        }
    }
}

// Numbers: f64, i64, usize

impl FromYaml for f64 {
    type Error = ();
    fn from_yaml(arg: Yaml, ctx: &HashMapContext) -> Result<Self, Self::Error> {
        match arg {
            Yaml::Real(s) => {
                s.parse::<f64>().or(Err(()))
            },
            Yaml::Integer(i) => {
                Ok(i as f64)
            },
            Yaml::String(s) => {
                eval_number_with_context(&s, ctx)
                    .or(Err(()))
                // s.parse::<meval::Expr>()
                //     .and_then(|expr| expr.eval_with_context(ctx))
            }
            _ => Err(())
        }
    }
}

impl FromYaml for i64 {
    type Error = ();
    fn from_yaml(arg: Yaml, _ctx: &HashMapContext) -> Result<Self, Self::Error> {
        match arg {
            Yaml::Integer(i) => Ok(i),
            _ => Err(())
        }
    }
}

impl FromYaml for usize {
    type Error = ();
    fn from_yaml(arg: Yaml, ctx: &HashMapContext) -> Result<Self, Self::Error> {
        let i: i64 = FromYaml::from_yaml(arg, ctx)?;
        usize::try_from(i).map_err(|_| ())
    }
}

// Vecs

impl FromYaml for Vec<String> {
    type Error = ();
    fn from_yaml(arg: Yaml, _ctx: &HashMapContext) -> Result<Self, Self::Error> {
        match arg {
            // turn a single String into a vec of length 1.
            Yaml::String(s) | Yaml::Real(s) => {
                Ok(vec![s.clone()])
            },
            Yaml::Integer(i) =>  {
                Ok(vec![i.to_string()])
            },
            Yaml::Boolean(b) =>  {
                Ok(vec![b.to_string()])
            },
            Yaml::Array(array) => {
                // a is a vec of Vec<Yaml>
                let take_yaml_string = |y: &Yaml| -> Option<String> {
                    match y {
                        Yaml::String(s) | Yaml::Real(s) => Some(s.clone()),
                        Yaml::Integer(i) => Some(i.to_string()),
                        Yaml::Boolean(b) => Some(b.to_string()),
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
impl FromYaml for Vec<f64> {
    type Error = ();
    fn from_yaml(arg: Yaml, ctx: &HashMapContext) -> Result<Self, Self::Error> {
        let strs: Vec<String> = FromYaml::from_yaml(arg, ctx)?;
        let v: Result<Vec<f64>, _> = strs.iter()
            .map(|s| {
                eval_number_with_context(&s, ctx)
                    .or(Err(()))
                // s.parse::<meval::Expr>()
                //     .and_then(|expr| expr.eval_with_context(ctx))
            })
            .collect();
        v
    }
}
