//! Post-processing of particle data

use std::fmt;

#[cfg(feature = "with-mpi")]
use mpi::{traits::*, collective::SystemOperation};
#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;

use crate::particle::*;

use super::{ParticleOutput, OutputError, functions};
use super::ParticleOutputType::*;

/// Ways an array of particle data can be reduced to
/// a single, representative value
#[derive(Debug, PartialEq)]
enum Reduction {
    Total,
    Fraction,
    Mean,
    Variance,
    Minimum,
    Maximum,
    CircMean,
    CircVar,
    CircStdDev,
}

impl fmt::Display for Reduction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match self {
            Reduction::Total => "total",
            Reduction::Fraction => "fraction",
            Reduction::Mean => "mean",
            Reduction::Variance => "variance",
            Reduction::Minimum => "minimum",
            Reduction::Maximum => "maximum",
            Reduction::CircMean => "circmean",
            Reduction::CircVar => "circvar",
            Reduction::CircStdDev => "circstd",
        };
        write!(f, "{}", name)
    }
}

impl Reduction {
    /// Returns `true` if this Reduction can only be applied to angular variables
    fn is_circular_stat(&self) -> bool {
        use Reduction::*;
        matches!(self, CircMean | CircVar | CircStdDev)
    }
}

#[derive(Clone)]
struct Operator {
    f: ParticleOutput,
    name: String,
    unit: String,
}

/// Summarizes a set of observations in a single value,
/// using the specified reduction operator (e.g., mean, variance).
pub struct SummaryStatistic {
    name: String,
    variable: Operator,
    filter: Option<Operator>,
    weight: Operator,
    op: Reduction,
    min: f64,
    max: f64,
    val: f64,
}

impl SummaryStatistic {
    /// Parses a string representation of a summary statistic:
    ///  `
    ///   [op] [var][`weight]
    ///   [op] [var][`weight] in ([min]; [max])
    ///   [op] [var][`weight] for [var2] in ([min]; [max])
    /// `
    /// where min and/or max can be 'auto' or a constant value.
    pub fn load<F: Fn(&str) -> Option<f64>>(spec: &str, parser: F) -> Result<Self, OutputError> {
        //println!("Got spec = '{}'", spec);

        // First, check if string contains a bracketed range spec.
        let start = spec.find('(');
        let end = spec.rfind(')');
        let range = if start.is_some() && end.is_some() {
            Some(&spec[start.unwrap()..=end.unwrap()])
        } else {
            None
        };

        // If it does, remove it from the original string
        let words = if let Some(from) = range {
            //println!("Range spec: '{}'", from);
            spec.replace(from, "")
        } else {
            spec.to_owned()
        };

        // which is then split up
        let words: Vec<&str> = words.split_whitespace().collect();

        //println!("Words: {:?}, length = {}", words, words.len());

        // Check for optional filter variable
        let filter = if words.len() == 5 {
            let tmp = functions::identify(words[3]);
            if tmp.is_none() || words[2] != "for" {
                return Err(OutputError::conversion(spec, "summary statistic"));
            } else {
                let unit = match tmp.unwrap().1 {
                    Dimensionless => "1",
                    Angle => "rad",
                    Length => "m",
                    Energy => "MeV",
                    Momentum => "MeV/c",
                };
                Some(Operator {
                    f: tmp.unwrap().0,
                    name: words[3].to_owned(),
                    unit: unit.to_owned(),
                })
            }
        } else {
            None
        };

        // Are we checking for a range?
        let (min, max) = if words.len() == 3 || words.len() == 5 {
            if range.is_none() || words[words.len() - 1] != "in" {
                return Err(OutputError::conversion(spec, "summary statistic"));
            }
            let range = range.unwrap();
            let rs: Vec<&str> = range[1..range.len()-1].split(';').collect();
            //println!("Broke range spec into {:?}", rs);
            let rs: Vec<String> = rs.iter().map(|&s| s.replace(char::is_whitespace, "")).collect();
            //println!("Broke range spec into {:?}", rs);
            let (min, max) = if rs.len() == 2 {
                let min = if rs[0] == "auto" {
                    Some(std::f64::NEG_INFINITY)
                } else {
                    parser(&rs[0])
                };
                let max = if rs[1] == "auto" {
                    Some(std::f64::INFINITY)
                } else {
                    parser(&rs[1])
                };
                (min, max)
            } else {
                (None, None)
            };
            if min.is_some() && max.is_some() {
                // So we have a valid range spec. If filter is None, then
                // this means that variable is also the filter function.
                (min.unwrap(), max.unwrap())
            } else {
                return Err(OutputError::conversion(spec, "summary statistic"));
            }
        } else {
            (std::f64::NEG_INFINITY, std::f64::INFINITY)
        };

        //println!("Bounds are {:e} to {:e}", min, max);

        let (op, variable, weight) = match words.len() {
            2 | 3 | 5 => {
                // The first word must be an Reduction
                let op = match words[0] {
                    "total" => Reduction::Total,
                    "fraction" | "frac" => Reduction::Fraction,
                    "mean" => Reduction::Mean,
                    "variance" | "var" => Reduction::Variance,
                    "minimum" | "min" => Reduction::Minimum,
                    "maximum" | "max" => Reduction::Maximum,
                    "circmean" | "cmean" => Reduction::CircMean,
                    "circvariance" | "cvariance" | "circvar" | "cvar" => Reduction::CircVar,
                    "circstd" | "cstd" => Reduction::CircStdDev,
                    _ => return Err(OutputError::conversion_explained(spec, "summary statistic", "the requested operation is not recognised")),
                };


                // The second word can be either 'variable' or 'variable`weight'
                let varstr: Vec<&str> = words[1].split('`').collect();
                let (varstr, weightstr) = match varstr.len() {
                    1 => (varstr[0], "unit"),
                    2 => (varstr[0], varstr[1]),
                    _ => return Err(OutputError::conversion(spec, "summary statistic")),
                };

                let variable = if let Some((f, f_type)) = functions::identify(varstr) {
                    let unit = match f_type {
                        Dimensionless => "1",
                        Angle => "rad",
                        Length => "m",
                        Energy => "MeV",
                        Momentum => "MeV/c",
                    };

                    // unit is associated with the function at the moment - but needs to
                    // be consistent with op
                    let unit = match op {
                        Reduction::Fraction => "1".to_owned(),
                        Reduction::Variance => if unit == "1" {
                            "1".to_owned()
                        } else {
                            format!("({})^2", unit)
                        },
                        Reduction::CircVar => "1".to_owned(),
                        _ => unit.to_owned(),
                    };

                    // CircMean, CircVar, CircStdDev can only be applied to angles
                    if op.is_circular_stat() && f_type != Angle {
                        return Err(OutputError::conversion_explained(spec, "summary statistic", "circular stats can only be applied to angles"));
                    } else {
                        Operator {
                            f,
                            name: varstr.to_owned(),
                            unit,
                        }
                    }
                } else {
                    return Err(OutputError::conversion_explained(spec, "summary statistic", "the requested variable is not recognised"));
                };

                let weight = if let Some((f, f_type)) = functions::identify(weightstr) {
                    let unit = match f_type {
                        Dimensionless => "1",
                        Angle => "rad",
                        Length => "m",
                        Energy => "MeV",
                        Momentum => "MeV/c",
                    };
                    Operator {
                        f: f,
                        name: weightstr.to_owned(),
                        unit: unit.to_owned(),
                    }
                } else {
                    return Err(OutputError::conversion(spec, "summary statistic"));
                };

                (op, variable, weight)
            }
            _ => return Err(OutputError::conversion(spec, "summary statistic"))
        };

        // if the range limits are something other than -inf and inf,
        // and filter is None, then we're using var as the selector
        let filter = if (min.is_finite() || max.is_finite()) && filter.is_none() {
            Some(variable.clone())
        } else {
            filter
        };

        Ok(SummaryStatistic {
            name: "".to_owned(),
            variable,
            filter,
            weight,
            op,
            min,
            max,
            val: 0.0,
        })
    }

    /// Returns a tuple of: the total of the entire list, the total of the
    /// filtered list, and the number of elements that satisfy the filter.
    fn total(&self, pt: &[Particle]) -> (f64, f64, f64) {
        let mut total = 0.0;
        let mut subtotal = 0.0;
        let mut count = 0.0;
        for p in pt {
            let within_bounds = if self.filter.is_some() {
                let f = self.filter.as_ref().unwrap().f;
                f(p) > self.min && f(p) < self.max
            } else {
                true
            };

            let val = (self.variable.f)(p) * (self.weight.f)(p);
            total += val;

            if within_bounds {
                subtotal += val;
                count += (self.weight.f)(p);
            }
        }

        (total, subtotal, count)
    }

    fn variance(&self, pt: &[Particle], comm: &impl Communicator) -> f64 {
        // Guess the mean first
        let local = {
            let mut total = 0.0;
            let mut count = 0.0;
            for p in pt.iter().take(10) {
                let within_bounds = if self.filter.is_some() {
                    let f = self.filter.as_ref().unwrap().f;
                    f(p) > self.min && f(p) < self.max
                } else {
                    true
                };

                if within_bounds {
                    total += (self.variable.f)(p) * (self.weight.f)(p);
                    count += (self.weight.f)(p);
                }
            }
            [total, count]
        };

        // Need to share approx_mean with all tasks
        let approx_mean = {
            let mut global = [0.0; 2];
            comm.all_reduce_into(&local[..], &mut global[..], SystemOperation::sum());
            global[0] / global[1]
        };

        let mut sum_sq_diffs = 0.0;
        let mut sum_diffs = 0.0;
        let mut count = 0.0;

        for p in pt {
            let within_bounds = if self.filter.is_some() {
                let f = self.filter.as_ref().unwrap().f;
                f(p) > self.min && f(p) < self.max
            } else {
                true
            };

            if within_bounds {
                let val = (self.variable.f)(p);
                let weight = (self.weight.f)(p);
                sum_sq_diffs += weight * (val - approx_mean).powi(2);
                sum_diffs += weight * (val - approx_mean);
                count += weight;
            }

        }

        let variance = {
            let local = [sum_sq_diffs, sum_diffs, count];
            let mut global = [0.0f64; 3];
            comm.all_reduce_into(&local[..], &mut global[..], SystemOperation::sum());
            (global[0] - global[1].powi(2) / global[2]) / global[2]
        };

        variance
    }

    fn min(&self, pt: &[Particle]) -> f64 {
        let mut min = std::f64::INFINITY;
        for p in pt {
            let within_bounds = if self.filter.is_some() {
                let f = self.filter.as_ref().unwrap().f;
                f(p) > self.min && f(p) < self.max
            } else {
                true
            };

            if within_bounds {
                let val = (self.variable.f)(p) * (self.weight.f)(p);
                if val < min {min = val;}
            }
        }
        min
    }

    fn max(&self, pt: &[Particle]) -> f64 {
        let mut max = std::f64::NEG_INFINITY;
        for p in pt {
            let within_bounds = if self.filter.is_some() {
                let f = self.filter.as_ref().unwrap().f;
                f(p) > self.min && f(p) < self.max
            } else {
                true
            };

            if within_bounds {
                let val = (self.variable.f)(p) * (self.weight.f)(p);
                if val > max {max = val;}
            }
        }
        max
    }

    /// Returns an array of: the mean cosine, the mean sine,
    /// and the number of elements that satisfy the filter.
    fn directmean(&self, pt: &[Particle]) -> [f64; 3] {
        let mut xbar = 0.0;
        let mut ybar = 0.0;
        let mut count = 0.0;
        for p in pt {
            let within_bounds = if self.filter.is_some() {
                let f = self.filter.as_ref().unwrap().f;
                f(p) > self.min && f(p) < self.max
            } else {
                true
            };
        
            if within_bounds {
                let val = (self.variable.f)(p);
                let weight = (self.weight.f)(p);
                xbar += weight * f64::cos(val);
                ybar += weight * f64::sin(val);
                count += weight;
            }
        }

        [xbar, ybar, count]

    }

    pub fn evaluate(&mut self, world: &impl Communicator, pt: &[Particle], name: &str) {
        self.name = name.to_owned();
        self.val = match self.op {
            Reduction::Total => {
                let (_, total, _) = self.total(pt);
                let mut gtotal = 0.0;
                world.all_reduce_into(&total, &mut gtotal, SystemOperation::sum());
                gtotal
            },
            Reduction::Fraction => {
                let (total, subtotal, _) = self.total(pt);
                let mut gsubtotal = 0.0;
                let mut gtotal = 0.0;
                world.all_reduce_into(&total, &mut gtotal, SystemOperation::sum());
                world.all_reduce_into(&subtotal, &mut gsubtotal, SystemOperation::sum());
                gsubtotal / gtotal
            },
            Reduction::Mean => {
                let (_, total, count) = self.total(pt);
                let mut gtotal = 0.0;
                let mut gcount = 0.0;
                world.all_reduce_into(&total, &mut gtotal, SystemOperation::sum());
                world.all_reduce_into(&count, &mut gcount, SystemOperation::sum());
                gtotal / gcount
            },
            Reduction::Variance => self.variance(pt, world),
            Reduction::Minimum => {
                let local = self.min(pt);
                let mut global = std::f64::INFINITY;
                world.all_reduce_into(&local, &mut global, SystemOperation::min());
                global
            },
            Reduction::Maximum => {
                let local = self.max(pt);
                let mut global = std::f64::NEG_INFINITY;
                world.all_reduce_into(&local, &mut global, SystemOperation::max());
                global
            },
            Reduction::CircMean => {
                let local = self.directmean(pt);
                let mut global = [0_f64; 3];
                world.all_reduce_into(&local[..], &mut global[..], SystemOperation::sum());
                // Get <X> and <Y>
                let means = [global[0]/global[2], global[1]/global[2]];
                // <θ> = arctan(<Y> / <X>)
                means[1].atan2(means[0])
            },
            Reduction::CircVar => {
                let local = self.directmean(pt);
                let mut global = [0_f64; 3];
                world.all_reduce_into(&local[..], &mut global[..], SystemOperation::sum());
                // Get <X> and <Y>
                let means = [global[0]/global[2], global[1]/global[2]];
                let r_length = means[0].hypot(means[1]); // resultant length, <R>
                // Circular variance = 1 - <R>
                1.0 - r_length
            },
            Reduction::CircStdDev => {
                let local = self.directmean(pt);
                let mut global = [0_f64; 3];
                world.all_reduce_into(&local[..], &mut global[..], SystemOperation::sum());
                // Get <X> and <Y>
                let means = [global[0]/global[2], global[1]/global[2]];
                let r_length = means[0].hypot(means[1]); // resultant length, <R>
                // Circular standard deviation = sqrt(-2 ln<R>)
                ( -2.0 * r_length.ln() ).sqrt()
            }
        };
    }
}

impl fmt::Display for SummaryStatistic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // "electron: fraction (1.00e+3 < energy 1.00e+4) = 5.61e-1 [1]"
        // "photon: mean energy (1.00e+3 < r_perp < 2.00e+3) = 1.000e+3 [MeV]"
        let bounds = if self.filter.is_some() {
            if self.min.is_finite() && self.max.is_infinite() {
                format!(" ({} [{}] > {:.6e})", self.filter.as_ref().unwrap().name, self.filter.as_ref().unwrap().unit, self.min)
            } else if self.min.is_infinite() && self.max.is_finite() {
                format!(" ({} [{}] < {:.6e})", self.filter.as_ref().unwrap().name, self.filter.as_ref().unwrap().unit, self.max)
            } else {
                format!(" ({:.6e} < {} [{}] < {:.6e})", self.min, self.filter.as_ref().unwrap().name, self.filter.as_ref().unwrap().unit, self.max)
            }
        } else {
            "".to_owned()
        };
        let wstr = if self.weight.name != "unit" {
            format!(" ({}-weighted)", self.weight.name)
        } else {
            "".to_owned()
        };
        write!(f, "{}: {} {}{}{} = {:.6e} [{}]", self.name, self.op, self.variable.name, wstr, bounds, self.val, self.variable.unit)
    }
}

/// Calculates the value given by a function of constant values.
pub struct StatsExpression {
    name: String,
    value: f64,
    formula: Option<String>,
    unit: String,
}

impl StatsExpression {
    /// Parses a string representation of a stats expression. If no unit is given,
    /// defaults to '1' (dimensionless). If a formula is provided, it must contain
    /// no whitespace.
    /// `
    ///     [name] [expression]
    ///     [name] [expression] [unit]
    ///     [name]`formula [expression]
    ///     [name]`formula [expression] [unit]
    /// `
    pub fn load<F: Fn(&str) -> Option<f64>>(spec: &str, parser: F) -> Result<Self, OutputError> {
        let vstr: Vec<&str> = spec.split_whitespace().collect();
        if vstr.len() < 2 {
            return Err(OutputError::conversion(spec, "stats expression"));
        }
        else {
            let (exprname, form) = if vstr[0].contains("`") {
                (vstr[0].split("`").collect::<Vec<&str>>()[0].to_owned(),
                Some(vstr[1].to_owned())
                )
            }
            else {
                (vstr[0].to_owned(), None)
            };

            Ok(StatsExpression {
                name: exprname,
                value: parser(&vstr[1].to_owned()).unwrap(),
                formula: form,
                unit: vstr.get(2).map_or("1", |&s| s).to_owned()
            })
        }
    }
}

impl fmt::Display for StatsExpression {
    /// Formats the stats expression as a string.
    /// `
    ///     "- expr quantum_chi = 0.1 [1]"
    ///     "- expr synchcut (0.44*initial_gamma*me_MeV*chi) = 72.34 [MeV]"
    /// `
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(formula) = &self.formula {
            write!(f, "expr: {} ({}) = {:.6e} [{}]", self.name, formula, self.value, self.unit)
        }
        else {
            write!(f, "expr: {} = {:.6e} [{}]", self.name, self.value, self.unit)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsing() {
        use evalexpr::*;
        let ctx = context_map! {
            "a" => 2.0,
        }.unwrap();
        let parser = |s: &str| -> Option<f64> {
            eval_number_with_context(s, &ctx).ok()
        };

        let test = "mean energy in (1.0; auto)";
        let stat = SummaryStatistic::load(test, &parser).unwrap();
        println!("Got stat = {}", stat);
        assert_eq!(stat.op, Reduction::Mean);
        assert_eq!(stat.min, 1.0);
        assert_eq!(stat.max, std::f64::INFINITY);

        let test = "variance angle_x`energy";
        let stat = SummaryStatistic::load(test, &parser).unwrap();
        println!("Got stat = {}", stat);
        assert_eq!(stat.op, Reduction::Variance);
        assert!(stat.filter.is_none());
        assert!(stat.weight.name == "energy");
        assert_eq!(stat.min, std::f64::NEG_INFINITY);
        assert_eq!(stat.max, std::f64::INFINITY);

        let test = "frac pelican in (0.0; 1.0)";
        let stat = SummaryStatistic::load(test, &parser);
        if let Err(ref e) = stat {
            println!("Got stat = {}", e);
        }
        assert!(stat.is_err());

        let test = "frac number for p_perp in (auto; 100.0)";
        let stat = SummaryStatistic::load(test, &parser).unwrap();
        println!("Got stat = {}", stat);
        assert_eq!(stat.op, Reduction::Fraction);
        assert!(stat.filter.is_some());
        assert_eq!(stat.min, std::f64::NEG_INFINITY);
        assert_eq!(stat.max, 100.0);

        let test = "total r_perp for p^- in (a / (1.0 + a); 1000.0)";
        let stat = SummaryStatistic::load(test, &parser).unwrap();
        println!("Got stat = {}", stat);
        assert_eq!(stat.op, Reduction::Total);
        assert!(stat.filter.is_some());
        assert_eq!(stat.min, 2.0 / 3.0);
        assert_eq!(stat.max, 1000.0);

        let test = "circmean angle_x`energy";
        let stat = SummaryStatistic::load(test, &parser).unwrap();
        println!("Got stat = {}", stat);
        assert_eq!(stat.op, Reduction::CircMean);
        assert!(stat.filter.is_none());
        assert!(stat.weight.name == "energy");
        assert_eq!(stat.min, std::f64::NEG_INFINITY);
        assert_eq!(stat.max, std::f64::INFINITY);

        let test = "circvar angle_x`energy";
        let stat = SummaryStatistic::load(test, &parser).unwrap();
        println!("Got stat = {}", stat);
        assert_eq!(stat.op, Reduction::CircVar);
        assert!(stat.filter.is_none());
        assert!(stat.weight.name == "energy");
        assert_eq!(stat.min, std::f64::NEG_INFINITY);
        assert_eq!(stat.max, std::f64::INFINITY);

        let test = "cstd angle_x`energy";
        let stat = SummaryStatistic::load(test, &parser).unwrap();
        println!("Got stat = {}", stat);
        assert_eq!(stat.op, Reduction::CircStdDev);
        assert!(stat.filter.is_none());
        assert!(stat.weight.name == "energy");
        assert_eq!(stat.min, std::f64::NEG_INFINITY);
        assert_eq!(stat.max, std::f64::INFINITY);

        let test = "circvar energy";
        let stat = SummaryStatistic::load(test, &parser);
        if let Err(ref e) = stat {
            println!("Got stat = {}", e);
        }
        assert!(stat.is_err());
    }

    #[test]
    fn parse_expr() {
        use evalexpr::*;
        let ctx = context_map! {
            "a" => 1.0,
            "b" => 5.0,
        }.unwrap();
        let parser = |s: &str| -> Option<f64> {
            eval_number_with_context(s, &ctx).ok()
        };

        let test = "test a*b";
        let spec = StatsExpression::load(test, &parser).unwrap();
        println!("Got stats expression -> {}", spec);
        assert!(spec.name == "test");
        assert!(spec.formula.is_none());
        assert_eq!(spec.value, 5.0);
        assert!(spec.unit == "1");

        let test = "test`formula a*b";
        let spec = StatsExpression::load(test, &parser).unwrap();
        println!("Got stats expression -> {}", spec);
        assert!(spec.name == "test");
        assert!(spec.formula.unwrap() == "a*b");
        assert_eq!(spec.value, 5.0);
        assert!(spec.unit == "1");

        let test = "test a*b mm";
        let spec = StatsExpression::load(test, &parser).unwrap();
        println!("Got stats expression -> {}", spec);
        assert!(spec.name == "test");
        assert!(spec.formula.is_none());
        assert_eq!(spec.value, 5.0);
        assert!(spec.unit == "mm");

        let test = "test`formula a*b mm";
        let spec = StatsExpression::load(test, &parser).unwrap();
        println!("Got stats expression -> {}", spec);
        assert!(spec.name == "test");
        assert!(spec.formula.unwrap() == "a*b");
        assert_eq!(spec.value, 5.0);
        assert!(spec.unit == "mm");

    }
}