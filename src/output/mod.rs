//! Creates and prints particle distribution functions
//! and other output

use std::fmt;

#[cfg(feature = "with-mpi")]
use mpi::traits::*;
#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;

use crate::particle::*;

mod error;
use error::*;

mod hgram;
use hgram::*;

mod functions;

mod stats;
pub use stats::*;

mod units;
pub use units::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FileFormat {
    PlainText,
    Fits,
}

//type ParticleOutput = Box<dyn Fn(&Particle) -> f64>;
type ParticleOutput = fn(&Particle) -> f64;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ParticleOutputType {
    Dimensionless,
    Angle,
    Length,
    Energy,
    Momentum,
}

impl ParticleOutputType {
    fn into_unit(self, us: &UnitSystem) -> Unit {
        match self {
            ParticleOutputType::Dimensionless => Unit::new(1.0, "1"),
            ParticleOutputType::Angle => Unit::new(1.0, "rad"),
            ParticleOutputType::Length => us.length.clone(),
            ParticleOutputType::Energy => us.energy.clone(),
            ParticleOutputType::Momentum => us.momentum.clone(),
        }
    }
}

/// Represents a range cut on the particle output.
/// A given particle `pt` will only contribute if
/// `min <= func(pt) < max`.
/// The default Filter accepts all particles.
struct Filter {
    func: Option<ParticleOutput>,
    name: String,
    min: f64,
    max: f64,
}

impl Filter {
    fn from_str<F: Fn(&str) -> Option<f64>>(s: &str, parser: F) -> Result<Self, OutputError> {
        // just in case
        let error = || OutputError::conversion(s, "filter");

        // s is a string like 'op in min, max'
        let (prefix, suffix) = s.split_once(',').ok_or_else(error)?;

        let mut word = prefix.split_whitespace();

        // First word must be a ParticleOutput
        let (func, func_type, name) = word.next()
            .and_then(|name| {
                let name = match name {
                    "p^-" | "p-" => "p_minus",
                    "p^+" | "p+" => "p_plus",
                    _ => name,
                };
                functions::identify(name).map(|(f, ftype)| (f, ftype, name))
            })
            .ok_or_else(error)?;

        // Next word must be 'in'
        word.next()
            .map(|s| s == "in")
            .ok_or_else(error)?;

        // All remaining words (up to the comma) form a math expression
        let min_expr: String = word.collect();
        let min = if min_expr.as_str() == "auto" {
            std::f64::NEG_INFINITY
        } else {
            parser(&min_expr).ok_or_else(error)?
        };

        // Everything after the comma should be a math expression as well
        let max = if suffix.trim() == "auto" {
            std::f64::INFINITY
        } else {
            parser(suffix).ok_or_else(error)?
        };

        // f(pt) returns a quantity in the default unit system, whereas min, max are in SI
        let unit = func_type.into_unit(&UnitSystem::si());
        let min = min.from_si(&unit);
        let max = max.from_si(&unit);

        let filter = Self {
            func: Some(func),
            name: name.to_owned(),
            min,
            max,
        };

        // println!("got filter {{..., {:.3e}, {:.3e}}} from \"{}\"", filter.min, filter.max, s);

        Ok(filter)
    }

    fn accepts(&self, pt: &Particle) -> bool {
        if let Some(f) = self.func {
            f(pt) >= self.min && f(pt) < self.max
        } else {
            true
        }
    }
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            func: None,
            name: "no".to_owned(),
            min: std::f64::NEG_INFINITY,
            max: std::f64::INFINITY,
        }
    }
}

pub struct DistributionFunction {
    dim: usize,
    bspec: BinSpec,
    hspec: HeightSpec,
    names: Vec<String>,
    //units: Vec<String>,
    weight: String,
    fweight: ParticleOutput,
    funcs: Vec<ParticleOutput>,
    func_types: Vec<ParticleOutputType>,
    filter: Filter,
}

impl fmt::Debug for DistributionFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "dim: {}", self.dim)?;
        writeln!(f, "bspec: {}", self.bspec)?;
        writeln!(f, "hspec: {}", self.hspec)?;
        writeln!(f, "names: {:?}", self.names)?;
        //writeln!(f, "units: {:?}", self.units)?;
        writeln!(f, "funcs = <>, len = {}", self.funcs.len())?;
        Ok(())
    }
}

impl DistributionFunction {
    pub fn load<F: Fn(&str) -> Option<f64>>(spec: &str, parser: F) -> Result<Self, OutputError> {
        // break into substrings, separated by colons
        let mut ss: Vec<&str> = spec.split(':').collect();

        // if the final string is bracketed AND there are at least
        // two substrings, that final string might be (bspec; hspec; filter)
        let (bspec, hspec, weight, filter) = if ss.len() >= 2 && ss.last().unwrap().starts_with('(') && ss.last().unwrap().ends_with(')') {
            let last = ss.pop().unwrap().trim_start_matches('(').trim_end_matches(')');
            // break this into substrings, separated by ';'
            let last: Vec<&str> = last.split(';').collect();
            match last.len() {
                1 => (BinSpec::Automatic, HeightSpec::Density, last[0].trim(), None),
                2 => (last[0].trim().into(), HeightSpec::Density, last[1].trim(), None),
                3 => (last[0].trim().into(), HeightSpec::Density, last[1].trim(), Some(last[2])),
                _ => (BinSpec::Automatic, HeightSpec::Density, "weight", None),
            }
        } else {
            (BinSpec::Automatic, HeightSpec::Density, "weight", None)
        };

        // Convert strings to closures, associated units and names.

        let (funcs, func_types): (Vec<Option<ParticleOutput>>, Vec<Option<ParticleOutputType>>) = ss
            .iter()
            .map(|&name| {
                if let Some(v) = functions::identify(name) {
                    (Some(v.0), Some(v.1))
                } else {
                    (None, None)
                }
            })
            .unzip();

        let names: Vec<Option<&str>> = ss
            .iter()
            .map(|&s| {
                match s {
                    "p^-" | "p-" => Some("p_minus"),
                    "p^+" | "p+" => Some("p_plus"),
                    _ => Some(s),
            }})
            .collect();

        let weight_function = match weight {
            "energy" => Some(functions::weighted_by_energy as ParticleOutput),
            "weight" | "auto" => Some(functions::weighted_by_number as ParticleOutput),
            "pol_x" => Some(functions::weighted_by_pol_x as ParticleOutput),
            "pol_y" => Some(functions::weighted_by_pol_y as ParticleOutput),
            "helicity" => Some(functions::weighted_by_helicity as ParticleOutput),
            _ => None,
        };

        let filter = match filter {
            Some(s) => Filter::from_str(s, parser),
            None => Ok(Filter::default())
        };

        if funcs.iter().all(Option::is_some) && weight_function.is_some() && filter.is_ok() {
            // successfully obtained a distribution function!
            Ok(DistributionFunction {
                dim: funcs.len(),
                bspec: bspec,
                hspec: hspec,
                names: names.into_iter().map(|u| u.unwrap().to_owned()).collect(),
                //units: units.into_iter().map(|u| u.unwrap().to_owned()).collect(),
                weight: weight.to_owned(),
                fweight: weight_function.unwrap(),
                funcs: funcs.into_iter().flatten().collect(),
                func_types: func_types.into_iter().map(|u| u.unwrap()).collect(),
                filter: filter.unwrap(),
            })
        } else {
            Err(OutputError::conversion(spec, "distribution function"))
        }
    }
}

impl DistributionFunction {
    pub fn write(&self, world: &impl Communicator, pt: &[Particle], us: &UnitSystem, prefix: &str, file_format: FileFormat) -> Result<(),OutputError> {
        let units: Vec<Unit> = self.func_types.iter()
            .map(|t| t.into_unit(us))
            .collect();

        let (hgram, suffix) = match self.dim {
            1 => {
                let hgram = Histogram::generate_1d(
                    world,
                    pt,
                    &|pt| {self.funcs[0](pt).convert(&units[0])},
                    &self.fweight,
                    &|pt| self.filter.accepts(pt),
                    &self.names[0],
                    units[0].name(),
                    self.bspec,
                    self.hspec
                );
                let mut suffix = format!("_{}", self.names[0]);
                if self.weight != "weight" && self.weight != "auto" {
                    suffix.push('_');
                    suffix.push_str(&self.weight);
                    suffix.push_str("-weight");
                }
                if self.bspec == BinSpec::LogScaled {
                    suffix.push_str("_log");
                }
                if self.filter.func.is_some() {
                    suffix.push('_');
                    suffix.push_str(&self.filter.name);
                    suffix.push_str("-cut");
                }
                (hgram, suffix)
            },
            2 => {
                let hgram = Histogram::generate_2d(
                    world,
                    pt,
                    &|pt| {self.funcs[0](pt).convert(&units[0])},
                    &|pt| {self.funcs[1](pt).convert(&units[1])},
                    &self.fweight,
                    &|pt| self.filter.accepts(pt),
                    [&self.names[0], &self.names[1]],
                    [units[0].name(), units[1].name()],
                    [self.bspec; 2],
                    self.hspec
                );
                let mut suffix = format!("_{}-{}", self.names[0], self.names[1]);
                if self.weight != "weight" && self.weight != "auto" {
                    suffix.push('_');
                    suffix.push_str(&self.weight);
                    suffix.push_str("-weight");
                }
                if self.bspec == BinSpec::LogScaled {
                    suffix.push_str("_log");
                }
                if self.filter.func.is_some() {
                    suffix.push('_');
                    suffix.push_str(&self.filter.name);
                    suffix.push_str("-cut");
                }
                (hgram, suffix)
            },
            _ => return Err(OutputError::Dimension(self.dim))
        };

        if world.rank() == 0 && hgram.is_some() {
            let filename = prefix.to_owned() + &suffix;
            let res = match file_format {
                FileFormat::PlainText => hgram.unwrap().write_plain_text(&filename),
                FileFormat::Fits => hgram.unwrap().write_fits(&filename),
            };
            res.map_err(|_e| OutputError::Write(filename.to_owned()))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_ospec() {
        let test = "angle_x:angle_y:(100;auto)";
        let parser = |_: &str| {None};
        let dstr = DistributionFunction::load(test, parser);
        println!("{:?}", dstr);
        assert!(dstr.is_ok());
    }
}