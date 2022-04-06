//! Creates and prints particle distribution functions
//! and other output

use std::fmt;
use std::str::FromStr;

#[cfg(feature = "with-mpi")]
use mpi::traits::*;
#[cfg(not(feature = "with-mpi"))]
use crate::no_mpi::*;

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

#[cfg(feature = "hdf5-output")]
mod binary;
#[cfg(feature = "hdf5-output")]
pub use binary::*;

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

impl FromStr for DistributionFunction {
    type Err = OutputError;

    fn from_str(spec: &str) -> Result<Self, Self::Err> {
        // break into substrings, separated by colons
        let mut ss: Vec<&str> = spec.split(':').collect();

        // if the final string is bracketed AND there are at least
        // two substrings, that final string might be (bspec; hspec)
        let (bspec, hspec, weight) = if ss.len() >= 2 && ss.last().unwrap().starts_with('(') && ss.last().unwrap().ends_with(')') {
            let last = ss.pop().unwrap().trim_start_matches('(').trim_end_matches(')');
            // break this into substrings, separated by ';'
            let last: Vec<&str> = last.split(';').collect();
            match last.len() {
                1 => (BinSpec::Automatic, HeightSpec::Density, last[0]),
                2 => (last[0].into(), HeightSpec::Density, last[1]),
                _ => (BinSpec::Automatic, HeightSpec::Density, "weight"),
            }
        } else {
            (BinSpec::Automatic, HeightSpec::Density, "weight")
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
            _ => None,
        };

        if funcs.iter().all(Option::is_some) && weight_function.is_some() {
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
            })
        } else {
            Err(OutputError::Conversion(spec.to_owned(), "distribution function".to_owned()))
        }
    }
}

impl DistributionFunction {
    pub fn write(&self, world: &impl Communicator, pt: &[Particle], us: &UnitSystem, prefix: &str, file_format: FileFormat) -> Result<(),OutputError> {
        let units: Vec<Unit> = self.func_types.iter()
            .map(|t| match t {
                    ParticleOutputType::Dimensionless => Unit::new(1.0, "1"),
                    ParticleOutputType::Angle => Unit::new(1.0, "rad"),
                    ParticleOutputType::Length => us.length.clone(),
                    ParticleOutputType::Energy => us.energy.clone(),
                    ParticleOutputType::Momentum => us.momentum.clone(),
                })
            .collect();

        let (hgram, suffix) = match self.dim {
            1 => {
                let hgram = Histogram::generate_1d(
                    world,
                    pt,
                    &|pt| {self.funcs[0](pt).convert(&units[0])},
                    &self.fweight,
                    &self.names[0],
                    units[0].name(),
                    self.bspec,
                    self.hspec
                );
                let mut suffix = format!("_{}", self.names[0]);
                if self.weight != "weight" && self.weight != "auto" {
                    suffix.push('_');
                    suffix.push_str(&self.weight);
                }
                if self.bspec == BinSpec::LogScaled {
                    suffix.push_str("_log");
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
                    [&self.names[0], &self.names[1]],
                    [units[0].name(), units[1].name()],
                    [self.bspec; 2],
                    self.hspec
                );
                let mut suffix = format!("_{}-{}", self.names[0], self.names[1]);
                if self.weight != "weight" && self.weight != "auto" {
                    suffix.push('_');
                    suffix.push_str(&self.weight);
                }
                if self.bspec == BinSpec::LogScaled {
                    suffix.push_str("_log");
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
        let dstr = DistributionFunction::from_str(test);
        println!("{:?}", dstr);
        assert!(dstr.is_ok());
    }
}