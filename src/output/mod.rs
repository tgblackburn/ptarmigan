//! Creates and prints particle distribution functions
//! and other output

use std::fmt;
use std::error::Error;
use std::str::FromStr;

use mpi::traits::*;
use crate::constants::*;
use crate::particle::*;

mod hgram;
use hgram::*;

pub enum OutputError {
    Conversion(String),
    Dimension(usize),
    Write(String),
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OutputError::Conversion(s) => write!(f, "'{}' does not specify a valid distribution function", s),
            OutputError::Dimension(d) => write!(f, "requested dimension was {}, only 1 and 2 are supported", d),
            OutputError::Write(s) => writeln!(f, "failed to write histogram to '{}'", s),
        }
    }
}

impl fmt::Debug for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl Error for OutputError {}

type ParticleOutput = Box<dyn Fn(&Particle) -> f64>;

pub struct DistributionFunction {
    dim: usize,
    bspec: BinSpec,
    hspec: HeightSpec,
    names: Vec<String>,
    units: Vec<String>,
    weight: String,
    fweight: ParticleOutput,
    funcs: Vec<ParticleOutput>,
}

impl fmt::Debug for DistributionFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "dim: {}", self.dim)?;
        writeln!(f, "bspec: {}", self.bspec)?;
        writeln!(f, "hspec: {}", self.hspec)?;
        writeln!(f, "names: {:?}", self.names)?;
        writeln!(f, "units: {:?}", self.units)?;
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

        // Possible particle outputs
        let angle_x = |pt: &Particle| {let p = pt.momentum(); p[1].atan2(-p[3])};
        let angle_y = |pt: &Particle| {let p = pt.momentum(); p[2].atan2(-p[3])};
        let px = |pt: &Particle| {let p = pt.momentum(); p[1]};
        let py = |pt: &Particle| {let p = pt.momentum(); p[2]};
        let pz = |pt: &Particle| {let p = pt.momentum(); p[3]};
        let p_perp = |pt: &Particle| {let p = pt.momentum(); p[1].hypot(p[2])};
        let p_minus = |pt: &Particle| {let p = pt.momentum(); p[0] - p[3]};
        let p_plus = |pt: &Particle| {let p = pt.momentum(); p[0] + p[3]};
        let gamma = |pt: &Particle| {let p = pt.normalized_momentum(); p[0]};
        let energy = |pt: &Particle| {let p = pt.normalized_momentum(); ELECTRON_MASS_MEV * p[0]};
        let unit_weight = |_pt: &Particle| 1.0;

        // convert strings to closures
        let funcs: Vec<Option<ParticleOutput>> = ss
            .iter()
            .map(|&s| {
                match s {
                    "angle_x" => Some(Box::new(angle_x) as ParticleOutput),
                    "angle_y" => Some(Box::new(angle_y) as ParticleOutput),
                    "px" => Some(Box::new(px) as ParticleOutput),
                    "py" => Some(Box::new(py) as ParticleOutput),
                    "pz" => Some(Box::new(pz) as ParticleOutput),
                    "p_perp" => Some(Box::new(p_perp) as ParticleOutput),
                    "p^-" | "p-" => Some(Box::new(p_minus) as ParticleOutput),
                    "p^+" | "p+" => Some(Box::new(p_plus) as ParticleOutput),
                    "gamma" => Some(Box::new(gamma) as ParticleOutput),
                    "energy" => Some(Box::new(energy) as ParticleOutput),
                    _ => None,
                }})
            .collect();

        let units: Vec<Option<&str>> = ss
            .iter()
            .map(|&s| {
                match s {
                    "gamma" => Some("1"),
                    "energy" => Some("MeV"),
                    "p^-" | "p-" | "p^+" | "p+" | "px" | "py" | "pz" | "p_perp" => Some("MeV/c"),
                    "angle_x" | "angle_y" => Some("rad"),
                    _ => None,
                }})
            .collect();

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
            "energy" => Some(Box::new(energy) as ParticleOutput),
            "weight" | "auto" => Some(Box::new(unit_weight) as ParticleOutput),
            _ => None,
        };

        if funcs.iter().all(Option::is_some) && weight_function.is_some() {
            // successfully obtained a distribution function!
            Ok(DistributionFunction {
                dim: funcs.len(),
                bspec: bspec,
                hspec: hspec,
                names: names.into_iter().map(|u| u.unwrap().to_owned()).collect(),
                units: units.into_iter().map(|u| u.unwrap().to_owned()).collect(),
                weight: weight.to_owned(),
                fweight: weight_function.unwrap(),
                funcs: funcs.into_iter().flatten().collect(),
            })
        } else {
            Err(OutputError::Conversion(spec.to_owned()))
        }
    }
}

impl DistributionFunction {
    pub fn write(&self, world: &impl Communicator, pt: &[Particle], prefix: &str) -> Result<(),OutputError> {
        let (hgram, suffix) = match self.dim {
            1 => {
                let hgram = Histogram::generate_1d(
                    world,
                    pt,
                    &self.funcs[0],
                    &self.fweight,
                    &self.names[0],
                    &self.units[0],
                    self.bspec,
                    self.hspec
                );
                let mut suffix = format!("_{}", self.names[0]);
                if self.weight != "weight" {
                    suffix.push('_');
                    suffix.push_str(&self.weight);
                }
                (hgram, suffix)
            },
            2 => {
                let hgram = Histogram::generate_2d(
                    world,
                    pt,
                    &self.funcs[0],
                    &self.funcs[1],
                    &self. fweight,
                    [&self.names[0], &self.names[1]],
                    [&self.units[0], &self.units[1]],
                    [self.bspec; 2],
                    self.hspec
                );
                let mut suffix = format!("_{}-{}", self.names[0], self.names[1]);
                if self.weight != "weight" {
                    suffix.push('_');
                    suffix.push_str(&self.weight);
                };
                (hgram, suffix)
            },
            _ => return Err(OutputError::Dimension(self.dim))
        };

        if let Some(hg) = hgram {
            let filename = prefix.to_owned() + &suffix;
            hg.write(&filename).map_err(|_e| OutputError::Write(filename.to_owned()))
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