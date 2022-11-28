//! Converting between units systems

use std::ops::Mul;
use std::default::Default;
use std::str::FromStr;
use crate::constants::*;

pub trait HasUnit {
    type Output;
    /// Converts a quantity from the default system of units
    /// to the target unit
    fn convert(self, unit: &Unit) -> Self::Output;

    /// Converts a quantity from SI to the default system of units
    fn from_si(self, unit: &Unit) -> Self::Output;
}

impl<T> HasUnit for T where T: Mul<f64> {
    type Output = T::Output;
    fn convert(self, unit: &Unit) -> Self::Output {
        self * unit.scale
    }

    fn from_si(self, unit: &Unit) -> Self::Output {
        self * unit.scale.recip()
    }
}

/// A container for the chosen system of units
#[derive(Debug)]
pub struct UnitSystem {
    pub length: Unit,
    pub energy: Unit,
    pub momentum: Unit,
}

impl UnitSystem {
    /// A High-Energy-Physics compatible system of units
    pub fn hep() -> Self {
        Self {
            length: Unit::mm(),
            energy: Unit::GeV(),
            momentum: Unit::GeV_c(),
        }
    }

    /// The SI system of units
    pub fn si() -> Self {
        Self {
            length: Unit::m(),
            energy: Unit::J(),
            momentum: Unit::kg_m_s(),
        }
    }
}

impl Default for UnitSystem {
    fn default() -> Self {
        Self {
            length: Unit::m(),
            energy: Unit::MeV(),
            momentum: Unit::MeV_c(),
        }
    }
}

/// Represents a dimensionalful quantity in the default
/// system of units (MeV-m)
#[derive(Debug, Clone)]
pub struct Unit {
    scale: f64,
    name: String,
}

impl Unit {
    /// Creates a new unit with given name.
    /// 'scale' should be a number by which the
    /// quantity, in its default unit, must be
    /// multiplied to convert to the new unit.
    /// For example, energy is measured in MeV
    /// by default - to convert to joules, 'scale'
    /// should be 1000 * elementary_charge
    pub fn new(scale: f64, name: &str) -> Self {
        Self {scale, name: name.to_owned()}
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Metres (length unit)
    pub fn m() -> Self {
        Self::new(1.0, "m")
    }

    /// Millimetres (length unit)
    pub fn mm() -> Self {
        Self::new(1.0e3, "mm")
    }

    /// Micrometres (length unit)
    pub fn um() -> Self {
        Self::new(1.0e6, "um")
    }

    /// Joules (energy unit)
    #[allow(non_snake_case)]
    pub fn J() -> Self {
        Self::new(1.0e6 * ELEMENTARY_CHARGE, "J")
    }

    /// MeV (energy unit)
    #[allow(non_snake_case)]
    pub fn MeV() -> Self {
        Self::new(1.0, "MeV")
    }

    /// GeV (energy unit)
    #[allow(non_snake_case)]
    pub fn GeV() -> Self {
        Self::new(1.0e-3, "GeV")
    }

    /// kg/m/s (momentum unit)
    pub fn kg_m_s() -> Self {
        Self::new(1.0e6 * ELEMENTARY_CHARGE / SPEED_OF_LIGHT, "kg/m/s")
    }

    /// MeV/c (momentum unit)
    #[allow(non_snake_case)]
    pub fn MeV_c() -> Self {
        Self::new(1.0, "MeV/c")
    }

    /// GeV/c (momentum unit)
    #[allow(non_snake_case)]
    pub fn GeV_c() -> Self {
        Self::new(1.0e-3, "GeV/c")
    }
}

impl FromStr for Unit {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            // length
            "um" | "micron" => Ok(Unit::um()),
            "mm" => Ok(Unit::mm()),
            "m" => Ok(Unit::m()),
            // energy
            "J" => Ok(Unit::J()),
            "GeV" => Ok(Unit::GeV()),
            "MeV" => Ok(Unit::MeV()),
            // momentum
            "kg/m/s" => Ok(Unit::kg_m_s()),
            "GeV/c" => Ok(Unit::GeV_c()),
            "MeV/c" => Ok(Unit::MeV_c()),
            _ => Err("unit not recognised"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::*;

    #[test]
    fn unit_conversion() {
        let p = FourVector::new(1000.0, 0.0, 0.0, 1000.0);
        let unit: Unit = "kg/m/s".parse().unwrap();
        let p2 = p.convert(&unit);
        println!("p[0] = {:.3e} MeV/c = {:.3e} {}", p[0], p2[0], unit.name());
        assert_eq!(p[0] * 1.0e6 * ELEMENTARY_CHARGE / SPEED_OF_LIGHT, p2[0]);

        let x = 1.5e-6;
        let unit: Unit = "mm".parse().unwrap();
        let x2 = x.convert(&unit);
        println!("x = {:.3e} m = {:.3e} {}", x, x2, unit.name());
        assert_eq!(x * 1.0e3, x2);
    }

}
