//! Field properties and parameters

#[cfg(feature = "hdf5-output")]
use hdf5_writer::{Hdf5Type, Datatype};

use crate::geometry::{FourVector, StokesVector};
use super::FieldData;

/// The polarization of an electromagnetic wave
#[allow(unused)]
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum Polarization {
    Linear = 0,
    Circular = 1,
}

#[cfg(feature = "hdf5-output")]
impl Hdf5Type for Polarization {
    fn new() -> Datatype {
        unsafe { Datatype::enumeration(&[
            ("linear", Polarization::Linear as u8),
            ("circular", Polarization::Circular as u8),
        ])}
    }
}

/// Temporal profile of the laser
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum Envelope {
    CosSquared = 0,
    Flattop = 1,
    Gaussian = 2,
}

#[cfg(feature = "hdf5-output")]
impl Hdf5Type for Envelope {
    fn new() -> Datatype {
        unsafe { Datatype::enumeration(&[
            ("cos^2", Envelope::CosSquared as u8),
            ("flattop", Envelope::Flattop as u8),
            ("gaussian", Envelope::Gaussian as u8),
        ])}
    }
}

/// How to propagate the charged particle through an EM field
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum EquationOfMotion {
    Lorentz,
    LandauLifshitz,
    ModifiedLandauLifshitz,
}

impl EquationOfMotion {
    pub fn includes_rr(&self) -> bool {
        match self {
            EquationOfMotion::LandauLifshitz | EquationOfMotion::ModifiedLandauLifshitz => true,
            EquationOfMotion::Lorentz => false,
        }
    }
}

/// How to treat radiation emission
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum RadiationMode {
    Quantum,
    Classical,
}

#[derive(Copy, Clone)]
pub struct RadiationEvent {
    /// The normalized momentum of the emitted photon
    pub k: FourVector,
    /// The normalized momentum of the recoiling electron/positron
    pub u_prime: FourVector,
    /// The polarization of the emitted photon
    pub pol: StokesVector,
    /// The effective a0 of the interaction
    pub a_eff: f64,
    /// The quantum parameter of the parent particle
    pub chi: f64,
    /// The energy absorbed from the field during the interaction,
    /// in units of the electron rest energy
    pub absorption: f64,
}

#[derive(Copy, Clone)]
pub struct PairCreationEvent {
    /// The normalized momentum of the electron
    pub u_e: FourVector,
    /// The normalized momentum of the positron
    pub u_p: FourVector,
    /// The fraction of the photon that has decayed
    pub frac: f64,
    /// The effective a0 of the interaction
    pub a_eff: f64,
    /// The quantum parameter of the parent particle
    pub chi: f64,
    /// The energy absorbed from the field during the interaction,
    /// in units of the electron rest energy
    pub absorption: f64,
}

/// Describes how the target electromagnetic field should be modelled
pub enum FieldStructure {
    Analytical {
        params: LaserParameters,
    },
    Numerical {
        data: FieldData,
    },
}

impl FieldStructure {
    pub fn params(&self) -> LaserParameters {
        match self {
            FieldStructure::Analytical { params } => *params,
            FieldStructure::Numerical { data } => data.params(),
        }
    }

    pub fn is_numerical(&self) -> Option<&FieldData> {
        match self {
            FieldStructure::Analytical { .. } => None,
            FieldStructure::Numerical { data } => Some(data),
        }
    }
}

#[derive(Copy, Clone)]
pub struct LaserParameters {
    pub a0: f64,
    pub wavelength: f64,
    pub pol: Polarization,
    pub pol_angle: f64,
    pub focusing: bool,
    pub waist: f64,
    pub envelope: Envelope,
    pub n_cycles: f64,
    pub chirp_b: f64,
}
