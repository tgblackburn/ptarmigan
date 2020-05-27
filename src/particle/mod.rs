//! Particles

use crate::constants::*;
use crate::geometry::*;

#[derive(Copy,Clone,PartialEq,Eq)]
pub enum Species {
    Electron,
    Positron,
    Photon,
}

/// A elementary particle is defined by its position
/// and momentum
#[derive(Copy,Clone)]
pub struct Particle {
    species: Species,
    r: [FourVector; 2],
    u: FourVector,
    optical_depth: f64,
}

/// A shower, or cascade, consists of the primary
/// particle and all the secondaries it produces by
/// various QED processes
pub struct Shower {
    pub primary: Particle,
    pub secondaries: Vec<Particle>,
}

impl Particle {
    /// Creates a new particle of the given species, at
    /// the given position
    pub fn create(species: Species, r: FourVector) -> Self {
        let u = match species {
            Species::Electron | Species::Positron => FourVector::new(1.0, 0.0, 0.0, 0.0),
            Species::Photon => FourVector::new(0.0, 0.0, 0.0, 0.0),
        };

        Particle {
            species,
            r: [r; 2],
            u,
            optical_depth: std::f64::INFINITY,
        }
    }

    /// Updates the particle position
    pub fn with_position(&mut self, r: FourVector) -> Self {
        self.r[1] = r;
        *self
    }

    /// Updates the particle normalized momentum
    pub fn with_normalized_momentum(&mut self, u: FourVector) -> Self {
        self.u = u;
        *self
    }

    /// Updates the particle optical depth
    pub fn with_optical_depth(&mut self, tau: f64) -> Self {
        self.optical_depth = tau;
        *self
    }

    /// The charge-to-mass ratio of the particle
    /// species, in units of C/kg
    pub fn charge_to_mass_ratio(&self) -> f64 {
        match self.species {
            Species::Electron => ELECTRON_CHARGE / ELECTRON_MASS,
            Species::Positron => -ELECTRON_CHARGE / ELECTRON_MASS,
            Species::Photon => 0.0,
        }
    }

    /// The particle normalized momentum
    pub fn normalized_momentum(&self) -> FourVector {
        self.u
    }

    /// The particle momentum, in units of MeV/c
    pub fn momentum(&self) -> FourVector {
        match self.species {
            Species::Electron | Species::Positron | Species::Photon => {
                ELECTRON_MASS_MEV * self.u
            }
        }
    }

    /// Returns the four-position at which the
    /// particle was created, in units of metres
    #[allow(unused)]
    pub fn was_created_at(&self) -> FourVector {
        self.r[0]
    }

    /// Returns the current four-position of the
    /// particle, in units of metres
    pub fn position(&self) -> FourVector {
        self.r[1]
    }
}

impl Shower {
    pub fn multiplicity(&self) -> usize {
        self.secondaries.len()
    }
}