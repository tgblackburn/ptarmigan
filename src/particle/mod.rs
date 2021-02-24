//! Particles

use std::fmt;

#[cfg(feature = "with-mpi")]
use {mpi::traits::*, mpi::datatype::{UserDatatype, SystemDatatype}, memoffset::*};

use crate::constants::*;
use crate::geometry::*;

#[derive(Copy,Clone,PartialEq,Eq)]
#[repr(u8)]
pub enum Species {
    Electron,
    Positron,
    Photon,
}

#[cfg(feature = "with-mpi")]
unsafe impl Equivalence for Species {
    type Out = SystemDatatype;
    fn equivalent_datatype() -> Self::Out {
        u8::equivalent_datatype()
    }
}

/// A elementary particle is defined by its position
/// and momentum
#[derive(Copy,Clone)]
pub struct Particle {
    species: Species,
    r: [FourVector; 2],
    u: [FourVector; 2],
    optical_depth: f64,
    payload: f64,
    interaction_count: f64,
    weight: f64,
    id: u64,
}

#[cfg(feature = "with-mpi")]
unsafe impl Equivalence for Particle {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        let blocklengths = [1, 2, 2, 1, 1, 1, 1, 1];
        let displacements = [
            offset_of!(Particle, species) as mpi::Address,
            offset_of!(Particle, r) as mpi::Address,
            offset_of!(Particle, u) as mpi::Address,
            offset_of!(Particle, optical_depth) as mpi::Address,
            offset_of!(Particle, payload) as mpi::Address,
            offset_of!(Particle, interaction_count) as mpi::Address,
            offset_of!(Particle, weight) as mpi::Address,
            offset_of!(Particle, id) as mpi::Address,
        ];
        let types: [&dyn Datatype; 8] = [
            &Species::equivalent_datatype(),
            &FourVector::equivalent_datatype(),
            &FourVector::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &f64::equivalent_datatype(),
            &u64::equivalent_datatype(),
        ];
        UserDatatype::structured(8, &blocklengths, &displacements, &types)
    }
}

impl fmt::Display for Particle {
    //"E (GeV) x (um) y (um) z (um) beta_x beta_y beta_z PDG_NUM MP_Wgt MP_ID t xi"
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let pdg_num = match self.species {
            Species::Electron => 11,
            Species::Positron => -11,
            Species::Photon => 22,
        };
        let p = 1.0e-3 * self.momentum(); // units of GeV
        //let v = p / p[0];
        let v = p;
        write!(f,
            "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{}\t{:.6e}\t{}\t{:.6e}\t{:.6e}",
            p[0], // E (GeV)
            1.0e6 * self.r[1][1], 1.0e6 * self.r[1][2], 1.0e6 * self.r[1][3], // x y z (um)
            v[1], v[2], v[3], // p (GeV/c) OR beta (1)
            pdg_num, // PDG_NUM
            self.weight, // MP_Wgt
            self.id, // MP_ID
            1.0e6 * self.r[1][0], // t (um/c)
            self.payload, // a at creation
        )
    }
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
            u: [u; 2],
            optical_depth: std::f64::INFINITY,
            payload: 0.0,
            interaction_count: 0.0,
            weight: 1.0,
            id: 0,
        }
    }

    /// Updates the particle position
    pub fn with_position(&mut self, r: FourVector) -> Self {
        self.r[1] = r;
        *self
    }

    /// Updates the particle normalized momentum
    pub fn with_normalized_momentum(&mut self, u: FourVector) -> Self {
        self.u[1] = u;
        *self
    }

    /// Updates the particle optical depth
    pub fn with_optical_depth(&mut self, tau: f64) -> Self {
        self.optical_depth = tau;
        *self
    }

    /// Updates the particle weight
    pub fn with_weight(&mut self, weight: f64) -> Self {
        self.weight = weight;
        *self
    }

    /// Number of real particles this particles represents
    pub fn weight(&self) -> f64 {
        self.weight
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
        self.u[1]
    }

    /// The particle momentum, in units of MeV
    pub fn momentum(&self) -> FourVector {
        match self.species {
            Species::Electron | Species::Positron | Species::Photon => {
                ELECTRON_MASS_MEV * self.u[1]
            }
        }
    }

    /// The particle momentum at creation, in units of MeV
    #[allow(unused)]
    pub fn initial_momentum(&self) -> FourVector {
        match self.species {
            Species::Electron | Species::Positron | Species::Photon => {
                ELECTRON_MASS_MEV * self.u[0]
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

    /// Loads something that will be tracked with the particle
    /// and may be read by the output routines.
    /// Overwrites whatever has already been written.
    pub fn with_payload(&mut self, value: f64) -> Self {
        self.payload = value;
        *self
    }

    pub fn payload(&self) -> f64 {
        self.payload
    }

    /// Increments the particle interaction count by the given
    /// delta.
    pub fn update_interaction_count(&mut self, delta: f64) -> Self {
        self.interaction_count = self.interaction_count + delta;
        *self
    }

    pub fn interaction_count(&self) -> f64 {
        self.interaction_count
    }

    /// In the default coordinate system, the laser propagates towards
    /// positive z, and it is the beam that is rotated when a finite collision
    /// angle is requested. This function transforms the particle momenta
    /// and positions such that the positive z axis points along the beam
    /// propagation axis instead.
    pub fn to_beam_coordinate_basis(&self, collision_angle: f64) -> Self {
        let theta = std::f64::consts::PI - collision_angle;

        let u0 = ThreeVector::from(self.u[0]).rotate_around_y(theta);
        let u0 = FourVector::new(self.u[0][0], u0[0], u0[1], u0[2]);

        let u = ThreeVector::from(self.u[1]).rotate_around_y(theta);
        let u = FourVector::new(self.u[1][0], u[0], u[1], u[2]);

        let r0 = ThreeVector::from(self.r[0]).rotate_around_y(theta);
        let r0 = FourVector::new(self.r[0][0], r0[0], r0[1], r0[2]);

        let r = ThreeVector::from(self.r[1]).rotate_around_y(theta);
        let r = FourVector::new(self.r[1][0], r[0], r[1], r[2]);

        Particle {
            species: self.species,
            r: [r0, r],
            u: [u0, u],
            optical_depth: self.optical_depth,
            payload: self.payload,
            interaction_count: self.interaction_count,
            weight: self.weight,
            id: self.id,
        }
    }

    /// Updates the particle ID
    pub fn with_id(&mut self, id: u64) -> Self {
        self.id = id;
        *self
    }

    /// ID of this particle
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Shower {
    pub fn multiplicity(&self) -> usize {
        self.secondaries.len()
    }
}