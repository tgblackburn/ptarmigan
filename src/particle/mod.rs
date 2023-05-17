//! Particles

use std::fmt;
use std::str::FromStr;

use crate::constants::*;
use crate::geometry::*;

mod builder;
pub use builder::BeamBuilder;

mod dstr;

#[derive(Copy,Clone,PartialEq,Eq,Debug)]
#[repr(u8)]
pub enum Species {
    Electron,
    Positron,
    Photon,
}

impl FromStr for Species {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "electron" => Ok(Species::Electron),
            "positron" => Ok(Species::Positron),
            "photon" => Ok(Species::Photon),
            _ => Err(format!("requested species '{}' is not one of 'electron', 'positron' or 'photon'", s)),
        }
    }
}

impl fmt::Display for Species {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Species::Electron => write!(f, "electron"),
            Species::Positron => write!(f, "positron"),
            Species::Photon => write!(f, "photon"),
        }
    }
}

/// A elementary particle is defined by its position
/// and momentum
#[derive(Copy,Clone)]
pub struct Particle {
    species: Species,
    r: [FourVector; 2],
    u: [FourVector; 2],
    pol: StokesVector,
    optical_depth: f64,
    payload: f64,
    interaction_count: f64,
    weight: f64,
    id: u64,
    parent_id: u64,
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
            pol: StokesVector::unpolarized(),
            optical_depth: std::f64::INFINITY,
            payload: 0.0,
            interaction_count: 0.0,
            weight: 1.0,
            id: 0,
            parent_id: 0,
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

    /// Returns the lab time of the particle, in seconds
    pub fn time(&self) -> f64 {
        self.r[1][0] / SPEED_OF_LIGHT
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

        // let pol = ThreeVector::from(self.pol).rotate_around_y(theta);
        // let pol = FourVector::new(self.pol[0], pol[0], pol[1], pol[2]);

        Particle {
            species: self.species,
            r: [r0, r],
            u: [u0, u],
            pol: self.pol,
            optical_depth: self.optical_depth,
            payload: self.payload,
            interaction_count: self.interaction_count,
            weight: self.weight,
            id: self.id,
            parent_id: self.parent_id,
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

    /// Updates the particle parent ID
    pub fn with_parent_id(&mut self, id: u64) -> Self {
        self.parent_id = id;
        *self
    }

    /// ID of this particle's parent
    pub fn parent_id(&self) -> u64 {
        self.parent_id
    }

    /// Photon, electron or positron
    pub fn species(&self) -> Species {
        self.species
    }

    /// Updates the particle polarization
    pub fn with_polarization(&mut self, pol: StokesVector) -> Self {
        self.pol = pol;
        *self
    }

    /// Returns the particle polarization, if it exists
    #[allow(unused)]
    pub fn polarization(&self) -> StokesVector {
        self.pol
    }

    /// Projects the particle polarization onto the given axis.
    /// `polarization_along_x` and `polarization_along_y` are
    /// provided for convenience
    pub fn polarization_along<T: Into<ThreeVector>>(&self, axis: T) -> f64 {
        let sv = self.polarization();
        let n = ThreeVector::from(self.normalized_momentum()).normalize();
        sv.project_onto(n, axis.into())
    }

    /// Projects the particle polarization onto the x axis.
    pub fn polarization_along_x(&self) -> f64 {
        self.polarization_along([1.0, 0.0, 0.0])
    }

    /// Projects the particle polarization onto the y axis.
    pub fn polarization_along_y(&self) -> f64 {
        self.polarization_along([0.0, 1.0, 0.0])
    }
}

impl Shower {
    pub fn multiplicity(&self) -> usize {
        self.secondaries.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts;
    use super::*;

    #[test]
    fn project_polarization() {
        let mut photon = Particle::create(Species::Photon, [0.0; 4].into());
        let angles = [
            (consts::PI, 0.0_f64),
            (0.75 * consts::PI, 0.0),
            (0.6 * consts::PI, consts::FRAC_PI_2),
            (0.8 * consts::PI, 1.0),
            (0.2 * consts::PI, 4.0),
        ];
        let x_axis = ThreeVector::from([1.0, 0.0, 0.0]);
        let y_axis = ThreeVector::from([0.0, 1.0, 0.0]);

        for (theta, phi) in &angles{
            let k = FourVector::lightlike(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
            photon.with_normalized_momentum(k);
            println!("photon k = {:.3} {:.3} {:.3}", k[1], k[2], k[3]);

            // LP E
            photon.with_polarization([1.0, 1.0, 0.0, 0.0].into());
            //let eps = ThreeVector::from(e1 - (k * e1) * n / (k * n)).normalize();
            let eps: ThreeVector = [k[3] / k[1].hypot(k[3]), 0.0, -k[1] / k[1].hypot(k[3])].into();

            let pol = photon.polarization_along_x();
            let target = (eps * x_axis).powi(2);
            println!("\tLPx: got pol_x = {:.3}, expected = {:.3}", pol, target);
            assert!(pol == target || (pol - target).abs() < 1.0e-6);

            let pol = photon.polarization_along_y();
            let target = (eps * y_axis).powi(2);
            println!("\tLPx: got pol_y = {:.3}, expected = {:.3}", pol, target);
            assert!(pol == target || (pol - target).abs() < 1.0e-6);

            // LP B
            photon.with_polarization([1.0, -1.0, 0.0, 0.0].into());
            let eps: ThreeVector = [k[3] / k[1].hypot(k[3]), 0.0, -k[1] / k[1].hypot(k[3])].into();
            let eps = eps.cross(k.into()).normalize();
            //let eps = ThreeVector::from(e2 - (k * e2) * n / (k * n)).normalize();

            let pol = photon.polarization_along_x();
            let target = (eps * x_axis).powi(2);
            println!("\tLPy: got pol_x = {:.3}, expected = {:.3}", pol, target);
            assert!(pol == target || (pol - target).abs() < 1.0e-6);

            let pol = photon.polarization_along_y();
            let target = (eps * y_axis).powi(2);
            println!("\tLPy: got pol_y = {:.3}, expected = {:.3}", pol, target);
            assert!(pol == target || (pol - target).abs() < 1.0e-6);

            // CP+
            photon.with_polarization([1.0, 0.0, 0.0, 1.0].into());
            let e1: ThreeVector = [k[3] / k[1].hypot(k[3]), 0.0, -k[1] / k[1].hypot(k[3])].into();
            let e1 = e1 / consts::SQRT_2;
            let e2 = e1.cross(k.into()).normalize() / consts::SQRT_2;

            let pol = photon.polarization_along_x();
            //let eps = ThreeVector::from(e1 - (k * e1) * n / (k * n)).normalize() / consts::SQRT_2;
            let target = (e1 * x_axis).powi(2) + (e2 * x_axis).powi(2);
            println!("\tCP+: got pol_x = {:.3}, expected = {:.3}", pol, target);
            assert!(pol == target || (pol - target).abs() < 1.0e-6);

            let pol = photon.polarization_along_y();
            //let eps = ThreeVector::from(e2 - (k * e2) * n / (k * n)).normalize() / consts::SQRT_2;
            let target = (e1 * y_axis).powi(2) + (e2 * y_axis).powi(2);
            println!("\tCP+: got pol_y = {:.3}, expected = {:.3}", pol, target);
            assert!(pol == target || (pol - target).abs() < 1.0e-6);
        }
    }
}