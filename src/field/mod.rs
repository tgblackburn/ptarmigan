//! Representation of the electromagnetic field in the simulation domain

use rand::prelude::*;
use enum_dispatch::enum_dispatch;
use crate::geometry::{FourVector, StokesVector};

#[cfg(feature = "hdf5-output")]
use hdf5_writer::{Hdf5Type, Datatype};

mod focused_laser;
mod fast_focused_laser;
mod plane_wave;
mod fast_plane_wave;

pub use self::focused_laser::*;
pub use self::fast_focused_laser::*;
pub use self::plane_wave::*;
pub use self::fast_plane_wave::*;

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
#[derive(Copy, Clone, Eq, PartialEq)]
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

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum EquationOfMotion {
    Lorentz,
    LandauLifshitz,
    ModifiedLandauLifshitz,
}

impl EquationOfMotion {
    fn includes_rr(&self) -> bool {
        match self {
            EquationOfMotion::LandauLifshitz | EquationOfMotion::ModifiedLandauLifshitz => true,
            EquationOfMotion::Lorentz => false,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum RadiationMode {
    Quantum,
    Classical,
}

/// Specific field structures, i.e. types that implement `trait Field`.
#[enum_dispatch]
pub enum Laser {
    PlaneWave,
    FastPlaneWave,
    FocusedLaser,
    FastFocusedLaser,
}

/// Represents the electromagnetic field in a spatiotemporal domain.
#[enum_dispatch(Laser)]
pub trait Field {
    /// Returns the largest usuable value of the timestep `dt`
    /// used in the particle push, or `None` if there is no
    /// particular restriction
    fn max_timestep(&self) -> Option<f64>;

    /// Is the specified four-position within the field?
    fn contains(&self, r: FourVector) -> bool;

    /// Advances the position `r` and normalized momentum `u`
    /// of a particle with charge to mass ratio `rqm`
    /// by a timestep `dt`, returning a tuple of the new
    /// position and momentum, as well as the change in
    /// lab time (which may differ from `dt`)
    fn push(&self, r: FourVector, u: FourVector, rqm: f64, dt: f64, eqn: EquationOfMotion) -> (FourVector, FourVector, f64);

    /// Checks to see whether an electron in the field, located at
    /// position `r` with momentum `u` emits a photon, and if so,
    /// returns the momentum of that photon,
    /// its polarization,
    /// the new momentum of the electron,
    /// and the effective a0 of the interaction.
    fn radiate<R: Rng>(&self, r: FourVector, u: FourVector, dt: f64, rng: &mut R, mode: RadiationMode) -> Option<(FourVector, StokesVector, FourVector, f64)>;

    /// Checks to see if an electron-positron pair is produced by
    /// a photon (position `r`, normalized momentum `ell`, polarization `pol`),
    /// returning the probability that it occurs in the specified interval `dt` and,
    /// if so, the fraction of the photon that decays, the
    /// new Stokes parameters of the photon, the
    /// the momentum of the electron and positron, and the effective
    /// a0 of the interaction.
    ///
    /// A non-unity `rate_increase` makes pair creation more probable
    /// by the given factor, increasing the statistics for what would
    /// otherwise be a rare event. The probability returned is *not*
    /// affected by this increase.
    fn pair_create<R: Rng>(&self, r: FourVector, ell: FourVector, pol: StokesVector, dt: f64, rng: &mut R, rate_increase: f64) -> (f64, f64, StokesVector, Option<(FourVector, FourVector, f64)>);

    /// Returns `z0` such that an ultrarelatistic particle, initialized with `z = z0` at time `-z0/c`, is
    /// sufficiently distant from the laser so as not to be affected by it.
    fn ideal_initial_z(&self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    #[test]
    fn cp_deflection() {
        let n_cycles = 10.0;
        let envelope = Envelope::Flattop;

        let fast_laser = FastFocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Circular)
            .with_envelope(envelope);
        let laser = FocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Circular)
            .with_envelope(envelope);

        let t_start = -20.0 * 0.8e-6 / (SPEED_OF_LIGHT);
        let x0 = 2.0e-6;

        let u = FourVector::new(0.0, 0.0, 0.0, -1000.0).unitize();
        let r = FourVector::new(0.0, x0, 0.0, 0.0) + u * SPEED_OF_LIGHT * t_start / u[0];

        // ponderomotive solver
        let dt = laser.max_timestep().unwrap();
        let mut pond = (r, u, dt);
        while laser.contains(pond.0) {
            pond = laser.push(pond.0, pond.1, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
        }
        let pond = pond.1;

        // Lorentz force solve
        // ponderomotive solver
        let dt = fast_laser.max_timestep().unwrap();
        let mut lorentz = (r, u, dt);
        while fast_laser.contains(lorentz.0) {
            lorentz = fast_laser.push(lorentz.0, lorentz.1, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
        }
        let lorentz = lorentz.1;

        let theory = 2.0 * match envelope {
            Envelope::CosSquared => 1.13724,
            Envelope::Flattop => 2.95684,
            Envelope::Gaussian => 3.22816,
        };

        let pond_angle = 1.0e3 * pond[1].atan2(-pond[3]);
        let lorentz_angle = 1.0e3 * lorentz[1].atan2(-lorentz[3]);
        let error = ((pond_angle - lorentz_angle) / lorentz_angle).abs();

        println!("angle [PF] = {:.3e} [{:.3e}], angle [LF] = {:.3e} [{:.3e}], error = {:.3}%, predicted = {:.3e}", pond_angle, 1.0e3 * pond[2].atan2(-pond[3]), lorentz_angle, 1.0e3 * lorentz[2].atan2(-lorentz[3]), 100.0 * error, theory);
        assert!(error < 1.0e-2);
    }

    #[test]
    fn lp_deflection() {
        let n_cycles = 10.0;
        let envelope = Envelope::Gaussian;

        let fast_laser = FastFocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Linear)
            .with_envelope(envelope);
        let laser = FocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Linear)
            .with_envelope(envelope);

        let t_start = -20.0 * 0.8e-6 / (SPEED_OF_LIGHT);
        let y0 = 2.0e-6;

        let u = FourVector::new(0.0, 0.0, 0.0, -1000.0).unitize();
        let r = FourVector::new(0.0, 0.0, y0, 0.0) + u * SPEED_OF_LIGHT * t_start / u[0];

        // ponderomotive solver
        let dt = laser.max_timestep().unwrap();
        let mut pond = (r, u, dt);
        while laser.contains(pond.0) {
            pond = laser.push(pond.0, pond.1, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
        }
        let pond = pond.1;

        // Lorentz force solve
        // ponderomotive solver
        let dt = fast_laser.max_timestep().unwrap();
        let mut lorentz = (r, u, dt);
        while fast_laser.contains(lorentz.0) {
            lorentz = fast_laser.push(lorentz.0, lorentz.1, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
        }
        let lorentz = lorentz.1;

        let theory = match envelope {
            Envelope::CosSquared => 1.13724,
            Envelope::Flattop => 2.95684,
            Envelope::Gaussian => 3.22816,
        };

        let pond_angle = 1.0e3 * pond[2].atan2(-pond[3]);
        let lorentz_angle = 1.0e3 * lorentz[2].atan2(-lorentz[3]);
        let error = ((pond_angle - lorentz_angle) / lorentz_angle).abs();

        println!("angle [PF] = {:.3e}, angle [LF] = {:.3e}, error = {:.3}%, predicted = {:.3e}", pond_angle, lorentz_angle, 100.0 * error, theory);
        assert!(error < 1.0e-2);
    }

}