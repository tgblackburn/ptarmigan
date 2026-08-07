//! Representation of the electromagnetic field in the simulation domain

use rand::prelude::*;
use enum_dispatch::enum_dispatch;
use crate::geometry::{FourVector, StokesVector, ThreeVector};

mod properties;
pub use properties::*;

mod focused_laser;
mod fast_focused_laser;
mod plane_wave;
mod fast_plane_wave;
mod lcf;

pub use self::focused_laser::*;
pub use self::fast_focused_laser::*;
pub use self::plane_wave::*;
pub use self::fast_plane_wave::*;

mod numerical;
pub use self::numerical::*;

/// Specific field structures, i.e. types that implement `trait Field`.
#[enum_dispatch]
pub enum Laser {
    PlaneWave,
    FastPlaneWave,
    FocusedLaser,
    FastFocusedLaser,
    NumericalPW,
    NumericalFastPW,
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

    /// Returns `z0` such that an ultrarelatistic particle, initialized with `z = z0` at time `-z0/c`, is
    /// sufficiently distant from the laser so as not to be affected by it.
    fn ideal_initial_z(&self) -> f64;

    /// Advances the position `r` and normalized momentum `u`
    /// of a particle with charge to mass ratio `rqm`
    /// by a timestep `dt`, returning a tuple of the new
    /// position and momentum, as well as the change in
    /// lab time (which may differ from `dt`)
    /// and the energy absorbed from the background field.
    #[allow(non_snake_case)]
    fn push(&self, r: FourVector, u: FourVector, rqm: f64, dt: f64, eqn: EquationOfMotion) -> (FourVector, FourVector, f64, f64) {
        use crate::constants::SPEED_OF_LIGHT;
        let r = r + 0.5 * SPEED_OF_LIGHT * u * dt / u[0];
        let (E, B, _) = self.fields(r);
        lcf::vay_push(r, u, E, B, rqm, dt, eqn)
    }

    /// Checks to see whether an electron or positron in the field, located at
    /// position `r` with momentum `u` emits a photon, and if so,
    /// returns information about the event (see [RadiationEvent]).
    #[allow(non_snake_case)]
    fn radiate<R: Rng>(&self, r: FourVector, u: FourVector, rqm: f64, dt: f64, rng: &mut R, mode: RadiationMode, rate_corr: RateCorrection) -> Option<RadiationEvent> {
        let (E, B, a) = self.fields(r);
        lcf::radiate(u, E, B, rate_corr.dv1, rate_corr.dv2, a, rqm, dt, rng, mode, rate_corr.modifier)
    }

    /// Checks to see if an electron-positron pair is produced by
    /// a photon (position `r`, normalized momentum `ell`, polarization `pol`),
    /// returning the probability that it occurs in the specified interval `dt`,
    /// new Stokes parameters of the photon, and information about
    /// the event (see [PairCreationEvent]).
    ///
    /// A non-unity `rate_increase` makes pair creation more probable
    /// by the given factor, increasing the statistics for what would
    /// otherwise be a rare event. The probability returned is *not*
    /// affected by this increase.
    #[allow(non_snake_case)]
    fn pair_create<R: Rng>(&self, r: FourVector, ell: FourVector, pol: StokesVector, dt: f64, rng: &mut R, rate_increase: f64, uncertainty: f64) -> (f64, StokesVector, Option<PairCreationEvent>) {
        let (E, B, a) = self.fields(r);
        let dv1 = 0.0;
        lcf::pair_create(ell, pol, E, B, dv1, a, dt, rng, rate_increase, uncertainty)
    }

    /// Returns a tuple of the electric and magnetic fields, as well
    /// as the normalized amplitude (if applicable), at the
    /// specified four-position `r`.
    #[allow(unused_variables)]
    fn fields(&self, r: FourVector) -> (ThreeVector, ThreeVector, f64) {
        ([0.0; 3].into(), [0.0; 3].into(), 0.0)
    }

    /// Returns the total energy of the electromagnetic field and the
    /// units of that energy (`"J"`, `"J/m"`, `"J/m^2"` , `"J/m^3"`, as appropriate).
    /// If the field is infinitely extended in one or more dimensions,
    /// the energy is calculated per unit length in those dimensions.
    fn energy(&self) -> (f64, &'static str);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    #[test]
    fn cp_deflection() {
        let n_cycles = 10.0;
        let envelope = Envelope::Flattop;

        let fast_laser = FastFocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Circular, 0.0)
            .with_envelope(envelope);
        let laser = FocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Circular, 0.0)
            .with_envelope(envelope);

        let t_start = -20.0 * 0.8e-6 / (SPEED_OF_LIGHT);
        let x0 = 2.0e-6;

        let u = FourVector::new(0.0, 0.0, 0.0, -1000.0).unitize();
        let r = FourVector::new(0.0, x0, 0.0, 0.0) + u * SPEED_OF_LIGHT * t_start / u[0];

        // ponderomotive solver
        let dt = laser.max_timestep().unwrap();
        let mut pond = (r, u, dt, 0.0);
        while laser.contains(pond.0) {
            pond = laser.push(pond.0, pond.1, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
        }
        let pond = pond.1;

        // Lorentz force solve
        // ponderomotive solver
        let dt = fast_laser.max_timestep().unwrap();
        let mut lorentz = (r, u, dt, 0.0);
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

        let fast_laser = FastFocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Linear, 0.0)
            .with_envelope(envelope);
        let laser = FocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Linear, 0.0)
            .with_envelope(envelope);

        let t_start = -20.0 * 0.8e-6 / (SPEED_OF_LIGHT);
        let y0 = 2.0e-6;

        let u = FourVector::new(0.0, 0.0, 0.0, -1000.0).unitize();
        let r = FourVector::new(0.0, 0.0, y0, 0.0) + u * SPEED_OF_LIGHT * t_start / u[0];

        // ponderomotive solver
        let dt = laser.max_timestep().unwrap();
        let mut pond = (r, u, dt, 0.0);
        while laser.contains(pond.0) {
            pond = laser.push(pond.0, pond.1, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
        }
        let pond = pond.1;

        // Lorentz force solve
        // ponderomotive solver
        let dt = fast_laser.max_timestep().unwrap();
        let mut lorentz = (r, u, dt, 0.0);
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