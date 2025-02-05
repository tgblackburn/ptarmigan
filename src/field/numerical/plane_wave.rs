//! A numerically defined plane-wave laser pulse, for use with the LMA
use rand::prelude::*;

use crate::constants::*;
use crate::field::{Field, Polarization, RadiationMode, EquationOfMotion, RadiationEvent, PairCreationEvent};
use crate::geometry::{FourVector, StokesVector};
use crate::nonlinear_compton;
use crate::pair_creation;

use super::FieldData;

/// Represents a plane-wave laser pulse that is defined by a numerical
/// vector potential
pub struct NumericalPW {
    inner: FieldData,
}

impl From<FieldData> for NumericalPW {
    fn from(laser: FieldData) -> Self {
        Self {inner: laser}
    }
}

impl From<&FieldData> for NumericalPW {
    fn from(laser: &FieldData) -> Self {
        Self {inner: laser.clone()}
    }
}

impl NumericalPW {
    /// Returns the dominant frequency component of the pulse.
    fn omega(&self) -> f64 {
        self.inner.omega
    }

    /// Returns the instantaneous wavevector of the pulse at the
    /// given position, taking into account any nonlinear chirp.
    fn wavevector(&self, r: FourVector) -> FourVector {
        let k = self.inner.omega / SPEED_OF_LIGHT;

        let phase = self.omega() * (r[0] - r[3]) / SPEED_OF_LIGHT;

        let scale = self.get_index(phase)
            .map(|(i, di)| {
                (1.0 - di) * self.inner.dpsi_dphi[i] + di * self.inner.dpsi_dphi[i+1]
            })
            .unwrap_or(1.0);

        [scale * k, 0.0, 0.0, scale * k].into()
    }

    fn get_index(&self, phase: f64) -> Option<(usize, f64)> {
        let index = (phase - self.inner.start) / self.inner.step;

        // Particle hasn't reached the laser pulse yet
        if index <= 0.0 {
            return None;
        }

        let i = index.trunc() as usize;
        let di = index.fract();

        if i >= self.inner.field.len() - 1 {
            return None;
        }

        Some((i, di))
    }

    pub fn a_sqd(&self, r: FourVector) -> f64 {
        let phase = self.omega() * (r[0] - r[3]) / SPEED_OF_LIGHT;

        self.get_index(phase)
            .map(|(i, di)| {
                0.5 * ((1.0 - di) * self.inner.a_sqd[i] + di * self.inner.a_sqd[i+1])
            })
            .unwrap_or(0.0)
    }

    /// Returns the four-gradient (index raised) of the cycle-averaged
    /// potential, i.e. ∇^μ <a^2> = (∂/∂t, -∂/∂x, -∂/∂y, -∂/∂z) <a^2>,
    /// as a function of four-position
    pub fn grad_a_sqd(&self, r: FourVector) -> FourVector {
        let phase = self.omega() * (r[0] - r[3]) / SPEED_OF_LIGHT;

        let grad = self.get_index(phase)
            .map(|(i, _)| {
                // ∂/∂z ⟨a^2⟩ = -∂/∂t ⟨a^2⟩ = -ω0/c ∂/∂ϕ ⟨a^2⟩
                let k = self.omega() / SPEED_OF_LIGHT;
                -0.5 * k * (self.inner.a_sqd[i+1] - self.inner.a_sqd[i]) / self.inner.step
            })
            .unwrap_or(0.0);

        FourVector::new(
            -grad,
            0.0,
            0.0,
            -grad,
        )
    }

    /// Returns the cycle-averaged radiation reaction force, du/dτ
    fn landau_lifshitz_force(&self, r: FourVector, u: FourVector) -> FourVector {
        // du/tau = -(2 ɑ / 3 tau_C) a_rms^2 (k.u/m)^2 u
        let eta = SPEED_OF_LIGHT * COMPTON_TIME * (self.wavevector(r) * u);
        -2.0 * ALPHA_FINE * self.a_sqd(r) * eta.powi(2) * u / (3.0 * COMPTON_TIME)
    }

    /// Returns the leading order contribution to the work done by the
    /// external field in association with radiation losses, per unit proper time.
    /// The work is cycle-averaged and normalized to the electron mass.
    fn landau_lifshitz_work(&self, r: FourVector, u: FourVector) -> f64 {
        let eta = SPEED_OF_LIGHT * COMPTON_TIME * (self.wavevector(r) * u);
        let omega = self.omega();
        let delta = 0.75; // assumed to be LP
        2.0 * ALPHA_FINE * omega * eta * delta * self.a_sqd(r).powi(2) / 3.0
    }
}

impl Field for NumericalPW {
    fn max_timestep(&self) -> Option<f64> {
        let dt = 0.5 * self.inner.step / self.omega();
        Some(dt)
    }

    fn contains(&self, r: FourVector) -> bool {
        let phase = self.omega() * (r[0] - r[3]) / SPEED_OF_LIGHT;
        phase < self.inner.end
    }

    fn ideal_initial_z(&self) -> f64 {
        -SPEED_OF_LIGHT * self.inner.start / (2.0 * self.omega())
    }

    fn energy(&self) -> (f64, &'static str) {
        (self.inner.energy_flux, "J/m^2")
    }

    /// Advances particle position and momentum using a leapfrog method
    /// in proper time. As a consequence, the change in the time may not
    /// be identical to the requested `dt`.
    fn push(&self, r: FourVector, u: FourVector, rqm: f64, dt: f64, eqn: EquationOfMotion) -> (FourVector, FourVector, f64, f64) {
        // equations of motion are:
        //   du/dtau = c grad<a^2>(r) / 2 = f(r)
        //   dr/dtau = c u
        //
        // proper time interval approx equivalent to dt
        let dtau = dt / u[0];
        let scale = (rqm / (ELECTRON_CHARGE / ELECTRON_MASS)).powi(2);

        // estimate u at midpoint, only needed for CRR
        let u_mid = match eqn {
            EquationOfMotion::Lorentz => [0.0; 4].into(),
            EquationOfMotion::LandauLifshitz => {
                let f = 0.5 * SPEED_OF_LIGHT * scale * self.grad_a_sqd(r);
                let g = scale.powi(2) * self.landau_lifshitz_force(r, u);
                u + 0.5 * (f + g) * dtau
            },
            EquationOfMotion::ModifiedLandauLifshitz => panic!("Gaunt factor correction is unavailable in LMA mode!"),
        };

        // r_{n+1/2} = r_n + c u_n * dtau / 2
        let r = r + 0.5 * SPEED_OF_LIGHT * u * dtau;
        let dt_actual = 0.5 * u[0] * dtau;

        // u_{n+1} = u_n + f(r_{n+1/2}) * dtau
        let f = 0.5 * SPEED_OF_LIGHT * scale * self.grad_a_sqd(r);

        // RR contribution
        let g: FourVector = match eqn {
            EquationOfMotion::Lorentz => [0.0; 4].into(),
            EquationOfMotion::LandauLifshitz =>  scale.powi(2) * self.landau_lifshitz_force(r, u_mid),
            EquationOfMotion::ModifiedLandauLifshitz => panic!("Gaunt factor correction is unavailable in LMA mode!"),
        };

        let u = u + (f + g) * dtau;

        let dwork = match eqn {
            EquationOfMotion::Lorentz => f[0] * dtau,
            EquationOfMotion::LandauLifshitz => (f[0] + self.landau_lifshitz_work(r, u_mid)) * dtau,
            EquationOfMotion::ModifiedLandauLifshitz => panic!("Gaunt factor correction is unavailable in LMA mode!"),
        };

        // r_{n+1} = r_{n+1/2} + c u_{n+1} * dtau / 2
        let r = r + 0.5 * SPEED_OF_LIGHT * u * dtau;
        let dt_actual = dt_actual + 0.5 * u[0] * dtau;

        // enforce correct mass
        let u = u.with_sqr(1.0 + self.a_sqd(r));

        (r, u, dt_actual, dwork)
    }

    fn radiate<R: Rng>(&self, r: FourVector, u: FourVector, dt: f64, rng: &mut R, mode: RadiationMode) -> Option<RadiationEvent> {
        let a = self.a_sqd(r).sqrt();
        let kappa = SPEED_OF_LIGHT * COMPTON_TIME * self.wavevector(r);

        let prob = nonlinear_compton::probability(kappa, u, dt, Polarization::Linear, mode).unwrap_or(0.0);

        if rng.gen::<f64>() < prob {
            let (n, k, pol) = nonlinear_compton::generate(kappa, u, Polarization::Linear, 0.0, mode, rng);
            let event = RadiationEvent {
                k,
                u_prime: u + (n as f64) * kappa - k,
                pol,
                a_eff: a,
                chi: a * (u * kappa),
                absorption: (n as f64) * kappa[0],
            };
            Some(event)
        } else {
            None
        }
    }

    fn pair_create<R: Rng>(&self, r: FourVector, ell: FourVector, pol: StokesVector, dt: f64, rng: &mut R, rate_increase: f64) -> (f64, StokesVector, Option<PairCreationEvent>) {
        let a = self.a_sqd(r).sqrt();
        let kappa = SPEED_OF_LIGHT * COMPTON_TIME * self.wavevector(r);

        let (prob, pol_new) = pair_creation::probability(ell, pol, kappa, a, dt, Polarization::Linear, 0.0);

        let rate_increase = if prob * rate_increase > 0.1 {
            0.1 / prob // limit the rate increase
        } else {
            rate_increase
        };

        if rng.gen::<f64>() < prob * rate_increase {
            let (n, q_p) = pair_creation::generate(ell, pol, kappa, a, Polarization::Linear, 0.0, rng);
            let event = PairCreationEvent {
                u_e: ell + (n as f64) * kappa - q_p,
                u_p: q_p,
                frac: 1.0 / rate_increase,
                a_eff: a,
                chi: a * (ell * kappa),
                absorption: (n as f64) * kappa[0],
            };
            (prob, pol_new, Some(event))
        } else {
            (prob, pol_new, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chirped_pulse() {
        use std::f64::consts;
        use crate::field::Coordinate;

        let lambda = 0.8e-6;
        let e0 = 2.0 * consts::PI * ELECTRON_MASS * SPEED_OF_LIGHT_SQD / (ELEMENTARY_CHARGE * lambda);
        let dz = lambda / 100.0;
        let n_cycles = 16.0;
        // chirp parameter
        let c = 1.0 / (2.0 * consts::PI);

        let field: Vec<f64> = (0..2000)
            .map(|i| {
                let z = -dz * ((i as f64) - 1000.0);
                let phi = 2.0 * consts::PI * z / lambda;
                // total phase
                let psi = phi * (1.0 + 0.5 * c * phi / n_cycles);
                // derivative of cos(phi/2n)^2 cos(psi)
                let fx = 0.5 * psi.cos() * (phi / n_cycles).sin() / n_cycles + (1.0 + c * phi / n_cycles) * (0.5 * phi / n_cycles).cos().powi(2) * psi.sin();
                if phi.abs() < consts::PI * n_cycles { e0 * fx } else { 0.0 }
            })
            .collect();

        let laser = FieldData::preprocess(Coordinate::Space,dz, &field).unwrap();
        let laser: NumericalPW = laser.into();

        for phi in [0.0, 2.0 * consts::PI, 8.0 * consts::PI, 14.0 * consts::PI].iter() {
            let z = -lambda * phi / (2.0 * consts::PI);
            let r: FourVector = [0.0, 0.0, 0.0, z].into();

            let a_rms = laser.a_sqd(r).sqrt();
            let target_a_rms = (0.5 * phi / n_cycles).cos().powi(2) / consts::SQRT_2;
            let error = (target_a_rms - a_rms).abs() / target_a_rms;

            println!(
                "phi = {:.1} pi, got a_rms of {:.3}, expected {:.3} => error = {:.3}%",
                phi / consts::PI, a_rms, target_a_rms, 100.0 * error
            );

            assert!(error < 0.01);

            let wavelength = 2.0 * consts::PI / laser.wavevector(r)[0];
            let target_wavelength = lambda / (1.0 + c * phi / n_cycles);
            let error = (target_wavelength - wavelength).abs() / wavelength;

            println!(
                "phi = {:.1} pi, got wavelength of {:.3} um, expected {:.3} um => error = {:.3}%",
                phi / consts::PI, 1.0e6 * wavelength, 1.0e6 * target_wavelength, 100.0 * error
            );

            assert!(error < 0.01);
        }
    }
}