use std::f64::consts;
use rand::prelude::*;

use crate::field::{Field, Polarization};
use crate::constants::*;
use crate::geometry::{ThreeVector, FourVector};
use crate::nonlinear_compton;

/// Represents the envelope of a focusing laser pulse, i.e.
/// the field after cycle averaging
pub struct FocusedLaser {
    a0: f64,
    waist: f64,
    duration: f64,
    wavevector: FourVector,
    pol: Polarization,
}

impl FocusedLaser {
    pub fn new(a0: f64, wavelength: f64, waist: f64, duration: f64, pol: Polarization) -> Self {
        let wavevector = (2.0 * consts::PI / wavelength) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        FocusedLaser {
            a0,
            waist,
            duration,
            wavevector,
            pol
        }
    }

    fn omega(&self) -> f64 {
        SPEED_OF_LIGHT * self.wavevector[0]
    }

    fn rayleigh_range(&self) -> f64 {
        0.5 * self.wavevector[0] * self.waist.powi(2)
    }

    /// Returns the local wavevector scaled by the electron mass,
    /// hbar k / (me c^2)
    fn k_at(&self, r: FourVector) -> FourVector {
        let zeta = r[3] / self.rayleigh_range();
        let rho = r[1].hypot(r[2]) / self.waist;
        let n_perp = if rho > 0.0 {
            let n_perp = 2.0 * zeta * rho / (self.wavevector[0] * self.waist * (1.0 + zeta * zeta));
            n_perp * ThreeVector::new(r[1], r[2], 0.0).normalize()
        } else {
            ThreeVector::new(0.0, 0.0, 0.0)
        };
        let n_long = ThreeVector::new(0.0, 0.0, 1.0);
        let n = (n_perp + n_long).normalize();
        SPEED_OF_LIGHT * COMPTON_TIME * self.wavevector[0] * FourVector::lightlike(n[0], n[1], n[2])
    }

    pub fn a_sqd(&self, r: FourVector) -> f64 {
        // Gaussian beam
        let z_r = self.rayleigh_range();
        let width_sqd = 1.0 + (r[3] / z_r).powi(2);
        let rho_sqd = (r[1].powi(2) + r[2].powi(2)) / self.waist.powi(2);
        let norm = match self.pol {
            Polarization::Linear => 0.5,
            Polarization::Circular => 1.0,
        };
        let beam = norm * self.a0.powi(2) * (-2.0 * rho_sqd / width_sqd).exp() / width_sqd;

        // Pulse envelope
        let phase = self.wavevector * r; // - r[3] * rho_sqd / (z_r * width_sqd);
        let tau = self.omega() * self.duration;
        let envelope = (-4.0 * consts::LN_2 * phase.powi(2) / tau.powi(2)).exp();

        beam * envelope
    }

    pub fn grad_a_sqd(&self, r: FourVector) -> FourVector {
        // Gaussian beam
        let z_r = self.rayleigh_range();
        let width_sqd = 1.0 + (r[3] / z_r).powi(2);
        let rho_sqd = (r[1].powi(2) + r[2].powi(2)) / self.waist.powi(2);
        let norm = match self.pol {
            Polarization::Linear => 0.5,
            Polarization::Circular => 1.0,
        };
        let beam = norm * self.a0.powi(2) * (-2.0 * rho_sqd / width_sqd).exp() / width_sqd;

        let grad_beam = [
            -4.0 * beam * r[1] / (self.waist.powi(2) * width_sqd),
            -4.0 * beam * r[2] / (self.waist.powi(2) * width_sqd),
            (2.0 * beam * r[3] / (z_r.powi(2) * width_sqd)) * (2.0 * rho_sqd / width_sqd - 1.0)
        ];

        // Pulse envelope
        let phase = self.wavevector * r; // - r[3] * rho_sqd / (z_r * width_sqd);
        let tau = self.omega() * self.duration;
        let envelope = (-4.0 * consts::LN_2 * phase.powi(2) / tau.powi(2)).exp();

        let grad_envelope = [
            0.0,
            0.0,
            8.0 * consts::LN_2 * self.wavevector[0] * phase * envelope / tau.powi(2)
        ];

        FourVector::new(
            0.0,
            grad_beam[0] * envelope,
            grad_beam[1] * envelope,
            beam * grad_envelope[2] + grad_beam[2] * envelope
        )
    }
}

impl Field for FocusedLaser {
    fn total_energy(&self) -> f64 {
        0.0
    }

    fn max_timestep(&self) -> Option<f64> {
        Some(1.0 / self.omega())
    }

    fn contains(&self, r: FourVector) -> bool {
        let phase: f64 = self.wavevector * r;
        phase < 3.0 * self.omega() * self.duration
    }

    fn push(&self, r: FourVector, u: FourVector, rqm: f64, dt: f64) -> (FourVector, FourVector) {
        // equations of motion are:
        //   du/dt = -c grad<a^2>(r) / (2 gamma) = f(r, u)
        //   dr/dt = u c / gamma = g(u)
        // where gamma^2 = 1 + <a^2> + |u|^2, i.e. u * u = 1 + <a^2>

        let scale = (rqm / (ELECTRON_CHARGE / ELECTRON_MASS)).powi(2);

        let f = |r: FourVector, u: FourVector| -> FourVector {
            -scale * SPEED_OF_LIGHT * self.grad_a_sqd(r) / (2.0 * u[0])
        };

        let g = |u: FourVector| -> FourVector {
            SPEED_OF_LIGHT * u / u[0]
        };

        // Heun's method, construct intermediate values
        let r_inter: FourVector = r + dt * g(u);
        let u_inter: FourVector = u + dt * f(r, u);
        let u_inter: FourVector = u_inter.with_sqr(1.0 + self.a_sqd(r_inter));

        // And corrected final values
        let r_new = r + 0.5 * dt * (g(u) + g(u_inter));
        let u_new = u + 0.5 * dt * (f(r, u) + f(r_inter, u_inter));
        let u_new = u_new.with_sqr(1.0 + self.a_sqd(r_new));

        (r_new, u_new)
    }

    fn radiate<R: Rng>(&self, r: FourVector, u: FourVector, dt: f64, rng: &mut R) -> Option<FourVector> {
        let k = self.k_at(r);
        let prob = nonlinear_compton::probability(k, u, dt).unwrap_or(0.0);
        if rng.gen::<f64>() < prob {
            let (_n, k) = nonlinear_compton::generate(k, u, rng, None);
            Some(k)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_axis() {
        let t_start = -20.0 * 0.8e-6 / (SPEED_OF_LIGHT);
        let dt = 0.25 * 0.8e-6 / (SPEED_OF_LIGHT);
        let laser = FocusedLaser::new(100.0, 0.8e-6, 4.0e-6, 30.0e-15, Polarization::Circular);

        let mut u = FourVector::new(0.0, 0.0, 0.0, -1000.0).unitize();
        let mut r = FourVector::new(0.0, 0.0, 0.0, 0.0) + u * SPEED_OF_LIGHT * t_start / u[0];

        for _i in 0..(20*2*5) {
            let new = laser.push(r, u, ELECTRON_CHARGE / ELECTRON_MASS, dt);
            r = new.0;
            u = new.1;
        }

        println!("final u_perp = ({:.3e}, {:.3e}), u^2 = {:.3e}", u[1], u[2], u * u);
        assert!(u[1] < 1.0e-3);
        assert!(u[2] < 1.0e-3);
        assert!((u * u - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn wavefront_curvature() {
        let laser = FocusedLaser::new(100.0, 0.8e-6, 2.0e-6, 30.0e-15, Polarization::Circular);
        let r = FourVector::new(0.0, 1.0e-6, 1.0e-6, 10.0e-6);
        let k = laser.k_at(r);
        let omega = 1.55e-6 / 0.511;
        println!("r = ({:.3e} {:.3e} {:.3e} {:.3e}), k^0 = {:.6e} [expected {:.6e}], k/k^0 = ({:.3e} {:.3e} {:.3e} {:.3e})", r[0], r[1], r[2], r[3], k[0], omega, 1.0, k[1]/k[0], k[2]/k[0], k[3]/k[0]);
        assert_eq!(k * k, 0.0);
    }
}