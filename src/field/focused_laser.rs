use std::f64::consts;
use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::field::{Field, Polarization};
use crate::constants::*;
use crate::geometry::FourVector;
use crate::nonlinear_compton;
use crate::pair_creation;

/// Represents the envelope of a focusing laser pulse, i.e.
/// the field after cycle averaging
pub struct FocusedLaser {
    a0: f64,
    waist: f64,
    duration: f64,
    wavevector: FourVector,
    pol: Polarization,
    bandwidth: f64,
}

impl FocusedLaser {
    pub fn new(a0: f64, wavelength: f64, waist: f64, duration: f64, pol: Polarization) -> Self {
        let wavevector = (2.0 * consts::PI / wavelength) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        FocusedLaser {
            a0,
            waist,
            duration,
            wavevector,
            pol,
            bandwidth: 0.0,
        }
    }

    pub fn with_finite_bandwidth(self) -> Self {
        let mut cpy = self;
        let n_fwhm = SPEED_OF_LIGHT * cpy.duration * cpy.wavevector[0] / (2.0 * consts::PI);
        cpy.bandwidth = (0.5 * consts::LN_2).sqrt() / (consts::PI * n_fwhm);
        cpy
    }

    fn omega(&self) -> f64 {
        SPEED_OF_LIGHT * self.wavevector[0]
    }

    fn rayleigh_range(&self) -> f64 {
        0.5 * self.wavevector[0] * self.waist.powi(2)
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

        #[cfg(feature = "cos2-envelope-in-3d")]
        let envelope = if phase.abs() < consts::PI * self.duration {
            (phase / (2.0 * self.duration)).cos().powi(4)
        } else {
            0.0
        };

        #[cfg(not(feature = "cos2-envelope-in-3d"))]
        let envelope = {
            let tau = self.omega() * self.duration;
            (-4.0 * consts::LN_2 * phase.powi(2) / tau.powi(2)).exp()
        };

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

        #[cfg(feature = "cos2-envelope-in-3d")]
        let (envelope, grad_envelope) = if phase.abs() < consts::PI * self.duration {
            let envelope = (phase / (2.0 * self.duration)).cos().powi(4);
            (envelope, 2.0 * self.wavevector[0] * (phase / (2.0 * self.duration)).tan() * envelope / self.duration)
        } else {
            (0.0, 0.0)
        };

        #[cfg(not(feature = "cos2-envelope-in-3d"))]
        let (envelope, grad_envelope) = {
            let tau = self.omega() * self.duration;
            let envelope = (-4.0 * consts::LN_2 * phase.powi(2) / tau.powi(2)).exp();
            (envelope, 8.0 * consts::LN_2 * self.wavevector[0] * phase * envelope / tau.powi(2))
        };

        let grad_envelope = [0.0, 0.0, grad_envelope];

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
        // peak power of an LP Gaussian beam
        let peak_field = ELECTRON_MASS * SPEED_OF_LIGHT * self.omega() * self.a0 / ELEMENTARY_CHARGE;
        let peak_power = 0.25 * consts::PI * self.waist.powi(2) * peak_field.powi(2) / (SPEED_OF_LIGHT * VACUUM_PERMEABILITY);
        // time = int f(t)^2 dt where f is the electric-field envelope
        let time = self.duration * (consts::PI / 16.0f64.ln()).sqrt();
        let norm = match self.pol {
            Polarization::Linear => 1.0,
            Polarization::Circular => 2.0,
        };
        norm * peak_power * time
    }

    fn max_timestep(&self) -> Option<f64> {
        Some(1.0 / self.omega())
    }

    #[cfg(feature = "cos2-envelope-in-3d")]
    fn contains(&self, r: FourVector) -> bool {
        let phase: f64 = self.wavevector * r;
        phase < consts::PI * self.duration
    }

    #[cfg(not(feature = "cos2-envelope-in-3d"))]
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

    fn radiate<R: Rng>(&self, _r: FourVector, u: FourVector, dt: f64, rng: &mut R) -> Option<FourVector> {
        let width = 1.0 + self.bandwidth * rng.sample::<f64,_>(StandardNormal);
        assert!(width > 0.0, "The fractional bandwidth of the pulse, {:.3e}, is large enough that the sampled frequency has fallen below zero!", self.bandwidth);
        let kappa = SPEED_OF_LIGHT * COMPTON_TIME * self.wavevector * width;
        let prob = nonlinear_compton::probability(kappa, u, dt).unwrap_or(0.0);
        if rng.gen::<f64>() < prob {
            let (_n, k) = nonlinear_compton::generate(kappa, u, rng, None);
            Some(k)
        } else {
            None
        }
    }

    fn pair_create<R: Rng>(&self, r: FourVector, ell: FourVector, dt: f64, rng: &mut R, rate_increase: f64) -> (f64, Option<(FourVector, FourVector)>) {
        let a = self.a_sqd(r).sqrt();
        let kappa = SPEED_OF_LIGHT * COMPTON_TIME * self.wavevector;
        let prob = pair_creation::probability(ell, kappa, a, dt).unwrap_or(0.0);
        if rng.gen::<f64>() < prob * rate_increase {
            let (n, q_p) = pair_creation::generate(ell, kappa, a, rng);
            (prob, Some((ell + (n as f64) * kappa - q_p, q_p)))
        } else {
            (prob, None)
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
}