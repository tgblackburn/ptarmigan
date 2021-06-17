use std::f64::consts;
use rand::prelude::*;
use num::complex::Complex;

use crate::field::{Field, Polarization};
use crate::constants::*;
use crate::geometry::{FourVector, ThreeVector};
use crate::lcfa;

/// Represents a focusing laser pulse, including
/// the fast oscillating carrier wave
pub struct FastFocusedLaser {
    a0: f64,
    waist: f64,
    duration: f64,
    wavevector: FourVector,
    pol: Polarization,
}

impl FastFocusedLaser {
    #[allow(unused)]
    pub fn new(a0: f64, wavelength: f64, waist: f64, duration: f64, pol: Polarization) -> Self {
        let wavevector = (2.0 * consts::PI / wavelength) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        FastFocusedLaser {
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

    /// The electric and magnetic fields of a Gaussian beam
    /// (including terms up to fourth order in the diffraction angle)
    /// at four position `r`, assuming a given carrier envelope `phase`.
    /// The beam is linearly polarized along the x axis.
    #[allow(non_snake_case)]
    fn beam(&self, r: FourVector, phase: f64) -> (ThreeVector, ThreeVector) {
        let x = r[1] / self.waist;
        let y = r[2] / self.waist;
        let z = r[3] / self.rayleigh_range();
        let rho = x.hypot(y);
        let e = self.waist / self.rayleigh_range();

        let i: Complex<f64> = Complex::new(0.0, 1.0);
        let f: Complex<f64> = i / (z + i);
        let prefactor = f * (-f * rho.powi(2) + i * (self.wavevector * r + phase)).exp();

        let Ex = -i * prefactor * (
            1.0
            + e.powi(2) * f * f * (x.powi(2) - f * rho.powi(4) / 4.0)
            + e.powi(4) * f * f * (1.0/8.0 - f * rho.powi(2) / 4.0 + f * f * ((x * rho).powi(2) - rho.powi(4) / 16.0) + f * f * f * rho.powi(4) * (-x * x - 0.5 * rho.powi(2)) / 4.0 + f * f * f * f * rho.powi(8) / 32.0)
        );

        let Ey = -i * prefactor * (
            e.powi(2) * f * f * x * y
            + e.powi(4) * x * y * f.powf(4.0) * rho.powi(2) * (1.0 - f * rho.powi(2) / 4.0)
        );

        let Ez = prefactor * (
            e * f * x
            + e.powi(3) * f * f * x * (-0.5 + f * rho.powi(2) - f * f * rho.powi(4) / 4.0)
        );

        let Bx = Complex::new(0.0, 0.0);

        let By = -i * prefactor * (
            1.0
            + e.powi(2) * f * f * rho.powi(2) * (1.0 - f * rho.powi(2) / 2.0) / 2.0
            + e.powi(4) * f * f * (-1.0 + 2.0 * f * rho.powi(2) + 2.5 * f * f * rho.powi(4) - 2.0 * f * f * f * rho.powi(6) + 0.25 * f * f * f * f * rho.powi(8)) / 8.0
        );

        let Bz = prefactor * (
            e * f * y
            + e.powi(3) * f * f * y * (1.0 + f * rho.powi(2) - f * f * rho.powi(4) / 2.0) / 2.0
        );
        
        let E0 = ELECTRON_MASS * SPEED_OF_LIGHT * self.omega() * self.a0 / ELEMENTARY_CHARGE;
        let E = E0 * ThreeVector::new(Ex.re, Ey.re, Ez.re);
        let B = E0 * ThreeVector::new(Bx.re, By.re, Bz.re) / SPEED_OF_LIGHT;

        (E, B)
    }

    /// Returns a tuple of the electric and magnetic fields E and B
    /// at the specified four position.
    /// 
    /// The result is accurate to fourth order in the diffraction angle
    /// ϵ = w0 / zR, where w0 is the waist and zR the Rayleigh range,
    /// using the analytical results given in Salamin,
    /// "Fields of a Gaussian beam beyond the paraxial approximation",
    /// Appl. Phys. B 86, 319 (2007).
    #[allow(non_snake_case)]
    fn fields(&self, r: FourVector) -> (ThreeVector, ThreeVector) {
        let (E, B) = match self.pol {
            Polarization::Linear => self.beam(r, 0.0),
            Polarization::Circular => {
                let (Ex, Bx) = self.beam(r, 0.0);
                let axis = ThreeVector::from(self.wavevector).normalize();
                // need to swap definitions of x and y, as well as rotating the E, B vectors
                let r_prime = ThreeVector::from(r).rotate_around(axis, -consts::FRAC_PI_2);
                let r_prime = FourVector::new(r[0], r_prime[0], r_prime[1], r_prime[2]);
                let (Ey, By) = self.beam(r_prime, consts::FRAC_PI_2);
                let Ey = Ey.rotate_around(axis, consts::FRAC_PI_2);
                let By = By.rotate_around(axis, consts::FRAC_PI_2);
                (Ex + Ey, Bx + By)
            }
        };

        // Field profile - compare to FocusedLaser, which is for the intensity profile
        let phase = self.wavevector * r;

        #[cfg(feature = "cos2-envelope-in-3d")]
        let envelope = if phase.abs() < consts::PI * self.duration {
            (phase / (2.0 * self.duration)).cos().powi(2)
        } else {
            0.0
        };

        #[cfg(not(feature = "cos2-envelope-in-3d"))]
        let envelope = {
            let tau = self.omega() * self.duration;
            (-2.0 * consts::LN_2 * phase.powi(2) / tau.powi(2)).exp()
        };

        (envelope * E, envelope * B)
    }

    /// Returns the position and momentum of a particle with charge-to-mass ratio `rqm`,
    /// which has been accelerated in an electric field `E` and magnetic field `B`
    /// over a time interval `dt`.
    #[allow(non_snake_case)]
    #[inline]
    pub fn vay_push(r: FourVector, ui: FourVector, E: ThreeVector, B: ThreeVector, rqm: f64, dt: f64) -> (FourVector, FourVector) {
        // velocity in SI units
        let u = ThreeVector::from(ui);
        let gamma = (1.0 + u * u).sqrt(); // enforce mass-shell condition
        let v = SPEED_OF_LIGHT * u / gamma;

        // u_i = u_{i-1/2} + (q dt/2 m c) (E + v_{i-1/2} x B)
        let alpha = rqm * dt / (2.0 * SPEED_OF_LIGHT);
        let u_half = u + alpha * (E + v.cross(B));

        // u' =  u_{i-1/2} + (q dt/2 m c) (2 E + v_{i-1/2} x B)
        let u_prime = u_half + alpha * E;
        let gamma_prime_sqd = 1.0 + u_prime * u_prime;

        // update Lorentz factor
        let tau = alpha * SPEED_OF_LIGHT * B;
        let u_star = u_prime * tau;
        let sigma = gamma_prime_sqd - tau * tau;

        let gamma = (
            0.5 * sigma +
            (0.25 * sigma.powi(2) + tau * tau + u_star.powi(2)).sqrt()
        ).sqrt();

        // and momentum
        let t = tau / gamma;
        let s = 1.0 / (1.0 + t * t);

        let u_new = s * (u_prime + (u_prime * t) * t + u_prime.cross(t));
        let gamma = (1.0 + u_new * u_new).sqrt();

        let u_new = FourVector::new(gamma, u_new[0], u_new[1], u_new[2]);
        let r_new = r + SPEED_OF_LIGHT * u_new * dt / gamma;

        (r_new, u_new)
    }

    /// Pseudorandomly emit a photon from an electron with normalized
    /// momentum `u`, which is accelerated by an electric field `E` and
    /// magnetic field `B`.
    #[allow(non_snake_case)]
    #[inline]
    pub fn emit_photon<R: Rng>(u: FourVector, E: ThreeVector, B: ThreeVector, dt: f64, rng: &mut R) -> Option<FourVector> {
        let beta = ThreeVector::from(u) / u[0];
        let E_rf_sqd = (E + SPEED_OF_LIGHT * beta.cross(B)).norm_sqr() - (E * beta).powi(2);
        let chi = if E_rf_sqd > 0.0 {
            u[0] * E_rf_sqd.sqrt() / CRITICAL_FIELD
        } else {
            0.0
        };
        let prob = dt * lcfa::photon_emission::rate(chi, u[0]);
        if rng.gen::<f64>() < prob {
            let (omega_mc2, theta, cphi) = lcfa::photon_emission::sample(
                chi, u[0], rng.gen(), rng.gen(), rng.gen()
            );
            if let Some(theta) = theta {
                let long: ThreeVector = beta.normalize();
                let perp: ThreeVector = long.orthogonal().rotate_around(long, cphi);
                let k: ThreeVector = omega_mc2 * (theta.cos() * long + theta.sin() * perp);
                let k = FourVector::lightlike(k[0], k[1], k[2]);
                Some(k)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Pseudorandomly create an electron-positron pair from a photon with
    /// normalized momentum `u`, in an electric field `E` and
    /// magnetic field `B`.
    #[allow(non_snake_case)]
    #[inline]
    pub fn create_pair<R: Rng>(u: FourVector, E: ThreeVector, B: ThreeVector, dt: f64, rng: &mut R, rate_increase: f64) -> (f64, f64, Option<(FourVector, FourVector)>) {
        let beta = ThreeVector::from(u).normalize();
        let E_rf_sqd = (E + SPEED_OF_LIGHT * beta.cross(B)).norm_sqr() - (E * beta).powi(2);
        let chi = if E_rf_sqd > 0.0 {
            u[0] * E_rf_sqd.sqrt() / CRITICAL_FIELD
        } else {
            0.0
        };
        let prob = dt * lcfa::pair_creation::rate(chi, u[0]);
        let rate_increase = if prob * rate_increase > 0.1 {
            0.1 / prob // limit the rate increase
        } else {
            rate_increase
        };
        if rng.gen::<f64>() < prob * rate_increase {
            let gamma_p = lcfa::pair_creation::sample(chi, u[0], rng);
            let gamma_e = u[0] - gamma_p;
            let u_p = gamma_p * (1.0 - 1.0 / (gamma_p * gamma_p)).sqrt() * beta;
            let u_e = gamma_e * (1.0 - 1.0 / (gamma_e * gamma_e)).sqrt() * beta;
            let u_p = FourVector::new(0.0, u_p[0], u_p[1], u_p[2]).unitize();
            let u_e = FourVector::new(0.0, u_e[0], u_e[1], u_e[2]).unitize();
            (prob, 1.0 / rate_increase, Some((u_e, u_p)))
        } else {
            (prob, 0.0, None)
        }
    }
}

impl Field for FastFocusedLaser {
    fn total_energy(&self) -> f64 {
        0.0
    }

    fn max_timestep(&self) -> Option<f64> {
        Some(0.1 / self.omega())
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

    #[allow(non_snake_case)]
    fn push(&self, r: FourVector, ui: FourVector, rqm: f64, dt: f64) -> (FourVector, FourVector) {
        let (E, B) = self.fields(r);
        FastFocusedLaser::vay_push(r, ui, E, B, rqm, dt)
    }

    #[allow(non_snake_case)]
    fn radiate<R: Rng>(&self, r: FourVector, u: FourVector, dt: f64, rng: &mut R) -> Option<(FourVector, FourVector)> {
        let (E, B) = self.fields(r);
        FastFocusedLaser::emit_photon(u, E, B, dt, rng).map(|k| (k, u - k))
    }

    #[allow(non_snake_case)]
    fn pair_create<R: Rng>(&self, r: FourVector, ell: FourVector, dt: f64, rng: &mut R, rate_increase: f64) -> (f64, f64, Option<(FourVector, FourVector)>) {
        let (E, B) = self.fields(r);
        FastFocusedLaser::create_pair(ell, E, B, dt, rng, rate_increase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_axis() {
        let t_start = -20.0 * 0.8e-6 / (SPEED_OF_LIGHT);
        let dt = 0.25 * 0.8e-6 / (SPEED_OF_LIGHT);
        let laser = FastFocusedLaser::new(100.0, 0.8e-6, 4.0e-6, 30.0e-15, Polarization::Circular);

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