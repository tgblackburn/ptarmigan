use std::f64::consts;
use rand::prelude::*;
use num_complex::Complex;

use crate::field::{Field, Polarization};
use crate::constants::*;
use crate::geometry::{FourVector, ThreeVector, StokesVector};
use crate::lcfa;

use super::{RadiationMode, EquationOfMotion, Envelope};

/// Represents a focusing laser pulse, including
/// the fast oscillating carrier wave
pub struct FastFocusedLaser {
    a0: f64,
    waist: f64,
    duration: f64,
    wavevector: FourVector,
    pol: Polarization,
    envelope: Envelope,
}

impl FastFocusedLaser {
    #[allow(unused)]
    pub fn new(a0: f64, wavelength: f64, waist: f64, n_cycles: f64, pol: Polarization) -> Self {
        let duration = n_cycles * wavelength / SPEED_OF_LIGHT;
        let wavevector = (2.0 * consts::PI / wavelength) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        FastFocusedLaser {
            a0,
            waist,
            duration,
            wavevector,
            pol,
            envelope: Envelope::Gaussian,
        }
    }

    pub fn with_envelope(self, envelope: Envelope) -> Self {
        let mut cpy = self;
        cpy.envelope = envelope;
        cpy
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
    fn beam(&self, r: FourVector, phase: f64) -> (ThreeVector, ThreeVector, ThreeVector, ThreeVector) {
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
        let B0 = E0 / SPEED_OF_LIGHT;

        (
            E0 * ThreeVector::new(Ex.re, Ey.re, Ez.re),
            E0 * ThreeVector::new(Ex.im, Ey.im, Ez.im),
            B0 * ThreeVector::new(Bx.re, By.re, Bz.re),
            B0 * ThreeVector::new(Bx.im, By.im, Bz.im),
        )
    }

    /// Returns the number of wavelengths corresponding to the pulse
    /// duration
    #[inline]
    fn n_cycles(&self) -> f64 {
        SPEED_OF_LIGHT * self.duration * self.wavevector[0] / (2.0 * consts::PI)
    }

    /// Returns the pulse envelope f(ϕ) and its gradient
    /// df(ϕ)/dϕ at the given phase ϕ
    fn envelope_and_grad(&self, phase: f64) -> (f64, f64) {
        match self.envelope {
            Envelope::CosSquared => {
                if phase.abs() < consts::PI * self.n_cycles() {
                    let envelope = (phase / (2.0 * self.n_cycles())).cos().powi(2);
                    (envelope, -1.0 * (phase / (2.0 * self.n_cycles())).tan() * envelope / self.n_cycles())
                } else {
                    (0.0, 0.0)
                }
            },

            Envelope::Flattop => {
                if phase.abs() > consts::PI * (self.n_cycles() + 1.0) {
                    (0.0, 0.0)
                } else if phase.abs() > consts::PI * (self.n_cycles() - 1.0) {
                    let arg = 0.25 * (phase.abs() - (self.n_cycles() - 1.0) * consts::PI);
                    (arg.cos().powi(2), -0.25 * phase.signum() * (2.0 * arg).sin())
                } else {
                    (1.0, 0.0)
                }
            },

            Envelope::Gaussian => {
                let tau = self.omega() * self.duration;
                let envelope = (-2.0 * consts::LN_2 * phase.powi(2) / tau.powi(2)).exp();
                (envelope, -4.0 * consts::LN_2 * phase * envelope / tau.powi(2))
            }
        }
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
        let phase = self.wavevector * r;
        let (f, df_phi) = self.envelope_and_grad(phase);

        // field components from A_x
        let (re_E, im_E, re_B, im_B) = self.beam(r, 0.0);
        // pulsed E = (f - i f') psi e^(i phi) => Re(pulsed E) = f Re(E) + f' Im(E)
        let (E_x, B_x) = (f * re_E + df_phi * im_E, f * re_B + df_phi * im_B);

        // field components from A_y
        let (E_y, B_y) = match self.pol {
            Polarization::Linear => ([0.0; 3].into(), [0.0; 3].into()),
            Polarization::Circular => {
                let axis = ThreeVector::from(self.wavevector).normalize();
                // need to swap definitions of x and y, as well as rotating the E, B vectors
                let r_prime = ThreeVector::from(r).rotate_around(axis, -consts::FRAC_PI_2);
                let r_prime = FourVector::new(r[0], r_prime[0], r_prime[1], r_prime[2]);
                let (re_E, im_E, re_B, im_B) = self.beam(r_prime, 0.0);
                let (E_y, B_y) = (f * im_E - df_phi * re_E, f * im_B - df_phi * re_B);
                (E_y.rotate_around(axis, consts::FRAC_PI_2), B_y.rotate_around(axis, consts::FRAC_PI_2))
            }
        };

        (E_x + E_y, B_x + B_y)
    }

    /// Returns the position and momentum of a particle with charge-to-mass ratio `rqm`,
    /// which has been accelerated in an electric field `E` and magnetic field `B`
    /// over a time interval `dt`.
    /// Assumes that ui is defined at t = 0, and r, E, B are defined at t = dt/2.
    /// If `with_rr` is true, the energy loss due to radiation emission is handled
    /// as part of the particle push, following the classical LL prescription.
    #[allow(non_snake_case)]
    #[inline]
    pub fn vay_push(r: FourVector, ui: FourVector, E: ThreeVector, B: ThreeVector, rqm: f64, dt: f64, eqn: EquationOfMotion) -> (FourVector, FourVector, f64) {
        // velocity in SI units
        let u = ThreeVector::from(ui);
        let gamma = (1.0 + u * u).sqrt(); // enforce mass-shell condition
        let v = SPEED_OF_LIGHT * u / gamma;

        // u_i = u_{i-1/2} + (q dt/2 m c) (E + v_{i-1/2} x B)
        let alpha = rqm * dt / (2.0 * SPEED_OF_LIGHT);
        let u_half = u + alpha * (E + v.cross(B));

        // (classical) radiated momentum
        let u_rad = if eqn.includes_rr() {
            let gamma = (1.0 + u_half * u_half).sqrt();
            let u_half_mag = u_half.norm_sqr().sqrt();
            let beta = u_half / u_half_mag;
            let E_rf_sqd = (E + SPEED_OF_LIGHT * beta.cross(B)).norm_sqr() - (E * beta).powi(2);
            let chi = if E_rf_sqd > 0.0 {
                gamma * E_rf_sqd.sqrt() / CRITICAL_FIELD
            } else {
                0.0
            };
            let power = 2.0 * ALPHA_FINE * chi * chi / (3.0 * COMPTON_TIME);
            let g_chi = match eqn {
                EquationOfMotion::ModifiedLandauLifshitz => lcfa::photon_emission::gaunt_factor(chi),
                _ => 1.0,
            };
            g_chi * power * dt * u_half / u_half_mag
        } else {
            [0.0; 3].into()
        };

        // u' =  u_{i-1/2} + (q dt/2 m c) E
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
        let u_new = u_new - u_rad;
        let gamma = (1.0 + u_new * u_new).sqrt();

        let u_new = FourVector::new(gamma, u_new[0], u_new[1], u_new[2]);
        let r_new = r + 0.5 * SPEED_OF_LIGHT * u_new * dt / gamma;

        (r_new, u_new, dt)
    }

    /// Pseudorandomly emit a photon from an electron with normalized
    /// momentum `u`, which is accelerated by an electric field `E` and
    /// magnetic field `B`.
    #[allow(non_snake_case)]
    #[inline]
    pub fn emit_photon<R: Rng>(u: FourVector, E: ThreeVector, B: ThreeVector, dt: f64, rng: &mut R, classical: bool) -> Option<(FourVector, StokesVector)> {
        let beta = ThreeVector::from(u) / u[0];
        let E_rf_sqd = (E + SPEED_OF_LIGHT * beta.cross(B)).norm_sqr() - (E * beta).powi(2);
        let chi = if E_rf_sqd > 0.0 {
            u[0] * E_rf_sqd.sqrt() / CRITICAL_FIELD
        } else {
            0.0
        };

        let prob = if classical {
            dt * lcfa::photon_emission::classical::rate(chi, u[0])
        } else {
            dt * lcfa::photon_emission::rate(chi, u[0])
        };

        if rng.gen::<f64>() < prob {
            let (omega_mc2, theta, cphi) = if classical {
                lcfa::photon_emission::classical::sample(chi, u[0], rng.gen(), rng.gen(), rng.gen())
            } else {
                lcfa::photon_emission::sample(chi, u[0], rng.gen(), rng.gen(), rng.gen())
            };

            if let Some(theta) = theta {
                let long: ThreeVector = beta.normalize();
                let w = -(E - (long * E) * long / E.norm_sqr().sqrt() + SPEED_OF_LIGHT * beta.cross(B)).normalize();
                let perp: ThreeVector = w.rotate_around(long, cphi);
                let k: ThreeVector = omega_mc2 * (theta.cos() * long + theta.sin() * perp);
                let k = FourVector::lightlike(k[0], k[1], k[2]);
                let pol = if classical {
                    lcfa::photon_emission::classical::stokes_parameters(k, chi, u[0], beta, w)
                } else {
                    lcfa::photon_emission::stokes_parameters(k, chi, u[0], beta, w)
                };
                Some((k, pol))
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
    pub fn create_pair<R: Rng>(u: FourVector, sv: StokesVector, E: ThreeVector, B: ThreeVector, dt: f64, rng: &mut R, rate_increase: f64) -> (f64, f64, StokesVector, Option<(FourVector, FourVector)>) {
        let n = ThreeVector::from(u).normalize();

        // transverse "acceleration"
        let a_perp = E - (E * n) * n + SPEED_OF_LIGHT * n.cross(B);
        let E_rf_sqd = a_perp.norm_sqr();

        let (chi, prob, sv_new) = if E_rf_sqd > 0.0 {
            let chi = u[0] * E_rf_sqd.sqrt() / CRITICAL_FIELD;
            let (prob, sv_new) = lcfa::pair_creation::probability(u, sv, chi, a_perp, dt);
            (chi, prob, sv_new)
        } else {
            (0.0, 0.0, sv)
        };

        let rate_increase = if prob * rate_increase > 0.1 {
            0.1 / prob // limit the rate increase
        } else {
            rate_increase
        };

        if rng.gen::<f64>() < prob * rate_increase {
            let (gamma_p, cos_theta, cphi, _, _) = lcfa::pair_creation::sample(u, sv, chi, a_perp, rng);
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let u_p = gamma_p * (1.0 - 1.0 / (gamma_p * gamma_p)).sqrt();

            // axes
            let e_1 = a_perp.normalize();
            let e_2 = n.cross(e_1);
            let u_p = u_p * (cos_theta * n + sin_theta * cphi.cos() * e_1 + sin_theta * cphi.sin() * e_2);

            // conserving three-momentum
            let u_e = ThreeVector::from(u) - u_p;
            let u_p = FourVector::new(0.0, u_p[0], u_p[1], u_p[2]).unitize();
            let u_e = FourVector::new(0.0, u_e[0], u_e[1], u_e[2]).unitize();
            (prob, 1.0 / rate_increase, sv_new, Some((u_e, u_p)))
        } else {
            (prob, 0.0, sv_new, None)
        }
    }
}

impl Field for FastFocusedLaser {
    fn max_timestep(&self) -> Option<f64> {
        Some(0.1 / self.omega())
    }

    fn contains(&self, r: FourVector) -> bool {
        let phase = self.wavevector * r;
        let max_phase = match self.envelope {
            Envelope::CosSquared => consts::PI * self.n_cycles(),
            Envelope::Flattop => consts::PI * (self.n_cycles() + 1.0),
            Envelope::Gaussian => 6.0 * consts::PI * self.n_cycles(), // = 3 omega tau
        };
        phase < max_phase
    }

    #[allow(non_snake_case)]
    fn push(&self, r: FourVector, ui: FourVector, rqm: f64, dt: f64, eqn: EquationOfMotion) -> (FourVector, FourVector, f64) {
        let r = r + 0.5 * SPEED_OF_LIGHT * ui * dt / ui[0];
        let (E, B) = self.fields(r);
        FastFocusedLaser::vay_push(r, ui, E, B, rqm, dt, eqn)
    }

    #[allow(non_snake_case)]
    fn radiate<R: Rng>(&self, r: FourVector, u: FourVector, dt: f64, rng: &mut R, mode: RadiationMode) -> Option<(FourVector, StokesVector, FourVector, f64)> {
        let (E, B) = self.fields(r);
        let a = ELEMENTARY_CHARGE * E.norm_sqr().sqrt() / (ELECTRON_MASS * SPEED_OF_LIGHT * self.omega());
        FastFocusedLaser::emit_photon(u, E, B, dt, rng, mode == RadiationMode::Classical)
            .map(|(k, pol)| (k, pol, u - k, a))
    }

    #[allow(non_snake_case)]
    fn pair_create<R: Rng>(&self, r: FourVector, ell: FourVector, pol: StokesVector, dt: f64, rng: &mut R, rate_increase: f64) -> (f64, f64, StokesVector, Option<(FourVector, FourVector, f64)>) {
        let (E, B) = self.fields(r);
        let a = ELEMENTARY_CHARGE * E.norm_sqr().sqrt() / (ELECTRON_MASS * SPEED_OF_LIGHT * self.omega());
        let (prob, frac, pol_new, momenta) = FastFocusedLaser::create_pair(ell, pol, E, B, dt, rng, rate_increase);
        (prob, frac, pol_new, momenta.map(|(p1, p2)| (p1, p2, a)))
    }

    fn ideal_initial_z(&self) -> f64 {
        let wavelength = 2.0 * consts::PI / self.wavevector[0];
        match self.envelope {
            Envelope::CosSquared => 0.5 * wavelength * self.n_cycles(),
            Envelope::Flattop => 0.5 * wavelength * (self.n_cycles() + 1.0),
            Envelope::Gaussian => 2.0 * wavelength * self.n_cycles(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_axis() {
        let t_start = -20.0 * 0.8e-6 / (SPEED_OF_LIGHT);
        let n_cycles = 10.0; // SPEED_OF_LIGHT * 30.0e-15 / 0.8e-6;
        let laser = FastFocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Circular)
            .with_envelope(Envelope::Gaussian);
        let dt = laser.max_timestep().unwrap();

        let mut u = FourVector::new(0.0, 0.0, 0.0, -1000.0).unitize();
        let mut r = FourVector::new(0.0, 0.0, 0.0, 0.0) + u * SPEED_OF_LIGHT * t_start / u[0];

        while laser.contains(r) {
            let new = laser.push(r, u, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
            r = new.0;
            u = new.1;
        }

        println!("final u_perp = ({:.3e}, {:.3e}), u^2 = {:.3e}", u[1], u[2], u * u);
        assert!(u[1].abs() < 1.0e-3);
        assert!(u[2].abs() < 1.0e-3);
        assert!((u * u - 1.0).abs() < 1.0e-3);
    }
}