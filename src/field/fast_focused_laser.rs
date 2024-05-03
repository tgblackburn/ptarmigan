use std::f64::consts;
use num_complex::Complex;

use crate::field::{Field, Polarization};
use crate::constants::*;
use crate::geometry::{FourVector, ThreeVector};

use super::Envelope;

/// Represents a focusing laser pulse, including
/// the fast oscillating carrier wave
pub struct FastFocusedLaser {
    a0: f64,
    waist: f64,
    duration: f64,
    wavevector: FourVector,
    pol: Polarization,
    pol_angle: f64,
    envelope: Envelope,
}

impl FastFocusedLaser {
    #[allow(unused)]
    pub fn new(a0: f64, wavelength: f64, waist: f64, n_cycles: f64, pol: Polarization, pol_angle: f64) -> Self {
        let duration = n_cycles * wavelength / SPEED_OF_LIGHT;
        let wavevector = (2.0 * consts::PI / wavelength) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        FastFocusedLaser {
            a0,
            waist,
            duration,
            wavevector,
            pol,
            pol_angle: match pol {
                Polarization::Circular => 0.0,
                Polarization::Linear => pol_angle,
            },
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
        let r = r.rotate_around_z(-self.pol_angle);
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
            E0 * ThreeVector::new(Ex.re, Ey.re, Ez.re).rotate_around_z(self.pol_angle),
            E0 * ThreeVector::new(Ex.im, Ey.im, Ez.im).rotate_around_z(self.pol_angle),
            B0 * ThreeVector::new(Bx.re, By.re, Bz.re).rotate_around_z(self.pol_angle),
            B0 * ThreeVector::new(Bx.im, By.im, Bz.im).rotate_around_z(self.pol_angle),
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
}

impl Field for FastFocusedLaser {
    fn max_timestep(&self) -> Option<f64> {
        let dt = 1.0 / self.omega();
        let multiplier = (3_f64.sqrt() / (5.0 * ALPHA_FINE * self.a0)).min(0.1);
        Some(dt * multiplier)
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

    fn ideal_initial_z(&self) -> f64 {
        let wavelength = 2.0 * consts::PI / self.wavevector[0];
        match self.envelope {
            Envelope::CosSquared => 0.5 * wavelength * self.n_cycles(),
            Envelope::Flattop => 0.5 * wavelength * (self.n_cycles() + 1.0),
            Envelope::Gaussian => 2.0 * wavelength * self.n_cycles(),
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
    fn fields(&self, r: FourVector) -> (ThreeVector, ThreeVector, f64) {
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

        let E = E_x + E_y;
        let B = B_x + B_y;
        let a = ELEMENTARY_CHARGE * E.norm_sqr().sqrt() / (ELECTRON_MASS * SPEED_OF_LIGHT * self.omega());

        (E, B, a)
    }

    fn energy(&self) -> (f64, &'static str) {
        let intensity = {
            let amplitude = (ELECTRON_MASS * SPEED_OF_LIGHT * self.omega() * self.a0) / ELEMENTARY_CHARGE;
            SPEED_OF_LIGHT * VACUUM_PERMITTIVITY * amplitude.powi(2)
        };

        let power = {
            let p0 = 0.5 * consts::PI * intensity * self.waist.powi(2);
            let e = self.waist / self.rayleigh_range();
            p0 * (1.0 + e * e / 4.0 + e.powi(4) / 8.0)
        };

        let delta = match self.pol {
            Polarization::Linear => 0.0,
            Polarization::Circular => 1.0,
        };

        let n_cycles = self.n_cycles();

        let duration = match self.envelope {
            Envelope::CosSquared => {
                let phase = (1.0 + 3.0 * n_cycles.powi(2)) * consts::PI / (8.0 * n_cycles);
                (1.0 + delta) * phase / self.omega()
            },
            Envelope::Gaussian => {
                let (phase_x, phase_y) = {
                    let arg = -(consts::PI * n_cycles).powi(2) / consts::LN_2;
                    let large_n_contr = 0.5 * n_cycles * (consts::PI.powi(3) / consts::LN_2).sqrt();
                    (
                        large_n_contr - arg.exp_m1() * (consts::LN_2 / consts::PI).sqrt() / (4.0 * n_cycles),
                        large_n_contr + (1.0 + arg.exp()) * (consts::LN_2 / consts::PI).sqrt() / (4.0 * n_cycles)
                    )
                };
                (phase_x + delta * phase_y) / self.omega()
            },
            Envelope::Flattop => {
                let phase = (n_cycles - 3.0 / 16.0) * consts::PI;
                (1.0 + delta) * phase / self.omega()
            },
        };

        (power * duration, "J")
    }
}

#[cfg(test)]
mod tests {
    use crate::EquationOfMotion;
    use super::*;

    #[test]
    fn on_axis() {
        let t_start = -20.0 * 0.8e-6 / (SPEED_OF_LIGHT);
        let n_cycles = 10.0; // SPEED_OF_LIGHT * 30.0e-15 / 0.8e-6;
        let laser = FastFocusedLaser::new(100.0, 0.8e-6, 4.0e-6, n_cycles, Polarization::Circular, 0.0)
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

    #[test]
    fn energy() {
        let expected_energy = 1.2_f64;
        let wavelength = 0.8e-6;
        let n_cycles = SPEED_OF_LIGHT * 30.0e-15 / wavelength;
        let a0 = 3.0;
        let waist = 147.839 * expected_energy.sqrt() * wavelength / (a0 * 30_f64.sqrt()); // from LUXE input file
        let pol = Polarization::Circular;
        let envelope = Envelope::Gaussian;

        let laser = FastFocusedLaser::new(a0, wavelength, waist, n_cycles, pol, 0.0)
            .with_envelope(envelope);

        let (energy, energy_unit) = laser.energy();
        let error = (energy - expected_energy).abs() / expected_energy;

        println!(
            "Laser energy ({:?}, {}) = {:.6e} [analytical] {:.6e} [expected] => error = {:.3e}",
            envelope, energy_unit, energy, expected_energy, error,
        );

        assert!(error < 1.0e-3);
    }
}