use std::f64::consts;
use rand::prelude::*;

use crate::field::{Field, Polarization, FastFocusedLaser};
use crate::constants::*;
use crate::geometry::{FourVector, ThreeVector, StokesVector};

use super::{RadiationMode, EquationOfMotion, RadiationEvent, Envelope};

/// Represents a plane-wave laser pulse, including the
/// fast oscillating carrier wave
pub struct FastPlaneWave {
    a0: f64,
    n_cycles: f64,
    wavevector: FourVector,
    pol: Polarization,
    pol_angle: f64,
    chirp_b: f64,
    envelope: Envelope,
}

impl FastPlaneWave {
    #[allow(unused)]
    pub fn new(a0: f64, wavelength: f64, n_cycles: f64, pol: Polarization, pol_angle: f64, chirp_b: f64) -> Self {
        let wavevector = (2.0 * consts::PI / wavelength) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        FastPlaneWave {
            a0,
            n_cycles,
            wavevector,
            pol,
            pol_angle: match pol {
                Polarization::Circular => 0.0,
                Polarization::Linear => pol_angle,
            },
            chirp_b,
            envelope: Envelope::CosSquared,
        }
    }

    pub fn with_envelope(self, envelope: Envelope) -> Self {
        let mut cpy = self;
        cpy.envelope = envelope;
        cpy
    }

    #[allow(unused)]
    fn k(&self) -> FourVector {
        self.wavevector
    }

    fn omega(&self) -> f64 {
        SPEED_OF_LIGHT * self.wavevector[0]
    }

    /// The electric and magnetic fields of a pulsed plane wave
    /// at four position `r`.
    #[allow(non_snake_case)]
    fn fields(&self, r: FourVector) -> (ThreeVector, ThreeVector) {
        // A^mu = (m c a0 / e) {0, sin(phi), delta cos(phi), 0} f(phi)
        // where delta = 0 for LP and 1 for CP
        // E = -d_t A => E = -omega d_phi (A_x, A_y, 0)
        // B = curl A = (-d_z A_y, d_z A_x, 0) => (omega/c) d_phi (A_y, -A_x, 0)

        let delta = match self.pol {
            Polarization::Linear => 0.0f64,
            Polarization::Circular => 1.0f64,
        };

        let phi: f64 = self.wavevector * r;

        // psi is the (potentially time-dependent) carrier phase
        let (psi, dpsi_dphi) = if cfg!(feature = "compensating-chirp") && self.envelope == Envelope::CosSquared {
            let beta = self.chirp_b * 0.5 * (1.0 + delta.powi(2)) * self.a0.powi(2);
            let f = (phi / (2.0 * self.n_cycles)).cos().powi(2);
            (
                phi + (beta / 16.0) * (6.0 * phi + 8.0 * self.n_cycles * (phi / self.n_cycles).sin() + self.n_cycles * (2.0 * phi / self.n_cycles).sin()),
                1.0 + beta * f * f,
            )
        } else {
            (
                phi * (1.0 + self.chirp_b * phi),
                1.0 + 2.0 * self.chirp_b * phi,
            )
        };

        // envelope and gradient
        let (f, df_dphi) = match self.envelope {
            Envelope::CosSquared => {
                if phi.abs() < self.n_cycles * consts::PI {
                    (
                        (phi / (2.0 * self.n_cycles)).cos().powi(2),
                        -(phi / self.n_cycles).sin() / (2.0 * self.n_cycles)
                    )
                } else {
                    (0.0, 0.0)
                }
            }

            Envelope::Flattop => {
                if phi.abs() > consts::PI * (self.n_cycles + 1.0) {
                    (0.0, 0.0)
                } else if phi.abs() > consts::PI * (self.n_cycles - 1.0) {
                    let arg = 0.25 * (phi.abs() - (self.n_cycles - 1.0) * consts::PI);
                    (arg.cos().powi(2), -phi.signum() * 0.25 * (2.0 * arg).sin())
                } else {
                    (1.0, 0.0)
                }
            },

            Envelope::Gaussian => {
                let arg = -0.5 * (phi / (consts::PI * self.n_cycles)).powi(2);
                (
                    arg.exp2(),
                    -consts::LN_2 * phi * arg.exp2() / (consts::PI * self.n_cycles).powi(2)
                )
            }
        };

        // a = A / (m c a0 / e):
        let dax_dphi = psi.sin() * df_dphi + psi.cos() * dpsi_dphi * f;
        let day_dphi = delta * (psi.cos() * df_dphi - psi.sin() * dpsi_dphi * f);

        let amplitude = (ELECTRON_MASS * SPEED_OF_LIGHT_SQD * self.wavevector[0] * self.a0) / ELEMENTARY_CHARGE;
        let E = -amplitude * ThreeVector::new(dax_dphi, day_dphi, 0.0);
        let B = (amplitude / SPEED_OF_LIGHT) * ThreeVector::new(day_dphi, -dax_dphi, 0.0);

        let E = E.rotate_around_z(self.pol_angle);
        let B = B.rotate_around_z(self.pol_angle);

        (E, B)
    }

    /// Calculates the integrated intensity, i.e. the power per unit area,
    /// by numerically integration.
    fn integrated_intensity(&self, points_per_wavelength: i32) -> f64 {
        let max_phase = match self.envelope {
            Envelope::CosSquared => consts::PI * self.n_cycles,
            Envelope::Flattop => consts::PI * (self.n_cycles + 1.0),
            Envelope::Gaussian => 6.0 * consts::PI * self.n_cycles,
        };

        let dphi = 2.0 * consts::PI / (points_per_wavelength as f64);
        let dz = dphi / self.wavevector[0];
        let n_pts = (max_phase / dphi) as i64;
        let mut energy_flux = 0.0;
        let mut c = 0.0;

        for n in -n_pts..n_pts {
            let phi = (n as f64) * dphi;
            let r: FourVector = [0.0, 0.0, 0.0, -phi / self.wavevector[0]].into();
            let (e, b) = self.fields(r);
            let u = 0.5 * VACUUM_PERMITTIVITY * (e.norm_sqr() + SPEED_OF_LIGHT_SQD * b.norm_sqr());

            let weight = if n % 2 == 0 { 2.0 / 3.0 } else { 4.0 / 3.0 };
            // let weight = 1.0;
            let y = weight * u * dz - c;
            let t = energy_flux + y;
            c = (t - energy_flux) - y;
            energy_flux = t;
            // energy_flux += u * dz;
        }

        energy_flux
    }
}

impl Field for FastPlaneWave {
    fn max_timestep(&self) -> Option<f64> {
        let chirp = if cfg!(feature = "compensating-chirp") {
            1.0 + self.a0.powi(2)
        } else {
            1.0 + 2.0 * self.chirp_b * consts::PI * self.n_cycles
        };
        let dt = 1.0 / (SPEED_OF_LIGHT * self.wavevector[0] * chirp);
        let multiplier = (3_f64.sqrt() / (5.0 * ALPHA_FINE * self.a0)).min(0.1);
        Some(dt * multiplier)
    }

    fn contains(&self, r: FourVector) -> bool {
        let phase = self.wavevector * r;
        let max_phase = match self.envelope {
            Envelope::CosSquared => consts::PI * self.n_cycles,
            Envelope::Flattop => consts::PI * (self.n_cycles + 1.0),
            Envelope::Gaussian => 6.0 * consts::PI * self.n_cycles, // = 3 omega tau
        };
        phase < max_phase
    }

    #[allow(non_snake_case)]
    fn push(&self, r: FourVector, ui: FourVector, rqm: f64, dt: f64, eqn: EquationOfMotion) -> (FourVector, FourVector, f64, f64) {
        let r = r + 0.5 * SPEED_OF_LIGHT * ui * dt / ui[0];
        let (E, B) = self.fields(r);
        FastFocusedLaser::vay_push(r, ui, E, B, rqm, dt, eqn)
    }

    #[allow(non_snake_case)]
    fn radiate<R: Rng>(&self, r: FourVector, u: FourVector, dt: f64, rng: &mut R, mode: RadiationMode) -> Option<RadiationEvent> {
        let (E, B) = self.fields(r);
        let a = ELEMENTARY_CHARGE * E.norm_sqr().sqrt() / (ELECTRON_MASS * SPEED_OF_LIGHT * self.omega());
        FastFocusedLaser::emit_photon(u, E, B, dt, rng, mode == RadiationMode::Classical)
            .map(|(k, pol)| {
                RadiationEvent {
                    k,
                    u_prime: u - k,
                    pol,
                    a_eff: a,
                    absorption: 0.0
                }
            })
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
            Envelope::CosSquared => 0.5 * wavelength * self.n_cycles,
            Envelope::Flattop => 0.5 * wavelength * (self.n_cycles + 1.0),
            Envelope::Gaussian => 2.0 * wavelength * self.n_cycles,
        }
    }

    fn energy(&self) -> (f64, &'static str) {
        if self.chirp_b != 0.0 || cfg!(feature = "compensating-chirp") {
            let ppw = 1.0 + 2.0 * consts::PI * self.chirp_b * self.n_cycles;
            let ppw = (10.0 * ppw) as i32;
            return (self.integrated_intensity(ppw), "J/m^2");
        }

        let intensity = {
            let amplitude = (ELECTRON_MASS * SPEED_OF_LIGHT * self.omega() * self.a0) / ELEMENTARY_CHARGE;
            SPEED_OF_LIGHT * VACUUM_PERMITTIVITY * amplitude.powi(2)
        };

        let delta = match self.pol {
            Polarization::Linear => 0.0,
            Polarization::Circular => 1.0,
        };

        let duration = match self.envelope {
            Envelope::CosSquared => {
                let phase = (1.0 + 3.0 * self.n_cycles.powi(2)) * consts::PI / (8.0 * self.n_cycles);
                (1.0 + delta) * phase / self.omega()
            },
            Envelope::Gaussian => {
                let (phase_x, phase_y) = {
                    let arg = -(consts::PI * self.n_cycles).powi(2) / consts::LN_2;
                    let large_n_contr = 0.5 * self.n_cycles * (consts::PI.powi(3) / consts::LN_2).sqrt();
                    (
                        large_n_contr - arg.exp_m1() * (consts::LN_2 / consts::PI).sqrt() / (4.0 * self.n_cycles),
                        large_n_contr + (1.0 + arg.exp()) * (consts::LN_2 / consts::PI).sqrt() / (4.0 * self.n_cycles)
                    )
                };
                (phase_x + delta * phase_y) / self.omega()
            },
            Envelope::Flattop => {
                let phase = (self.n_cycles - 3.0 / 16.0) * consts::PI;
                (1.0 + delta) * phase / self.omega()
            },
        };

        (intensity * duration, "J/m^2")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_wave_cp() {
        //use std::fs::File;
        //use std::io::Write;

        let n_cycles = 8.0;
        let wavelength = 0.8e-6;
        let t_start = -0.5 * n_cycles * wavelength / (SPEED_OF_LIGHT);
        let dt = 0.005 * 0.8e-6 / (SPEED_OF_LIGHT);
        let a0 = 100.0;
        let laser = FastPlaneWave::new(a0, wavelength, n_cycles, Polarization::Circular, 0.0, 0.0)
            .with_envelope(Envelope::CosSquared);

        let mut u = FourVector::new(0.0, 0.0, 0.0, -100.0).unitize();
        let mut r = FourVector::new(0.0, 0.0, 0.0, 0.0) + u * SPEED_OF_LIGHT * t_start / u[0];
        
        let mut u_perp_max = 0.0;
        let mut phase_max = 0.0;

        //let mut file = File::create("output/fast_plane_wave.dat").unwrap();

        for _k in 0..2 {
            for _i in 0..800 {
                let new = laser.push(r, u, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
                r = new.0;
                u = new.1;
                let u_perp = u[1].hypot(u[2]);
                let phase = laser.k() * r;
                if u_perp > u_perp_max {
                    u_perp_max = u_perp;
                    phase_max = phase;
                }
                //writeln!(file, "{:.6e} {:.6e} {:.6e}", phase, u[1], u[2]).unwrap();
            }
        }

        let err = (u_perp_max - a0).abs() / a0;
        println!("max u_perp = {:.3e}, occured at phi = {:.3e}, err = {:.3e}", u_perp_max, phase_max, err);

        assert!(err < 1.0e-3);
        assert!(u[1] < 1.0e-3);
        assert!(u[2] < 1.0e-3);
        assert!((u * u - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn energy_flux() {
        let n_cycles = 2.0;
        let wavelength = 0.8e-6;
        let a0 = 10.0;
        let pol = Polarization::Circular;

        for envelope in [Envelope::CosSquared, Envelope::Gaussian, Envelope::Flattop].iter() {
            let laser = FastPlaneWave::new(a0, wavelength, n_cycles, pol, 0.0, 0.0)
                .with_envelope(*envelope);

            let (energy, energy_unit) = laser.energy();
            let numerical_energy = laser.integrated_intensity(10);
            let error = (energy - numerical_energy).abs() / numerical_energy;

            println!(
                "Laser energy ({:?}, {}) = {:.6e} [analytical] {:.6e} [integrated] => error = {:.3e}",
                envelope, energy_unit, energy, numerical_energy, error
            );

            assert!(error < 1.0e-4);
        }
    }

    #[test]
    fn depletion() {
        let n_cycles = 8.0;
        let wavelength = 0.8e-6;
        let a0 = 20.0;
        let pol = Polarization::Linear;

        let laser = FastPlaneWave::new(a0, wavelength, n_cycles, pol, 0.0, 0.0)
            .with_envelope(Envelope::Gaussian);

        let z0 = laser.ideal_initial_z();
        let dt = 0.1 * laser.max_timestep().unwrap();

        let mut r = FourVector::new(-z0, 0.0, 0.0, z0);
        let mut u = FourVector::new(0.0, 0.0, 0.0, -1000.0).unitize();
        let mut work = 0.0;

        while laser.contains(r) {
            let (r_new, u_new, _, dwork) = laser.push(r, u, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::LandauLifshitz);
            r = r_new;
            u = u_new;
            work += dwork;
        }

        let expected_work = {
            let phase = consts::PI.powf(1.5) * laser.n_cycles / 4_f64.ln().sqrt();
            let omega_mc2 = SPEED_OF_LIGHT * COMPTON_TIME * laser.wavevector[0];
            let delta = match pol { Polarization::Circular => 1.0, Polarization::Linear => 1.0 / 8.0 };
            ALPHA_FINE * a0.powi(4) * omega_mc2 * delta * phase / 3.0
        };

        let error = (work - expected_work).abs() / expected_work;

        println!("LL + LCFA: work/mc^2 = {:.6e} [numerical], {:.6e} [analytical] => error = {:.3e}", work, expected_work, error);

        assert!(error < 1.0e-3);
    }
}