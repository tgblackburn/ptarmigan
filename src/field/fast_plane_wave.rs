use std::f64::consts;
use rand::prelude::*;

use crate::field::{Field, Polarization, FastFocusedLaser};
use crate::constants::*;
use crate::geometry::{FourVector, ThreeVector};

/// Represents a plane-wave laser pulse, including the
/// fast oscillating carrier wave
pub struct FastPlaneWave {
    a0: f64,
    n_cycles: f64,
    wavevector: FourVector,
    pol: Polarization,
}

impl FastPlaneWave {
    #[allow(unused)]
    pub fn new(a0: f64, wavelength: f64, n_cycles: f64, pol: Polarization) -> Self {
        let wavevector = (2.0 * consts::PI / wavelength) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        FastPlaneWave {
            a0,
            n_cycles,
            wavevector,
            pol
        }
    }

    #[allow(unused)]
    fn k(&self) -> FourVector {
        self.wavevector
    }

    /// The electric and magnetic fields of a pulsed plane wave
    /// at four position `r`.
    #[allow(non_snake_case)]
    fn fields(&self, r: FourVector) -> (ThreeVector, ThreeVector) {
        // A^mu = (m c a0 / e) {0, sin(phi), delta cos(phi), 0} cos[phi/(2n)]^2
        // where delta = 0 for LP and 1 for CP
        // E = -d_t A => E = -omega d_phi (A_x, A_y, 0)
        // B = curl A = (-d_z A_y, d_z A_x, 0) => (omega/c) d_phi (A_y, -A_x, 0)
        let delta = match self.pol {
            Polarization::Linear => 0.0,
            Polarization::Circular => 1.0,
        };
        let phi = self.wavevector * r;
        let envelope = if phi.abs() < self.n_cycles * consts::PI {
            (phi / (2.0 * self.n_cycles)).cos().powi(2)
        } else {
            0.0
        };
        let amplitude = (ELECTRON_MASS * SPEED_OF_LIGHT_SQD * self.wavevector[0] * self.a0) / ELEMENTARY_CHARGE;
        let E = -amplitude * envelope * ThreeVector::new(
            phi.cos() - phi.sin() * (phi / (2.0 * self.n_cycles)).tan() / (self.n_cycles as f64),
            -delta * (phi.sin() + phi.cos() * (phi / (2.0 * self.n_cycles)).tan() / self.n_cycles),
            0.0,
        );
        let B = -(amplitude * envelope / SPEED_OF_LIGHT) * ThreeVector::new(
            delta * (phi.sin() + phi.cos() * (phi / (2.0 * self.n_cycles)).tan() / (self.n_cycles as f64)),
            phi.cos() - phi.sin() * (phi / (2.0 * self.n_cycles)).tan() / self.n_cycles,
            0.0,
        );

        (E, B)
    }
}

impl Field for FastPlaneWave {
    fn total_energy(&self) -> f64 {
        0.0
    }

    fn max_timestep(&self) -> Option<f64> {
        Some( 0.1 / (SPEED_OF_LIGHT * self.wavevector[0]) )
    }

    fn contains(&self, r: FourVector) -> bool {
        let phase = self.wavevector * r;
        phase < consts::PI * self.n_cycles
    }

    #[allow(non_snake_case)]
    fn push(&self, r: FourVector, ui: FourVector, rqm: f64, dt: f64) -> (FourVector, FourVector) {
        let (E, B) = self.fields(r);
        FastFocusedLaser::vay_push(r, ui, E, B, rqm, dt)
    }

    #[allow(non_snake_case)]
    fn radiate<R: Rng>(&self, r: FourVector, u: FourVector, dt: f64, rng: &mut R) -> Option<FourVector> {
        let (E, B) = self.fields(r);
        FastFocusedLaser::emit_photon(u, E, B, dt, rng)
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
        let dt = 0.01 * 0.8e-6 / (SPEED_OF_LIGHT);
        let a0 = 100.0;
        let laser = FastPlaneWave::new(a0, wavelength, n_cycles, Polarization::Circular);

        let mut u = FourVector::new(0.0, 0.0, 0.0, -100.0).unitize();
        let mut r = FourVector::new(0.0, 0.0, 0.0, 0.0) + u * SPEED_OF_LIGHT * t_start / u[0];
        
        let mut u_perp_max = 0.0;
        let mut phase_max = 0.0;

        //let mut file = File::create("output/fast_plane_wave.dat").unwrap();

        for _k in 0..2 {
            for _i in 0..400 {
                let new = laser.push(r, u, ELECTRON_CHARGE / ELECTRON_MASS, dt);
                u = new.1;
                let u_perp = u[1].hypot(u[2]);
                // vay push leapfrogs r and u
                let phase = 0.5 * laser.k() * (r + new.0);
                if u_perp > u_perp_max {
                    u_perp_max = u_perp;
                    phase_max = phase;
                }
                r = new.0;
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
}