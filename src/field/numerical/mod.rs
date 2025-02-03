//! Electromagnetic fields that are defined on a mesh

use std::f64::consts;
use std::error::Error;
use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::constants::*;

use super::properties::*;

mod fast_plane_wave;
pub use fast_plane_wave::NumericalFastPW;

mod plane_wave;
pub use plane_wave::NumericalPW;

/// Data necessary to define a custom field structure
#[allow(unused)]
#[derive(Clone)]
pub struct FieldData {
    params: LaserParameters,
    start: f64, // phase
    step: f64, // phase
    end: f64, // phase
    omega: f64, // angular frequency
    energy_flux: f64,
    field: Vec<f64>, // electric field
    a_sqd: Vec<f64>, // squared, normalised potential
    psi: Vec<f64>, // local frequency shift
}

pub enum Coordinate {
    Space,
    Time,
}

pub struct FieldDataError {
    pub cause: String
}

impl std::fmt::Debug for FieldDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed to preprocess custom laser due to {}", self.cause)
    }
}

impl std::fmt::Display for FieldDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for FieldDataError {}

impl FieldData {
    pub fn preprocess(coord: Coordinate, delta: f64, field: &[f64]) -> Result<Self, FieldDataError> {
        // Start by computing the carrier frequency
        let mut planner = FftPlanner::new();
        let mut buffer: Vec<Complex64> = field.iter().map(|ex| Complex64::new(*ex, 0.0)).collect();
        let n = buffer.len();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Take the opportunity to remove any DC component
        buffer[0] = Complex64::new(0.0, 0.0);

        let omega = {
            let mut kappa = 0.0;
            let mut max = 0.0;

            for i in 0..n/2 {
                let tmp = buffer[i].norm_sqr();
                if tmp > max {
                    max = tmp;
                    kappa = (2.0 * consts::PI * (i as f64)) / ((n as f64) * delta);
                }
            }

            match coord {
                Coordinate::Space => SPEED_OF_LIGHT * kappa,
                Coordinate::Time => kappa,
            }
        };

        let dphi = match coord {
            Coordinate::Space => omega * delta / SPEED_OF_LIGHT, // lost minus sign
            Coordinate::Time => omega * delta,
        };

        // FFT backwards
        let fft = planner.plan_fft_inverse(n);
        fft.process(&mut buffer);
        for ex in buffer.iter_mut() {
            *ex /= n as f64;
        }

        // phi = omega (t - z/c) => reverse order if function of z
        let mut field = match coord {
            Coordinate::Space => buffer.into_iter().rev().collect(),
            Coordinate::Time => buffer,
        };

        let processed_field: Vec<f64> = field.iter().map(|ex| ex.re).collect();

        // Integrate over field to get potential
        let e_rel = ELECTRON_MASS * SPEED_OF_LIGHT * omega / ELEMENTARY_CHARGE;
        let mut a = Complex64::new(0.0, 0.0);
        let mut a0 = 0.0;

        for ex in field.iter_mut() {
            let da= *ex * dphi / e_rel;
            *ex = a;
            a += da;
            if a.re > a0 { a0 = a.re; }
        }

        let mut a = field; // dimensionless potential a = e A / m c

        // Extract analytical signal!

        // First, FFT forwards to get a(omega)
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut a);

        // Zero out negative frequency components
        for i in 1..n/2 {
            a[i] *= 2.0;
        }
        for i in (n/2 + 1)..n {
            a[i] *= 0.0;
        }

        // Go backwards to a(t), get envelope and instantaneous phase
        let fft = planner.plan_fft_inverse(n);
        fft.process(&mut a);

        let mut env: Vec<f64> = Vec::with_capacity(n);
        let mut psi: Vec<f64> = Vec::with_capacity(n);

        for a_phi in a.iter_mut() {
            *a_phi /= n as f64;
            env.push(a_phi.norm_sqr());
            psi.push(a_phi.arg());
        }

        // Phase is restricted to -pi, pi, i.e. not continuous
        let mut psi_cont = psi.clone();
        for i in 1..n {
            let diff = if psi[i] < 0.0 && psi[i-1] > 0.0 {
                psi[i] - psi[i-1] + 2.0 * consts::PI
            } else {
                psi[i] - psi[i-1]
            };
            psi_cont[i] = psi_cont[i-1] + diff;
        }

        let diff = 0.5 * (psi_cont[n-1] - psi_cont[0]);
        for phi in psi_cont.iter_mut() {
            *phi -= diff;
        }

        // Find FWHM in terms of phase
        let n_cycles = {
            let index = env.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap_or(0);

            let env_max = env[index];
            let mut i_plus = n;
            for i in index..n {
                if env[i] < 0.5 * env_max { i_plus = i; break; }
            }
            let mut i_minus = 0;
            for i in (0..=index).rev() {
                if env[i] < 0.5 * env_max { i_minus = i; break; }
            }

            let delta_phi = ((i_plus - i_minus) as f64) * dphi;
            delta_phi / (2.0 * consts::PI)
        };

        // Get energy flux from electric field
        let dz = SPEED_OF_LIGHT * dphi / omega;
        let energy_flux = processed_field.iter()
            .map(|ex| VACUUM_PERMITTIVITY * ex * ex * dz)
            .sum();
   
        let params = LaserParameters {
            a0,
            wavelength: 2.0 * consts::PI * SPEED_OF_LIGHT / omega,
            pol: Polarization::Linear,
            pol_angle: 0.0,
            focusing: false,
            envelope: Envelope::Gaussian,
            waist: std::f64::INFINITY,
            n_cycles,
            chirp_b: 0.0
        };

        Ok(Self {
            params,
            start: -0.5 * (n as f64) * dphi,
            step: dphi,
            end: 0.5 * (n as f64) * dphi,
            omega,
            energy_flux,
            field: processed_field,
            a_sqd: env,
            psi: psi_cont,
        })
    }

    pub fn params(&self) -> LaserParameters {
        self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocessing() {
        use std::fs::File;
        use std::io::Write;

        let dz = 0.05e-6;
        let target_lambda = 0.8e-6;
        let e0 = 2.0 * consts::PI * ELECTRON_MASS * SPEED_OF_LIGHT_SQD / (ELEMENTARY_CHARGE * target_lambda);
        let z0 = 8.0e-6;

        let field: Vec<f64> = (0..1000)
            .map(|i| {
                let z = dz * ((i as f64) - 500.0);
                let phi = 2.0 * consts::PI * z / target_lambda;
                let ex = e0 * phi.sin() * (-(z / z0).powi(2)).exp();
                ex
            })
            .collect();

        let laser = FieldData::preprocess(Coordinate::Space,dz, &field).unwrap();
        let params = laser.params;

        let print_data = true;
        if print_data {
            let mut file = File::create("output/custom_laser_preprocessing.dat").unwrap();
            for i in 0..1000 {
                writeln!(
                    file,
                    "{:.6e} {:.6e} {:.6e} {:.6e}",
                    laser.start + (i as f64) * laser.step, laser.field[i], laser.a_sqd[i], laser.psi[i]
                ).unwrap();
            }
        }

        let error = (params.wavelength - target_lambda).abs() / target_lambda;

        println!(
            "Got wavelength of {:.3} um, expected {:.3} um => error = {:.3}%",
            1.0e6 * params.wavelength, 1.0e6 * target_lambda, 100.0 * error
        );

        assert!(error < 0.02);

        let error = (params.a0 - 1.0).abs();

        println!(
            "Got a0 of {:.3}, expected {:.3} => error = {:.3}%",
            params.a0, 1.0, 100.0 * error
        );

        assert!(error < 0.02);

        let n_cycles = (2.0 * consts::LN_2).sqrt() * z0 / target_lambda;
        let error = (params.n_cycles - n_cycles).abs() / n_cycles;

        println!(
            "Got n_cycles of {:.3}, expected {:.3} => error = {:.3}%",
            params.n_cycles, n_cycles, 100.0 * error
        );

        assert!(error < 0.02);
    }
}