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

mod hilbert;
use hilbert::{ AnalyticalSignal, Bandpass };

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
    bandwidth: f64, // rms, normalised
    field: Vec<Complex64>, // complex electric field, Ex + i Ey
    a_sqd: Vec<f64>, // squared, normalised potential
    dpsi_dphi: Vec<f64>, // local frequency normalised to omega
}

#[derive(PartialEq, Eq)]
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

impl FieldDataError {
    pub fn raise(cause: &str) -> Self {
        FieldDataError { cause: cause.to_owned() }
    }
}

impl FieldData {
    /// Applies a Gaussian filter with standard deviation equal to `r` pixels to the input
    /// array, writing the filtered data to `output`.
    fn gaussian_filter(input: &[f64], output: &mut [f64], r: i32) -> Result<(), FieldDataError> {
        let err = || FieldDataError::raise("gaussian filter");

        let n = input.len();
        if n == 0 || r == 0 || n != output.len() {
            return Err(err());
        }

        let filter: Vec<f64> = (0..4*r)
            .map(|i| {
                let x = (i as f64) / (r as f64);
                (-0.5 * x * x).exp()
            })
            .collect();

        let total: f64 = filter.iter().skip(1).sum();
        let total = 2.0 * total + filter[0];
        let left_pad = input.first().ok_or_else(err)?;
        let right_pad = input.last().ok_or_else(err)?;

        for i in 0..n {
            let mut tmp = 0_f64;
            // go forwards from input[i]
            for j in 0..filter.len() {
                tmp = tmp + filter[j] * input.get(i + j).unwrap_or(right_pad).abs();
            }
            // and then backwards
            for j in 1..filter.len() {
                tmp = tmp + filter[j] * input.get(i - j).unwrap_or(left_pad).abs();
            }
            output[i] = tmp / total;
        }

        Ok(())
    }

    pub fn preprocess(coord: Coordinate, delta: f64, field: &[f64], high_pass: f64, waist: f64, pol: Polarization) -> Result<Self, FieldDataError> {
        // Start by computing the carrier frequency
        let mut planner = FftPlanner::new();
        let mut buffer: Vec<Complex64> = field.iter().map(|ex| ex.into()).collect();
        let n = buffer.len();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

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

        // Determine bandwidth of the pulse
        let bandwidth = {
            let mut num = 0_f64;
            let mut denom = 0_f64;

            for i in 1..n/2 {
                let kappa = (2.0 * consts::PI * (i as f64)) / ((n as f64) * delta);
                let omega_i = match coord {
                    Coordinate::Space => SPEED_OF_LIGHT * kappa,
                    Coordinate::Time => kappa,
                };
                let weight = buffer[i].norm_sqr();
                num = num + (omega_i - omega).powi(2) * weight;
                denom = denom + weight;
            }

            (num / denom).sqrt() / omega
        };

        let dphi = match coord {
            Coordinate::Space => omega * delta / SPEED_OF_LIGHT, // lost minus sign
            Coordinate::Time => omega * delta,
        };

        // phi = omega (t - z/c) => reverse order if function of z
        let mut field = field.to_vec();
        if coord == Coordinate::Space { field.reverse(); }

        field.high_pass_filter(delta, high_pass);

        // In order to get paraxial components, or circular polarisation,
        // we need the analytical signal from the fields, i.e. the complex
        // form E = E0 e^(i omega t) of which Ex = Re(E).
        let complex_field = field.analytical_signal();

        // Integrate over field to get potential
        let e_rel = ELECTRON_MASS * SPEED_OF_LIGHT * omega / ELEMENTARY_CHARGE;

        let mut a: Vec<f64> = Vec::with_capacity(n);
        let mut acc= 0_f64;
        let mut a0 = 0_f64;

        for ex in complex_field.iter() {
            a.push(acc);
            let da = ex.re * dphi / e_rel;
            acc += da;
            if acc > a0 { a0 = acc; }
        }

        // Extract analytical signal from dimensionless potential a = e A / m c
        let a = a.analytical_signal();

        let mut env: Vec<f64> = Vec::with_capacity(n);
        let mut psi: Vec<f64> = Vec::with_capacity(n);

        for a_phi in a.iter() {
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

        // Get local frequency (as fraction of central) by
        // differentiating the instantaneous phase and smoothing
        for i in 1..n {
            psi[i] = (psi_cont[i] - psi_cont[i-1]) / dphi;
        }
        psi[0] = psi[1];

        let r = (1.0 * consts::PI / dphi) as i32;
        let mut dpsi_dphi = psi_cont; // reuse
        Self::gaussian_filter(&psi, &mut dpsi_dphi, r)?;

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
        let energy_flux = complex_field.iter()
            .map(|ex| VACUUM_PERMITTIVITY * ex.re * ex.re * dz)
            .sum();
   
        let params = LaserParameters {
            a0,
            wavelength: 2.0 * consts::PI * SPEED_OF_LIGHT / omega,
            pol,
            pol_angle: 0.0,
            focusing: waist.is_finite(),
            envelope: Envelope::Gaussian,
            waist,
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
            bandwidth,
            field: complex_field,
            a_sqd: env,
            dpsi_dphi,
        })
    }

    pub fn params(&self) -> LaserParameters {
        self.params
    }

    /// Returns a slice of all the electric field values
    #[cfg(feature = "hdf5-output")]
    pub fn electric_field(&self) -> &[Complex64] {
        &self.field
    }

    /// Returns a slice of all the squared potential values
    #[cfg(feature = "hdf5-output")]
    pub fn a_sqd(&self) -> &[f64] {
        &self.a_sqd
    }

    /// Returns a slice of the instantaneous frequency shift
    #[cfg(feature = "hdf5-output")]
    pub fn inst_norm_freq(&self) -> &[f64] {
        &self.dpsi_dphi
    }

    /// Returns the phase difference between adjacent values
    /// of the field, rms potential etc
    #[cfg(feature = "hdf5-output")]
    pub fn phase_step(&self) -> f64 {
        self.step
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

        let field: Vec<f64> = (0..2000)
            .map(|i| {
                let z = dz * ((i as f64) - 1000.0);
                let phi = 2.0 * consts::PI * z / target_lambda;
                let ex = e0 * phi.sin() * (-(z / z0).powi(2)).exp();
                ex
            })
            .collect();

        let laser = FieldData::preprocess(Coordinate::Space,dz, &field, 0.0, std::f64::INFINITY, Polarization::Linear).unwrap();
        let params = laser.params;

        let print_data = false;
        if print_data {
            let mut file = File::create("output/custom_laser_preprocessing.dat").unwrap();
            for i in 0..2000 {
                writeln!(
                    file,
                    "{:.6e} {:.6e} {:.6e} {:.6e}",
                    laser.start + (i as f64) * laser.step, laser.field[i].re, laser.a_sqd[i], laser.dpsi_dphi[i]
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

        let bandwidth =  (0.5 * consts::LN_2).sqrt() / (consts::PI * n_cycles);
        let error = (laser.bandwidth - bandwidth).abs() / bandwidth;

        println!(
            "Got fractional bandwidth of {:.3}%, expected {:.3}% => error = {:.3}%",
            100.0 * laser.bandwidth, 100.0 * bandwidth, 100.0 * error
        );

        assert!(error < 0.02);

        let total: f64 = laser.dpsi_dphi.iter()
            .zip(laser.a_sqd.iter())
            .map(|(f, a)| f * a)
            .sum();
        let mean_freq = total / laser.a_sqd.iter().sum::<f64>();
        let error = (mean_freq - 1.0).abs();

        println!(
            "Got avg. instantaneous frequency of {:.3}% of central => error = {:.3}%",
            100.0 * mean_freq, 100.0 * error
        );

        assert!(error < 0.02);
    }
}