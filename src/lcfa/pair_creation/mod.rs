//! Nonlinear pair creation, gamma -> e- + e+, in a background field

use rand::prelude::*;
use crate::constants::*;
use crate::geometry::*;

mod angular_spectra;
mod energy_spectrum;
mod tables;

/// Routines for a handling a nonlinear Breit-Wheeler pair creation event,
/// as considered within the LCFA.
pub struct PairCreation {
    ell: FourVector,
    sv: StokesVector,
    cos_2theta: f64,
    sin_2theta: f64,
    chi: f64,
    dv1: f64,
    uncertainty: f64,
}

impl PairCreation {
    /// Sets up a nonlinear Breit-Wheeler pair creation event for a photon with four-momentum `ell`
    /// and Stokes parameter `sv` in a constant, crossed field.
    /// The field is defined by the transverse acceleration `a_perp`, quantum
    /// parameter `chi`, and duration `dt` (in seconds).
    pub fn new(ell: FourVector, sv: StokesVector, chi: f64, a_perp: ThreeVector) -> Self {
        let (sv, cos_2theta, sin_2theta) = sv.in_basis(a_perp, ell.into());

        Self {
            ell,
            sv,
            cos_2theta,
            sin_2theta,
            chi,
            dv1: 0.0,
            uncertainty: 0.0,
        }
    }

    /// Sets the uncertainty in the pair creation rate to be a fraction of the leading order correction
    /// to the LCFA, which depends on the derivative term `dv1 = (3 e.e'' + e'.e') / [45 (e.e)^2]`.
    pub fn with_uncertainty(self, dv1: f64, frac: f64) -> Self {
        Self {
            dv1,
            uncertainty: frac,
            ..self
        }
    }

    /// Returns the value of the auxiliary function T for photons that are polarized parallel,
    /// and perpendicular to, the instantaneous acceleration (respectively), as well as the
    /// value of Y, the auxiliary function that controls derivative corrections.
    fn auxiliary_t_and_y(chi: f64) -> (f64, f64, f64) {
        use tables::*;
        if chi <= 0.01 {
            // if chi < 5e-3, T(chi) < 1e-117, so ignore
            // 3.0 * 3.0f64.sqrt() / (8.0 * consts::SQRT_2) * (-4.0 / (3.0 * chi)).exp()
            (0.0, 0.0, 0.0)
        } else if chi < 100.0 {
            let i = (chi.ln() - MIN_LN_CHI) / DELTA_LN_CHI;
            let dx = i.fract();
            let i = i as usize;

            let lower = &SCALED_T_Y_TABLE[i];
            let upper = &SCALED_T_Y_TABLE[i+1];

            let prefactor = -8.0 / (3.0 * chi);

            let t_par = (prefactor + (1.0 - dx) * lower[0] + dx * upper[0]).exp();
            let t_perp = (prefactor + (1.0 - dx) * lower[1] + dx * upper[1]).exp();
            let y = prefactor.exp() * (((1.0 - dx) * lower[2] + dx * upper[2]).exp() + tables::Y_SHIFT);

            (t_par, t_perp, y)
        } else {
            // use asymptotic expression, which is accurate to better than 0.3%
            // for chi > 100:
            //   T(x) = [C - C_1 x^(-2/3)] x^(-1/3)
            // where <C> = 5 Gamma(5/6) (2/3)^(1/3) / [14 Gamma(7/6)] and C_1 = 2/3
            let t_par = 0.3036898468348568 / chi.cbrt() - 2.0 / (3.0 * chi);
            let t_perp = 0.4555347702522852 / chi.cbrt() - 2.0 / (3.0 * chi);

            // Much less accurate
            let y = 0.41757353939792824 * chi.powf(2.0 / 3.0);

            (t_par, t_perp, y)
        }
    }

    /// Returns the probability that pair creation occurs in the given time interval,
    /// and the Stokes parameters the photon should have if it does not.
    pub fn probability(&self, dt: f64) -> (f64, StokesVector) {
        let gamma = self.ell[0];

        let parallel_proj = 0.5 * (1.0 + self.sv[1]);
        let perp_proj = 0.5 * (1.0 - self.sv[1]);

        let (t_par, t_perp, y) = PairCreation::auxiliary_t_and_y(self.chi);

        let prob = ALPHA_FINE * self.chi * (parallel_proj * t_par + perp_proj * t_perp) * dt / (COMPTON_TIME * gamma);
        let delta_prob = ALPHA_FINE * y * self.uncertainty * self.dv1 * dt / (COMPTON_TIME * gamma);

        // If pair creation does not occur, photon Stokes parameters should be changed to:
        let sv: StokesVector = {
            let prob_avg = ALPHA_FINE * self.chi * 0.5 * (t_par + t_perp) * dt / (COMPTON_TIME * gamma);
            let delta = ALPHA_FINE * self.chi * 0.5 * (t_par - t_perp) * dt / (COMPTON_TIME * gamma);

            let sv1 = (self.sv[1] * (1.0 - prob_avg) - delta) / (1.0 - prob);
            let sv2 = self.sv[2] * (1.0 - prob_avg) / (1.0 - prob);

            // transform to simulation basis, rotating by -theta
            [
                self.sv[0],
                self.cos_2theta * sv1 + self.sin_2theta * sv2,
                -self.sin_2theta * sv1 + self.cos_2theta * sv2,
                self.sv[3] * (1.0 - prob_avg) / (1.0 - prob),
            ].into()
        };

        (prob + delta_prob, sv)
    }

    fn sample_internal<R: Rng>(&self, rng: &mut R) -> (f64, f64, f64) {
        // Rejection sampling for s
        let spectrum = energy_spectrum::Spectrum::new(self.chi, self.sv[1]);
        let max = spectrum.ceiling();
        let s = loop {
            let s = rng.gen::<f64>();
            let u = rng.gen::<f64>();
            let f = spectrum.value(s);
            if u <= f / max {
                break s;
            }
        };

        // Now that s is fixed, sample from the angular spectrum
        let spectrum = angular_spectra::Spectrum::new(s, self.chi, self.sv);
        let max = spectrum.polar_ceiling();
        let z = loop {
            let y = rng.gen::<f64>();
            let u = rng.gen::<f64>();
            let f = spectrum.polar(y);
            if u <= f / max {
                break 1.0 + 4.0 * self.chi * y * y;
            }
        };

        let phi = spectrum.sample_azimuthal(z, rng);

        (s, z, phi)
    }

    /// Samples the nonlinear Breit-Wheeler spectrum, returning the positron Lorentz factor,
    /// the cosine of the scattering angle, and the azimuthal angle.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> (f64, f64, f64) {
        let gamma = self.ell[0];
        let (s, z, phi) = self.sample_internal(rng);

        // recall z = 2 gamma^2 (1 - beta cos_theta), where
        // beta = sqrt(1 - 1/gamma^2), so cos_theta is close
        // to (2 gamma^2 - z^(2/3)) / (2 gamma^2 - 1)
        // note that gamma here is the positron gamma
        let gamma_p = s * gamma;
        let cos_theta = (2.0 * gamma_p * gamma_p - z.powf(2.0/3.0)) / (2.0 * gamma_p * gamma_p - 1.0);
        let cos_theta = cos_theta.max(-1.0);

        (gamma_p, cos_theta, phi)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use crate::quadrature::*;
    use super::*;

    #[test]
    fn lcfa_rate() {
        let max_error = 1.0e-2;

        let pts = [
            (0.042, 6.077538994929929904e-29),
            (0.105, 2.1082097875655204834e-12),
            (0.42,  0.00037796132366581330636),
            (1.05,  0.015977478443872017101),
            (4.2,   0.08917816786414408900),
            (12.0,  0.10884579479913803705),
            (42.0,  0.09266735324318656466),
        ];

        for (chi, target) in &pts {
            let (t_par, t_perp, _) = PairCreation::auxiliary_t_and_y(*chi);
            let result = 0.5 * (t_par + t_perp);

            let prefactor = 1.0 / (3_f64.sqrt() * std::f64::consts::PI * chi);
            let spectrum = energy_spectrum::Spectrum::new(*chi, 0.0);
            let intgd: f64 = GAUSS_32_NODES.iter()
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(t, w)| {
                    let s = 0.5 * (1.0 + t);
                    prefactor * 0.5 * w * spectrum.value(s)
                })
                .sum();

            let error = (result - target).abs() / target;
            let intgd_error = (intgd - target).abs() / target;
            println!("chi = {:>9.3e}, t(chi) = {:>12.6e} | {:>12.6e} [interp|intgd], error = {:.3e} | {:.3e}", chi, result, intgd, error, intgd_error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn lcfa_rate_pol_resolved() {
        let pts = [
            0.15,
            0.75,
            1.2,
            8.7,
            43.0,
            225.0,
        ];

        for chi in &pts {
            let (target_t_par, target_t_perp, _) = PairCreation::auxiliary_t_and_y(*chi);

            let spectrum = energy_spectrum::Spectrum::new(*chi, 1.0);
            let t_par: f64 = GAUSS_32_NODES.iter()
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(t, w)| {
                    let s = 0.5 * (1.0 + t);
                    w * spectrum.value (s)
                })
                .sum();

            let spectrum = energy_spectrum::Spectrum::new(*chi, -1.0);
            let t_perp: f64 = GAUSS_32_NODES.iter()
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(t, w)| {
                    let s = 0.5 * (1.0 + t);
                    w * spectrum.value(s)
                })
                .sum();

            let target = target_t_par / target_t_perp;
            let result = t_par / t_perp;
            let error = (target - result).abs() / target;
            println!("chi = {:.3e}, expected T_parallel / T_perp = {:.3}, error = {:.3}%", chi, target, 100.0 * error);
            assert!(error < 1.0e-2);
        }
    }

    #[test]
    #[ignore]
    fn pair_spectrum_sampling() {
        let s1 = 0.0;
        let chi = 2.0;
        let gamma = 1000.0;
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let path = format!("output/lcfa_pair_spectrum_{}_{}.dat", chi, s1);
        let mut file = File::create(path).unwrap();
        for _i in 0..200000 {
            let event = PairCreation::new(
                [gamma, 0.0, 0.0, -gamma].into(),
                [1.0, s1, 0.0, 0.0].into(),
                chi,
                [1.0, 0.0, 0.0].into()
            );
            let (s, z, phi) = event.sample_internal(&mut rng);
            assert!(s > 0.0 && s < 1.0);
            assert!(z >= 1.0);
            writeln!(file, "{:.6e} {:.6e} {:.6e}", s, z, phi).unwrap();
        }
    }

    #[test]
    fn pure_states_remain_so() {
        let ell: FourVector = [100.0, 0.0, 0.0, -100.0].into();
        let a_perp: ThreeVector = [1.0, 0.0, 0.0].into();

        let svs: [StokesVector; 2] = [
            [1.0,  1.0,  0.0,  0.0].into(),
            [1.0, -1.0,  0.0,  0.0].into(),
        ];

        for sv in &svs {
            let event = PairCreation::new(ell, *sv, 1.0, a_perp);
            let (_, sv_new) = event.probability(1.0e-6 / SPEED_OF_LIGHT);
            assert!((sv[1] - sv_new[1]).abs() < 1.0e-12);
        }
    }

    fn positron_yield<R: Rng>(chi_max: f64, gamma: f64, sv1: f64, bias: f64, change_sv1: bool, rng: &mut R) -> f64 {
        let tau = 10.0;
        let dt = 1.0e-6 / (20.0 * SPEED_OF_LIGHT);
        let ell: FourVector = [gamma, 0.0, 0.0, -gamma].into();
        let a_perp: ThreeVector = [1.0, 0.0, 0.0].into();
        let mut sv: StokesVector = [1.0, sv1, 0.0, 0.0].into();
        let mut weight = 1.0;
        let mut count = 0.0;
        for i in -300..300 {
            let t = (i as f64) * 0.05;
            let chi = chi_max * (-(t/tau).powi(2)).exp();
            let event = PairCreation::new(ell, sv, chi, a_perp);
            let (prob, sv_new) = event.probability(dt);
            let bias = if prob * bias > 0.1 { 0.1 / prob } else { bias };
            let prob = prob * bias;
            if rng.gen::<f64>() < prob {
                count += weight / bias;
                weight -= weight / bias;
                if weight <= 0.0 {
                    break;
                }
            }
            if change_sv1 {
                sv = sv_new;
            }
        }
        count
    }

    #[test]
    #[ignore]
    fn polarization_change() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let n = 50_000;
        let gamma = 1000.0;
        // let (chi_max, bias) = (1.0, 1.0);
        let (chi_max, bias) = (0.5, 100.0);

        // incoherent sum of sigma and pi
        let mut ps = vec![];
        for _bx in 0..10 {
            let mut count = 0.0;
            for _i in 0..n {
                count += positron_yield(chi_max, gamma, 1.0, bias, false, &mut rng);
                count += positron_yield(chi_max, gamma, -1.0, bias, false, &mut rng);
            }
            ps.push(count / (2.0 * n as f64));
        }

        let mean = ps.iter().sum::<f64>() / 10.0;
        let sdev = ps.iter().map(|p| (p - mean).powi(2)).sum::<f64>().sqrt() / 10.0;

        // unpolarized (correct)
        let mut ps = vec![];
        for _bx in 0..10 {
            let mut count = 0.0;
            for _i in 0..(2 * n) {
                count += positron_yield(chi_max, gamma, 0.0, bias, true, &mut rng);
            }
            ps.push(count / (2.0 * n as f64));
        }

        let mean2 = ps.iter().sum::<f64>() / 10.0;
        let sdev2 = ps.iter().map(|p| (p - mean2).powi(2)).sum::<f64>().sqrt() / 10.0;

        // unpolarized (incorrect)
        let mut ps = vec![];
        for _bx in 0..10 {
            let mut count = 0.0;
            for _i in 0..(2 * n) {
                count += positron_yield(chi_max, gamma, 0.0, bias, false, &mut rng);
            }
            ps.push(count / (2.0 * n as f64));
        }

        let mean3 = ps.iter().sum::<f64>() / 10.0;
        let sdev3 = ps.iter().map(|p| (p - mean3).powi(2)).sum::<f64>().sqrt() / 10.0;

        println!("50-50 pure = {:.6e} ± {:.6e}, 100 mixed = {:.6e} ± {:.6e}, 100 mixed != {:.6e} ± {:.6e}", mean, sdev, mean2, sdev2, mean3, sdev3);

        let diff = mean - mean2;
        let width = sdev.hypot(sdev2);
        let diff3 = mean - mean3;
        let width3 = sdev.hypot(sdev3);

        println!("diff = {:.6e} ± {:.6e} [{:.2} sigma] or {:.6e} ± {:.6e} [{:.2} sigma]", diff, width, diff.abs() / width, diff3, width3, diff3.abs() / width3);
        assert!(diff.abs() < width);
    }
}
