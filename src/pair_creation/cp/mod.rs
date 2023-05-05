//! Nonlinear Breit-Wheeler pair creation in CP backgrounds

use std::f64::consts;
use num_complex::Complex64;
use rand::prelude::*;

#[cfg(test)]
use rayon::prelude::*;

use crate::constants::ALPHA_FINE;
use crate::geometry::StokesVector;
use crate::special_functions::*;
use crate::quadrature::*;

// Stokes parameter of the background field
const LASER_S3: f64 = -1.0;

/// Lookup table for total rate, for S_3 = ±1
mod total;

/// Represents the double-differential partial rate, `d^2 W_n / (ds dθ)`,
/// at harmonic order n.
/// Multiply by `ɑ / η` to get `dP_n/(ds dθ dphase)`.
struct DoubleDiffPartialRate {
    n: i32,
    a: f64,
    eta: f64,
}

impl DoubleDiffPartialRate {
    /// Constructs the double-differential pair-creation spectrum at the given
    /// harmonic order `n`.
    fn new(n: i32, a: f64, eta: f64) -> Self {
        Self { n, a, eta }
    }

    fn s_bounds(&self) -> (f64, f64) {
        let sn = 2.0 * (self.n as f64) * self.eta / (1.0 + self.a.powi(2));
        let s_min = 0.5 * (1.0 - (1.0 - 4.0 / sn).sqrt());
        let s_min = s_min.min(0.5);
        (s_min, 0.5)
    }

    /// The double-differential rate contains three contributions:
    /// one polarization-averaged term, one combining S_1 and S_2, and
    /// one for S_3.
    fn elements(&self, s: f64) -> (f64, f64, f64) {
        let ell = self.n as f64;
        let sn = 2.0 * ell * self.eta / (1.0 + self.a.powi(2));

        let z = {
            let tmp = 1.0 / (sn * s * (1.0 - s));
            ((4.0 * ell * ell) * (self.a.powi(2) / (1.0 + self.a.powi(2))) * tmp * (1.0 - tmp)).sqrt()
        };

        let (j_nm1, j_n, j_np1) = z.j_pm(self.n);

        let unpol = j_n.powi(2) + 0.25 * self.a.powi(2) * ((1.0 - s) / s + s / (1.0 - s)) * (j_nm1.powi(2) + j_np1.powi(2) - 2.0 * j_n.powi(2));
        let delta_1_2 = (2.0 * (ell * self.a / z).powi(2) - (1.0 + self.a.powi(2))) * j_n.powi(2) - 0.5 * self.a.powi(2) * (j_nm1.powi(2) + j_np1.powi(2));
        let delta_3 = 0.25 * self.a.powi(2) * ((1.0 - s) / s + s / (1.0 - s)) * (1.0 - 2.0 / (sn * s * (1.0 - s))) * (j_nm1.powi(2) - j_np1.powi(2));

        (unpol, delta_1_2, delta_3)
    }

    /// Calculates the pair creation rate, differential in s (fractional lightfront momentum transfer)
    /// and theta (azimuthal angle), for a partially polarized photon with Stokes parameters
    /// S_1, S_2 and S_3.
    fn fully_resolved(&mut self, s: f64, theta: f64, sv1: f64, sv2: f64, sv3: f64) -> f64 {
        let (unpol, delta_1_2, delta_3) = self.elements(s);
        let (cos_2theta, sin_2theta) = (2.0 * theta).sin_cos();
        (unpol - (sv1 * cos_2theta + sv2 * sin_2theta) * delta_1_2 - LASER_S3 *  sv3 * delta_3) / (2.0 * consts::PI)
    }

    fn integrate_over_theta(&self, s: f64) -> Complex64 {
        let (unpol, _, delta_3) = self.elements(s);
        Complex64::new(
            unpol - LASER_S3 * delta_3,
            unpol + LASER_S3 * delta_3,
        )
    }

    fn integrate_over_theta_with_s3(&self, s: f64, sv3: f64) -> f64 {
        let sp = self.integrate_over_theta(s);
        // re => sv3 = +1, im => sv3 = -1
        0.5 * (sp.re * (1.0 + sv3) + sp.im * (1.0 - sv3))
    }

    /// Integrates the double-differential partial rate over the entire domain
    /// `0 < θ < 2π` and `s_min < s < s_max`, returning the result for the
    /// two polarization components (+1, -1) as a single complex number.
    /// Multiply by `ɑ / η` to get `dP_n/dphase`.
    fn integrate(&self) -> Complex64 {
        let (n, a, eta) = (self.n, self.a, self.eta);
        let ell = n as f64;
        let sn = 2.0 * ell * eta / (1.0 + a * a);

        // equivalent to n > 2 (1 + a^2) / eta
        if sn <= 4.0 {
            return 0_f64.into()
        }

        // approx position at which probability is maximised
        let beta = a.powi(4) * (1.0/ell + 1.0).powi(2) + 16.0 * (a * a - 2.0).powi(2) / (sn * sn) - 8.0 * a * a * (a * a - 2.0) / sn;
        let beta = beta.sqrt() / (a * a - 2.0);
        let alpha = (a * a + 2.0 * ell) / (ell * (2.0 - a * a)) - 4.0 / sn;
        let tmp = alpha + beta;
        let s_peak = 0.5 * (1.0 - tmp.sqrt());

        let (s_min, s_max) = self.s_bounds();
        // println!("alpha = {}, beta = {}, tmp = {}, s_peak = {:.6}, s_min = {:.6e}", alpha, beta, tmp, s_peak, s_min);

        let pr = if s_peak.is_finite() {
            let s_mid = 2.0 * s_peak - s_min;
            if s_mid > s_max {
                // do integral in two stages, from s_min to s_peak and then
                // s_peak to s_max
                let lower: Complex64 = GAUSS_32_NODES.iter()
                    .map(|x| 0.5 * (s_peak - s_min) * x + 0.5 * (s_min + s_peak))
                    .zip(GAUSS_32_WEIGHTS.iter())
                    .map(|(s, w)| {
                        let sp = self.integrate_over_theta(s);
                        // println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                        0.5 * (s_peak - s_min) * w * sp
                    })
                    .sum();

                let upper: Complex64 = GAUSS_32_NODES.iter()
                    .map(|x| 0.5 * (s_max - s_peak) * x + 0.5 * (s_peak + s_max))
                    .zip(GAUSS_32_WEIGHTS.iter())
                    .map(|(s, w)| {
                        let sp = self.integrate_over_theta(s);
                        // println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                        0.5 * (s_max - s_peak) * w * sp
                    })
                    .sum();

                2.0 * (upper + lower)
            } else {
                // do integral in three stages, from s_min to s_peak,
                // s_peak to s_mid, and s_mid to s_max
                let lower: Complex64 = GAUSS_32_NODES.iter()
                    .map(|x| 0.5 * (s_peak - s_min) * x + 0.5 * (s_min + s_peak))
                    .zip(GAUSS_32_WEIGHTS.iter())
                    .map(|(s, w)| {
                        let sp = self.integrate_over_theta(s);
                        // println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                        0.5 * (s_peak - s_min) * w * sp
                    })
                    .sum();

                let middle: Complex64 = GAUSS_32_NODES.iter()
                    .map(|x| 0.5 * (s_mid - s_peak) * x + 0.5 * (s_peak + s_mid))
                    .zip(GAUSS_32_WEIGHTS.iter())
                    .map(|(s, w)| {
                        let sp = self.integrate_over_theta(s);
                        // println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                        0.5 * (s_mid - s_peak) * w * sp
                    })
                    .sum();

                let upper: Complex64 = GAUSS_32_NODES.iter()
                    .map(|x| 0.5 * (s_max - s_mid) * x + 0.5 * (s_mid + s_max))
                    .zip(GAUSS_32_WEIGHTS.iter())
                    .map(|(s, w)| {
                        let sp = self.integrate_over_theta(s);
                        // println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                        0.5 * (s_max - s_mid) * w * sp
                    })
                    .sum();

                2.0 * (lower + middle + upper)
            }
        } else {
            let total: Complex64 = GAUSS_32_NODES.iter()
                .map(|x| 0.5 * (s_max - s_min) * x + 0.5 * (s_min + s_max))
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(s, w)| {
                    let sp = self.integrate_over_theta(s);
                    // println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                    0.5 * (s_max - s_min) * w * sp
                })
                .sum();
            2.0 * total
        };

        pr
    }

    /// Samples the double-differential spectrum, returning a pseudorandomly selected
    /// lightfront-momentum fraction `s` and azimuthal angle `theta`.
    fn sample<R: Rng>(&mut self, sv1: f64, sv2: f64, sv3: f64, mut rng: R) -> (f64, f64) {
        let (s_min, s_max) = self.s_bounds();

        // Approximate maximum value of the probability density:
        let max: f64 = GAUSS_32_NODES.iter()
            .map(|x| 0.5 * (s_max - s_min) * x + 0.5 * (s_min + s_max)) // from x in [-1,1] to s in [s_min, smax]
            .map(|s| self.integrate_over_theta_with_s3(s, sv3))
            .fold(0.0f64 / 0.0f64, |a: f64, b: f64| a.max(b));
        let max = 1.5 * max;

        // Rejection sampling for s:
        let s = loop {
            let s = s_min + (s_max - s_min) * rng.gen::<f64>();
            let z = max * rng.gen::<f64>();
            let f = self.integrate_over_theta_with_s3(s, sv3);
            if z < f {
                break s;
            }
        };

        // Fix s, which is [s_min, 1/2] at the moment
        let s = match rng.gen_range(0, 2) {
            0 => 1.0 - s,
            1 => s,
            _ => unreachable!(),
        };

        // Now handle azmimuthal angle theta:

        // At fixed s, the spectrum = A - B (S_1 cos2theta + S_2 sin2theta),
        // which is maximised at tan(2 theta) = S_2 / S_1
        let theta_opt = 0.5 * sv2.atan2(sv1);
        let max = (0..4)
            .map(|i| {
                let theta = theta_opt + (i as f64) * consts::FRAC_PI_2;
                self.fully_resolved(s, theta, sv1, sv2, sv3)
            })
            .reduce(f64::max)
            .map(|max| 1.01 * max)
            .unwrap();

        let theta = loop {
            let theta = 2.0 * consts::PI * rng.gen::<f64>();
            let z = max * rng.gen::<f64>();
            let f = self.fully_resolved(s, theta, sv1, sv2, sv3);
            if z < f {
                break theta;
            }
        };

        (s, theta)
    }
}

/// Represents the total pair-creation rate, i.e. [DoubleDiffPartialRate]
/// integrated over all s and θ, then summed over all n.
/// Multiply by `ɑ / η` to get `Σ_n dP_n/dphase`.
pub(super) struct TotalRate {
    a: f64,
    eta: f64,
}

impl TotalRate {
    pub(super) fn new(a: f64, eta: f64) -> Self {
        Self {
            a,
            eta,
        }
    }

    /// Returns the sum, over harmonic index, of the partial nonlinear
    /// Breit-Wheeler rates. Implemented as a table lookup.
    /// The rate is a function of the third Stokes parameter, S_3,
    /// which determines degree of circular polarization.
    fn value(&self, sv3: f64) -> f64 {
        let f = if self.a < 0.02 || self.is_too_small() {
            [0.0, 0.0]
        } else if self.a < total::LN_MIN_A.exp() {
            self.by_summation()
        } else {
            self.by_lookup()
        };

        0.5 * (f[0] * (1.0 + sv3) + f[1] * (1.0 - sv3))
    }

    /// Returns the probability that pair creation occurs in a phase
    /// interval `dphi`, as well as the Stokes parameters of the photon.
    /// These must change whether pair creation occurs or not to avoid biased results.
    pub(super) fn probability(&self, sv: StokesVector, dphi: f64) -> (f64, StokesVector) {
        let [f0, f1] = if self.a < 0.02 || self.is_too_small() {
            [0.0, 0.0]
        } else if self.a < total::LN_MIN_A.exp() {
            self.by_summation()
        } else {
            self.by_lookup()
        };

        let prob = {
            let f = 0.5 * ((f0 + f1) + sv[3] * (f0 - f1));
            ALPHA_FINE * f * dphi / self.eta
        };

        let prob_avg = {
            let f = 0.5 * (f0 + f1);
            ALPHA_FINE * f * dphi / self.eta
        };

        let delta = {
            let f = 0.5 * (f0 - f1);
            ALPHA_FINE * f * dphi / self.eta
        };

        let sv1 = sv[1] * (1.0 - prob_avg) / (1.0 - prob);
        let sv2 = sv[2] * (1.0 - prob_avg) / (1.0 - prob);
        let sv3 = (sv[3] * (1.0 - prob_avg) - delta) / (1.0 - prob);

        (prob, [sv[0], sv1, sv2, sv3].into())
    }

    /// Returns a pseudorandomly sampled n (harmonic order), s (lightfront momentum
    /// transfer) and theta (azimuthal angle in the ZMF) for a pair creation event that
    /// occurs at normalized amplitude a and energy parameter eta.
    pub(super) fn sample<R: Rng>(&self, sv1: f64, sv2: f64, sv3: f64, rng: &mut R) -> (i32, f64, f64) {
        let a = self.a;
        let eta = self.eta;

        let n = {
            let (n_min, n_max) = self.sum_limits();
            let target = self.value(sv3);
            let target = target * rng.gen::<f64>();
            let mut cumsum: f64 = 0.0;
            let mut index: i32 = -1; // invalid harmonic order
            for k in n_min..=n_max {
                let pr = DoubleDiffPartialRate::new(k, a, eta).integrate();
                let pr = 0.5 * ((1.0 + sv3) * pr.re + (1.0 - sv3) * pr.im);
                cumsum += pr;
                if cumsum > target {
                    index = k;
                    break;
                }
            };
            // interpolation errors mean that even after the sum, cumsum could be < target
            if index == -1 {
                eprintln!("pair_creation::generate failed to sample a harmonic order (a = {:.3e}, eta = {:.3e}, {} <= n < {}), falling back to {}.", a, eta, n_min, n_max, n_max - 1);
                index = n_max - 1;
            }
            assert!(index >= n_min && index <= n_max);
            index
        };

        let mut spectrum = DoubleDiffPartialRate::new(n, a, eta);
        let (s, theta) = spectrum.sample(sv1, sv2, sv3, rng);

        (n, s, theta)
    }

    /// Returns the range of harmonics that contribute to the total rate.
    fn sum_limits(&self) -> (i32, i32) {
        let (a, eta) = (self.a, self.eta);
        let n_min = (2.0 * (1.0 + a * a) / eta).ceil();
        let range = 2.0 * (0.25 + a * a) / eta + 13.0 * a * (2.0 + a);

        let test = 0.25 - (1.0 + a * a) / (2.0 * (n_min as f64) * eta);
        if test <= f64::EPSILON {
            ((n_min as i32) + 1, (n_min + 1.0 + range) as i32)
        } else {
            (n_min as i32, (n_min + range) as i32)
        }
    }

    /// Checks if a and eta are small enough such that the rate < exp(-200)
    fn is_too_small(&self) -> bool {
        self.eta.log10() < -1.0 - (self.a.log10() + 2.0).powi(2) / 4.5
    }

    #[cfg(test)]
    fn by_parallel_summation(&self) -> [f64; 2] {
        let (n_min, n_max) = self.sum_limits();

        let mut rates: Vec<(i32, Complex64)> = (n_min..=n_max).into_par_iter()
            .map(|n| (n, DoubleDiffPartialRate::new(n, self.a, self.eta).integrate()))
            .collect();

        // cumulative sum
        let mut total = Complex64::new(0.0, 0.0);
        for (_, pr) in rates.iter_mut() {
            total = total + *pr;
            *pr = total;
        }

        [total.re, total.im]
    }

    fn by_summation(&self) -> [f64; 2] {
        let (n_min, n_max) = self.sum_limits();

        let total: Complex64 = (n_min..=n_max)
            .map(|n| DoubleDiffPartialRate::new(n, self.a, self.eta).integrate())
            .sum();

        [total.re, total.im]
    }

    fn by_lookup(&self) -> [f64; 2] {
        use total::*;

        let (a, eta) = (self.a, self.eta);
        let (x, y) = (a.ln(), eta.ln());

        if x < LN_MIN_A {
            panic!("NBW [CP] rate lookup out of bounds: a = {:.3e}, eta = {:.3e}", a, eta);
        } else if y < LN_MIN_ETA {
            [0.0, 0.0]
        } else {
            let ix = ((x - LN_MIN_A) / LN_A_STEP) as usize;
            let iy = ((y - LN_MIN_ETA) / LN_ETA_STEP) as usize;
            if ix < N_COLS - 1 && iy < N_ROWS - 1 {
                if a * eta > 2.0 {
                    // linear interpolation of: log y against log x, best for power law
                    let dx = (x - LN_MIN_A) / LN_A_STEP - (ix as f64);
                    let dy = (y - LN_MIN_ETA) / LN_ETA_STEP - (iy as f64);
                    let w = [
                        (1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy,
                    ];
                    let pt = [
                        TABLE[iy][ix], TABLE[iy][ix+1], TABLE[iy+1][ix], TABLE[iy+1][ix+1]
                    ];
                    let f = [
                        (w[0] * pt[0][0] + w[1] * pt[1][0] + w[2] * pt[2][0] + w[3] * pt[3][0]).exp(),
                        (w[0] * pt[0][1] + w[1] * pt[1][1] + w[2] * pt[2][1] + w[3] * pt[3][1]).exp(),
                    ];
                    f
                } else {
                    // linear interpolation of: 1 / log y against x, best for exp(-1/x)?
                    let a_min = (LN_MIN_A + (ix as f64) * LN_A_STEP).exp();
                    let a_max = (LN_MIN_A + ((ix+1) as f64) * LN_A_STEP).exp();
                    let eta_min = (LN_MIN_ETA + (iy as f64) * LN_ETA_STEP).exp();
                    let eta_max = (LN_MIN_ETA + ((iy+1) as f64) * LN_ETA_STEP).exp();
                    let dx = (a - a_min) / (a_max - a_min);
                    let dy = (eta - eta_min) / (eta_max - eta_min);
                    let w = [
                        (1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy,
                    ];
                    let pt = [
                        TABLE[iy][ix], TABLE[iy][ix+1], TABLE[iy+1][ix], TABLE[iy+1][ix+1]
                    ];
                    let f = [
                        w[0] / pt[0][0] + w[1] / pt[1][0] + w[2] / pt[2][0] + w[3] / pt[3][0],
                        w[0] / pt[0][1] + w[1] / pt[1][1] + w[2] / pt[2][1] + w[3] / pt[3][1],
                    ];
                    [(1.0 / f[0]).exp(), (1.0 / f[1]).exp()]
                }
            } else {
                panic!("NBW [CP] rate lookup out of bounds: a = {:.3e}, eta = {:.3e}", a, eta);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_rates() {
        let max_error = 1.0e-6;

        // n, a, eta, target
        let pts = [
            // At chi = a eta = 0.1:
            (50,   1.0, 0.1,  1.4338498960396018931e-18),
            (100,  1.0, 0.1,  1.3654528056555865291e-18),
            (200,  1.0, 0.1,  2.0884399327604375975e-34),
            (4000, 5.0, 0.02, 9.2620552570985535880e-61),
            (5000, 5.0, 0.02, 5.9413657401296979089e-17),
            (6000, 5.0, 0.02, 1.1629168979497463847e-18),
            (8000, 5.0, 0.02, 1.6722921069034930599e-23),
            (3,    0.1, 1.0,  2.7926363804797338348e-7),
            (5,    0.1, 1.0,  1.4215822192894496185e-10),
            (20,   0.1, 1.0,  1.1028787567587238051e-37),
            // At chi = a eta = 1:
            (3000, 10.0, 0.1, 2.8532353822421244101e-48),
            (4000, 10.0, 0.1, 1.4356925580571873594e-5),
            (8000, 10.0, 0.1, 1.5847977567888444504e-7),
            (6,    1.0,  1.0, 0.0031666996194000280745),
            (20,   1.0,  1.0, 1.2280171339973043031e-5),
            (50,   1.0,  1.0, 7.7268893728561315057e-11),
            // At chi = a eta = 10:
            (300,  10.0, 1.0, 1.6230747656967905300e-7),
            (400,  10.0, 1.0, 0.0031791285538880908649),
            (2000, 10.0, 1.0, 5.9533991784012215406e-5),
            // At chi = a eta = 0.01:
            (640,  1.0, 0.01, 9.7351336009776642509e-115),
            (1000, 1.0, 0.01, 2.3257373741363993054e-156),
            (25,   0.1, 0.1,  5.7778053795802739886e-52),
            (50,   0.1, 0.1,  3.3444371706672986244e-90),
        ];

        for (n, a, eta, target) in &pts {
            let spectrum = DoubleDiffPartialRate::new(*n, *a, *eta);
            let result = spectrum.integrate();
            let delta = 0.5 * (result.im - result.re);
            let result = 0.5 * (result.re + result.im);
            let error = (target - result).abs() / target;
            println!("n = {}, a = {}, eta = {}, result = {:.6e} (1 ± {:.3}), error = {:.6e}", n, a, eta, result, delta / result, error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn total_rates() {
        for i in 0..10 {
            for j in 0..14 {
                let a = 0.305 * 10.0f64.powf((i as f64) / 5.0);
                let eta = 0.003 * 10.0f64.powf((j as f64) / 5.0);
                let rate = TotalRate::new(a, eta);

                if rate.is_too_small() {
                    continue;
                }

                let target = rate.by_summation();
                let result = rate.by_lookup();
                let error = [
                    (target[0] - result[0]).abs() / target[0],
                    (target[1] - result[1]).abs() / target[1],
                ];

                println!("{:>6.3} {:>6.3} => {:.3}% {:.3}% [avg = {:.6e}]", a, eta, 100.0 * error[0], 100.0 * error[1], 0.5 * (result[0] + result[1]));
                assert!(error[0] < 0.01);
                assert!(error[1] < 0.01);
            }
        }
    }

    fn find_sum_limits(a: f64, eta: f64, max_error: f64) -> (i32, i32, i32, Complex64) {
        let (n_min, _) = TotalRate::new(a, eta).sum_limits();

        let partial_rate = |n, a, eta| {
            DoubleDiffPartialRate::new(n, a, eta).integrate()
        };

        let mut total = partial_rate(n_min, a, eta) + partial_rate(n_min + 1, a, eta) + partial_rate(n_min + 2, a, eta);
        let mut prev = partial_rate(n_min + 2, a, eta);
        let mut n = n_min + 3;
        let mut n_peak = n_min;
        let mut dn = 1;

        loop {
            let next = partial_rate(n, a, eta);
            if next.norm_sqr() > prev.norm_sqr() {
                n_peak = n;
            }
            prev = next;
            total += next * (dn as f64);

            let error = (next.re / total.re).hypot(next.im / total.im);
            if error < max_error || n - n_min > 100_000 {
                break;
            }

            dn = 1 + (n - n_min) / 10;
            n += dn;
        }

        (n_min, n_peak, n, total)
    }

    #[test]
    #[ignore]
    fn summation_limits() {
        let max_error = 1.0e-4;

        for eta in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0].iter() {
            for a in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0].iter() {
                let rate = TotalRate::new(*a, *eta);
                if rate.is_too_small() {
                    continue;
                }
                let (n_min, n_peak, n_max, _) = find_sum_limits(*a, *eta, max_error);
                let (_, expected) = rate.sum_limits();
                println!("{:.1} {:.2} {} {} {} {}", a, eta, n_min, n_peak, n_max, expected);
            }
        }
    }
}

#[cfg(test)]
mod table_generation {
    use std::fs::File;
    use std::io::Write;
    use std::time::Duration;
    use indicatif::{HumanDuration, ProgressBar, ProgressState, ProgressStyle};
    use super::*;

    fn smoothed_eta(s: &ProgressState, w: &mut dyn std::fmt::Write) {
        match (s.pos(), s.len()) {
            (0, _) => write!(w, "-").unwrap(),
            (pos, Some(len)) => write!(
                w,
                "{:#}",
                HumanDuration(Duration::from_millis(
                    (s.elapsed().as_millis() * (len as u128 - pos as u128) / (pos as u128))
                        as u64
                ))
            )
            .unwrap(),
            _ => write!(w, "-").unwrap(),
        }
    }

    #[test]
    #[ignore]
    fn create() {
        const LOW_ETA_LIMIT: f64 = 0.002;
        const LOW_A_LIMIT: f64 = 0.3;
        const A_DENSITY: usize = 40; // 20;
        const ETA_DENSITY: usize = 40; // 20;
        const N_COLS: usize = 74; // 38;
        const N_ROWS: usize = 3 * ETA_DENSITY + 1;
        let mut table = [[[0.0; 2]; N_COLS]; N_ROWS];

        println!("Generating pair-creation rate tables (CP)...");

        let style = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({smoothed_eta}): {msg}")
            .unwrap()
            .with_key("smoothed_eta", smoothed_eta);
        let pb = ProgressBar::new((N_COLS * N_ROWS) as u64).with_style(style);
        pb.enable_steady_tick(Duration::from_millis(100));

        for i in 0..N_ROWS {
            let eta = LOW_ETA_LIMIT * 10.0f64.powf((i as f64) / (ETA_DENSITY as f64));
            for j in 0..N_COLS {
                let a = LOW_A_LIMIT * 10.0f64.powf((j as f64) / (A_DENSITY as f64));
                let rate = TotalRate::new(a, eta);

                let (n_min, n_max) = rate.sum_limits();
                pb.set_message(format!("a = {:.3}, eta = {:.3e}, n = {}..{}", a, eta, n_min, n_max));

                let rate = if rate.is_too_small() {
                    [0.0, 0.0]
                } else {
                    rate.by_parallel_summation()
                };

                table[i][j] = rate;
                pb.suspend(|| println!(
                    "CP NBW: eta = {:>9.3e}, a = {:>9.3e}, i = {:>3}, j = {:>3} => {:>15.6e} {:>15.6e}",
                    eta, a, i, j, rate[0].ln(), rate[1].ln(),
                ));
                pb.inc(1);
            }
        }

        let path = "output/nbw_rate_table.rs";
        let mut file = File::create(&path).unwrap();
        writeln!(file, "use std::f64::NEG_INFINITY;").unwrap();
        writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
        writeln!(file, "pub const N_ROWS: usize = {};", N_ROWS).unwrap();
        writeln!(file, "pub const LN_MIN_A: f64 = {:.16e};", LOW_A_LIMIT.ln()).unwrap();
        writeln!(file, "pub const LN_MIN_ETA: f64 = {:.16e};", LOW_ETA_LIMIT.ln()).unwrap();
        writeln!(file, "pub const LN_A_STEP: f64 = {:.16e};", consts::LN_10 / (A_DENSITY as f64)).unwrap();
        writeln!(file, "pub const LN_ETA_STEP: f64 = {:.16e};", consts::LN_10 / (ETA_DENSITY as f64)).unwrap();
        writeln!(file, "pub const TABLE: [[[f64; 2]; {}]; {}] = [", N_COLS, N_ROWS).unwrap();
        for row in table.iter() {
            write!(file, "\t[").unwrap();
            for val in row.iter() {
                let par = val[0].ln();
                let perp = val[1].ln();
                if par.is_finite() {
                    write!(file, "[{:>18.12e}", par).unwrap();
                } else {
                    write!(file, "[{:>18}", "NEG_INFINITY").unwrap();
                }
                if perp.is_finite() {
                    write!(file, ", {:>18.12e}], ", perp).unwrap();
                } else {
                    write!(file, ", {:>18}], ", "NEG_INFINITY").unwrap();
                }
            }
            writeln!(file, "],").unwrap();
        }
        writeln!(file, "];").unwrap();
        println!("Rate data written to {}", path);
    }
}