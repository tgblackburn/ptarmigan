//! Rates and spectra for circularly polarized backgrounds
use std::f64::consts;
use rand::prelude::*;
use crate::special_functions::*;
use crate::geometry::StokesVector;
use crate::quadrature::*;

// Stokes parameter of the background field
const LASER_S3: f64 = -1.0;

// Lookup tables
mod tables;
pub mod classical;

/// Partial rate (at fixed harmonic order), differential in v = s / s_max.
/// Multiply by alpha / eta to get dP_n/(ds dphi).
fn diff_partial_rate(n: i32, a: f64, eta: f64, v: f64) -> f64 {
    if v < 0.0 || v >= 1.0 {
        return 0.0;
    }

    let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
    let smax = sn / (1.0 + sn);
    let vsmax = v * smax;
    let z = (
        4.0 * (n as f64).powi(2)
        * (a * a / (1.0 + a * a))
        * (vsmax / (sn * (1.0 - vsmax)))
        * (1.0 - vsmax / (sn * (1.0 - vsmax)))
    ).sqrt();
    let (j_nm1, j_n, j_np1) = z.j_pm(n);

    -smax * (
        j_n.powi(2)
        + 0.5 * a * a * (1.0 + 0.5 * vsmax.powi(2) / (1.0 - vsmax))
        * (2.0 * j_n.powi(2) - j_np1.powi(2) - j_nm1.powi(2))
    )
}

/// Returns the largest value of the differential partial rate, d W_n / d v,
/// multiplied by a small safety factor.
fn ceiling_diff_partial_rate(n: i32, a: f64, eta: f64) -> f64 {
    let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
    // approx harmonic index when sigma / mu < 0.25
    let n_switch = (32.3 * (1.0 + 0.476 * a.powf(1.56))) as i32;

    if sn < 2.0 || n < n_switch {
        let v_opt = (1.0 + sn) / (2.0 + sn);
        let lower = (0..16)
            .map(|i| (i as f64) * v_opt / 16.0)
            .map(|v| diff_partial_rate(n, a, eta, v))
            .reduce(f64::max)
            .unwrap();
        let upper = (0..16)
            .map(|i| v_opt + (i as f64) * (1.0 - v_opt) / 16.0)
            .map(|v| diff_partial_rate(n, a, eta, v))
            .reduce(f64::max)
            .unwrap();
        1.2 * lower.max(upper)
    } else {
        // Partition the range into 0 < u < 1 + sn, and
        // 1 + sn < u < 3(1 + sn)
        let lower: f64 = GAUSS_16_NODES.iter()
            .map(|x| 0.5 * (1.0 + sn) * (x + 1.0))
            .map(|u| diff_partial_rate(n, a, eta, u / (1.0 + u)))
            .reduce(f64::max)
            .unwrap();
        let upper: f64 = GAUSS_16_NODES.iter()
            .map(|x| (1.0 + sn) + 0.5 * 2.0 * (1.0 + sn) * (x + 1.0))
            .map(|u| diff_partial_rate(n, a, eta, u / (1.0 + u)))
            .reduce(f64::max)
            .unwrap();
        1.2 * lower.max(upper)
    }
}

/// Equivalent to `spectrum(n, a, eta, v) / eta` for eta -> 0.
fn spectrum_low_eta(n: i32, a: f64, v: f64) -> f64 {
    if v < 0.0 || v >= 1.0 {
        return 0.0;
    }

    let z = (4.0 * a * a * (n as f64).powi(2) * v * (1.0-v) / (1.0 + a * a)).sqrt();
    let (j_nm1, j_n, j_np1) = z.j_pm(n);
    (n as f64) * (a * a * j_nm1.powi(2) - 2.0 * (1.0 + a * a) * j_n.powi(2) + a * a * j_np1.powi(2)) / (1.0 + a * a)
}

/// Integrates [diff_partial_rate] over v, returning the rate to emit photons
/// at a given harmonic order.
/// Multiply by alpha / eta to get dP_n/dphi.
#[allow(unused)]
fn partial_rate(n: i32, a: f64, eta: f64) -> f64 {
    let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
    // approx harmonic index when sigma / mu < 0.25
    let n_switch = (32.3 * (1.0 + 0.476 * a.powf(1.56))) as i32;
    let integral: f64 = if sn < 2.0 || n < n_switch {
        let vmid = (1.0 + sn) / (2.0 + sn);
        let lower: f64 = GAUSS_32_NODES.iter()
            .map(|x| 0.5 * vmid * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(v, w)| 0.5 * vmid * w * diff_partial_rate(n, a, eta, v))
            .sum();
        // integrate from v = vmid to v = 1
        let upper: f64 = GAUSS_32_NODES.iter()
            .map(|x| vmid + 0.5 * (1.0 - vmid) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(v, w)| 0.5 * (1.0 - vmid) * w * diff_partial_rate(n, a, eta, v))
            .sum();
        lower + upper
    } else {
        // If peak of spectrum v >= 3/4 and peak is sufficiently narrow,
        // switch to integrating over u = v / (1 - v) instead.
        // Partition the range into 0 < u < 1 + sn, and
        // 1 + sn < u < 3(1 + sn) and integrate each separately,
        // to ensure we capture the peak.
        let lower: f64 = GAUSS_32_NODES.iter()
            .map(|x| 0.5 * (1.0 + sn) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(u, w)|
                0.5 * (1.0 + sn) * w * diff_partial_rate(n, a, eta, u / (1.0 + u)) / (1.0 + u).powi(2)
            )
            .sum();
        let upper: f64 = GAUSS_32_NODES.iter()
            .map(|x| (1.0 + sn) + 0.5 * 2.0 * (1.0 + sn) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(u, w)|
                 0.5 * 2.0 * (1.0 + sn) * w * diff_partial_rate(n, a, eta, u / (1.0 + u)) / (1.0 + u).powi(2)
            )
            .sum();
        lower + upper
    };
    integral
}

/// Equivalent to `integrated_spectrum(n, a, eta, v) / eta` for eta -> 0.
#[allow(unused)]
fn integrated_spectrum_low_eta(n: i32, a: f64) -> f64 {
    // integrate from v = 0 to v = 1/2
    let lower: f64 = GAUSS_32_NODES.iter()
        .map(|x| 0.25 * (x + 1.0))
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(v, w)| 0.25 * w * spectrum_low_eta(n, a, v))
        .sum();

    // integrate from v = 1/2 to v = 1
    let upper: f64 = GAUSS_32_NODES.iter()
        .map(|x| 0.25 * (x + 3.0))
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(v, w)| 0.25 * w * spectrum_low_eta(n, a, v))
        .sum();

    lower + upper
}

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Compton rates. Equivalent to calling
/// ```
/// let nmax = (10.0 * (1.0 + a.powi(3))) as i32;
/// let rate = (1..=nmax).map(|n| partial_rate(n, a, eta)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
pub(super) fn rate(a: f64, eta: f64) -> Option<f64> {
    if eta.ln() < tables::LN_ETA_MIN {
        // hand over to classical physics
        classical::rate(a, eta)
    } else if a.ln() < tables::LN_A_MIN {
        // linear Compton rate for arbitrary eta
        Some(a * a * (2.0 + 8.0 * eta + 9.0 * eta * eta + eta * eta * eta) / (2.0 * eta * (1.0 + 2.0 * eta).powi(2))
            - a * a * (2.0 + 2.0 * eta - eta * eta) * (1.0 + 2.0 * eta).ln() / (4.0 * eta * eta))
    } else if tables::contains(a, eta) {
        Some(tables::interpolate(a, eta))
    } else {
        eprintln!("NLC (CP) rate lookup out of bounds: a = {:.3e}, eta = {:.3e}", a, eta);
        None
    }
}

/// Returns a pseudorandomly sampled n (harmonic order), s (lightfront momentum
/// transfer) and theta (azimuthal angle in the ZMF) for a photon emission that
/// occurs at normalized amplitude a and energy parameter eta.
pub(super) fn sample<R: Rng>(a: f64, eta: f64, rng: &mut R, fixed_n: Option<i32>) -> (i32, f64, f64, StokesVector) {
    let n = fixed_n.unwrap_or_else(|| {
        tables::invert(a, eta, rng.gen())
    });

    // Approximate maximum value of the probability density:
    let max: f64 = ceiling_diff_partial_rate(n, a, eta);

    // Rejection sampling
    let v = loop {
        let v = rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = diff_partial_rate(n, a, eta, v);
        if u <= f / max {
            break v;
        }
    };

    // Four-momentum transfer s = k.l / k.q
    let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
    let s = {
        let smax = sn / (1.0 + sn);
        v * smax
    };

    // Azimuthal angle in ZMF
    let theta = 2.0 * consts::PI * rng.gen::<f64>();

    // Stokes parameters
    let z = (
        ((4 * n * n) as f64)
        * (a * a / (1.0 + a * a))
        * (s / (sn * (1.0 - s)))
        * (1.0 - s / (sn * (1.0 - s)))
    ).sqrt();

    let (j_nm1, j_n, j_np1) = z.j_pm(n);

    // Defined w.r.t. rotated basis: x || k x k' and y || k'_perp
    let sv: StokesVector = {
        let xi_0 = (1.0 - s + 1.0 / (1.0 - s)) * (j_nm1 * j_nm1 + j_np1 * j_np1 - 2.0 * j_n * j_n) - (2.0 * j_n / a).powi(2);
        let xi_1 = 2.0 * (j_nm1 * j_nm1 + j_np1 * j_np1 - 2.0 * j_n * j_n) + 8.0 * j_n * j_n * (1.0 - ((n as f64) / z).powi(2) - 0.5 / (a * a));
        let xi_2 = 0.0;
        let xi_3 = LASER_S3 * (1.0 - s + 1.0 / (1.0 - s)) * (1.0 - 2.0 * s / (sn * (1.0 - s))) * (j_nm1 * j_nm1 - j_np1 * j_np1); // +/-1 depending on wave handedness
        [1.0, xi_1 / xi_0, xi_2 / xi_0, xi_3 / xi_0].into()
    };

    // assert!(sv.dop() <= 1.0);

    // So that Stokes vector is defined w.r.t. to e_1 and e_2
    let sv = sv.rotate_by(consts::FRAC_PI_2 + theta);

    (n, s, theta, sv)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use rayon::prelude::*;
    use super::*;

    #[test]
    fn rate_ceiling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let mut err_ceiling = -1_f64;

        for _i in 0..100 {
            let a = (0.2_f64.ln() + (20_f64.ln() - 0.2_f64.ln()) * rng.gen::<f64>()).exp();
            let eta = (0.001_f64.ln() + (1.0_f64.ln() - 0.001_f64.ln()) * rng.gen::<f64>()).exp();
            let n_max = (10.0 * (1.0 + a.powi(3))) as i32;
            let harmonics: Vec<_> = if n_max > 200 {
                (0..=10).map(|i| (2_f64.ln() + 0.1 * (i as f64) * ((n_max as f64).ln() - 2_f64.ln())).exp() as i32).collect()
            } else if n_max > 10 {
                let mut low = vec![1, 2, 3];
                let mut high: Vec<_> = (0..=4).map(|i| (5_f64.ln() + 0.25 * (i as f64) * ((n_max as f64).ln() - 5_f64.ln())).exp() as i32).collect();
                low.append(&mut high);
                low
            } else {
                (1..n_max).collect()
            };

            for n in &harmonics {
                let true_max: f64 = (0..10_000)
                    .map(|i| (i as f64) / 10000.0)
                    .map(|v| diff_partial_rate(*n, a, eta, v))
                    .reduce(f64::max)
                    .unwrap();
                let max = ceiling_diff_partial_rate(*n, a, eta);
                let err = (true_max - max) / true_max;
                err_ceiling = err_ceiling.max(err);
                println!(
                    "a = {:>9.3e}, eta = {:>9.3e}, n = {:>4} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                    a, eta, n, true_max, max, 100.0 * err,
                );
                assert!(err < 0.0);
            }
        }

        println!("Largest error detected was {:.2}%", 100.0 * err_ceiling);
    }

    #[test]
    #[ignore]
    fn partial_spectrum() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let n = 100;
        let a = 1.0;
        let eta = 0.1;

        let rt = std::time::Instant::now();
        let vs: Vec<(i32,f64,f64,_)> = (0..10_000)
            .map(|_n| {
                sample(a, eta, &mut rng, Some(n))
            })
            .collect();
        let rt = rt.elapsed();

        println!("a = {:.3e}, eta = {:.3e}, {} samples takes {:?}", a, eta, vs.len(), rt);
        let filename = format!("output/nlc_cp_partial_spectrum_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        for (_, s, phi, _) in vs {
            writeln!(file, "{:.6e} {:.6e}", s, phi).unwrap();
        }
    }

    #[test]
    fn partial_rate_accuracy() {
        let max_error = 1.0e-6;

        // n, a, eta, target
        let pts = [
            (2,    0.5, 0.15,   2.748486539e-3),
            (10,   1.0, 0.2,    1.984654425e-4),
            (80,   2.0, 0.2,    3.751480198e-6),
            (160,  2.0, 0.2,    6.842944878e-9),
            (50,   3.0, 0.1,    5.090018978e-4),
            (200,  3.0, 0.1,    3.504645316e-6),
            (200,  4.0, 0.1,    5.564288841e-5),
            (500,  4.0, 0.1,    9.722534139e-7),
            (100,  5.0, 0.1,    6.745093014e-4),
            (500,  5.0, 0.1,    1.258283729e-5),
            (1000, 5.0, 0.1,    4.137051481e-7),
            (40,   7.0, 0.1,    3.198368332e-3),
            (160,  7.0, 0.1,    6.698029091e-4),
            (640,  7.0, 0.1,    5.063579159e-5),
            (2560, 7.0, 0.1,    2.322138448e-7),
            (100,  9.5, 0.1,    1.656736051e-3),
            (1000, 9.5, 0.1,    6.425026440e-5),
            (8000, 9.5, 0.1,    2.056455838e-8),
            (100,  9.5, 0.01,   1.981917068e-4),
            (1000, 9.5, 0.01,   1.624055198e-5),
            (100,  9.5, 0.0012, 2.424143218e-5),
            (1000, 9.5, 0.0012, 2.319090168e-6),
            (5000, 9.5, 0.0012, 3.366698038e-8),
        ];

        for (n, a, eta, target) in &pts {
            let result = partial_rate(*n, *a, *eta);
            let error = (target - result).abs() / target;
            println!("n = {}, a = {:.2e}, eta = {:.2e} => rate = (alpha/eta) {:.6e}, err = {:.3e}", n, a, eta, result, error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn total_rate() {
        let max_error = 1.0e-3;

        // nmax = 10 (1 + a^3), a, eta
        let pts = [
            (10,    0.5,  0.2),
            (20,    1.0,  0.2),
            (280,   3.0,  0.12),
            (650,   4.0,  0.12),
            (1260,  5.0,  0.12),
            (20,    1.0,  0.75),
            (280,   3.0,  0.75),
            (3440,  7.0,  0.1),
            (6151,  8.5,  0.6),
            (10010, 10.0, 0.0012),
            (10010, 10.0, 0.04),
            (10010, 10.0, 0.08),
            (10010, 10.0, 0.16),
            (33760, 15.0, 0.04),
            (33760, 15.0, 0.12),
            (33760, 15.0, 0.42)
        ];

        for (nmax, a, eta) in &pts {
            let rates: Vec<f64> = (1..=*nmax).map(|n| partial_rate(n, *a, *eta)).collect();
            let total: f64 = rates.iter().sum();
            let target = rate(*a, *eta).unwrap();
            let error = ((total - target) / target).abs();
            println!("a = {:.2e}, eta = {:.2e} => sum_{{n=1}}^{{{}}} rate_n = (alpha/eta) {:.6e}, err = {:.3e}", a, eta, nmax, total, error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn total_rate_low_eta() {
        let max_error = 1.0e-3;

        let pts = [
            (280, 3.0, 0.0005),
            (10010, 10.0, 0.0005),
        ];

        for (nmax, a, eta) in &pts {
            let rates: Vec<f64> = (1..=*nmax).map(|n| integrated_spectrum_low_eta(n, *a)).collect();
            let total: f64 = eta * rates.iter().sum::<f64>();
            let target = rate(*a, *eta).unwrap();
            let error = ((total - target) / target).abs();
            println!("a = {:.2e}, eta = {:.2e} => sum_{{n=1}}^{{{}}} rate_n = (alpha/eta) {:.6e}, err = {:.3e}", a, eta, nmax, total, error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn harmonic_index_sampling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let pts = [
            (0.946303, 0.105925),
            (2.37700, 0.105925),
            (4.74275, 0.105925),
            (9.46303, 0.105925),
            (14.1915, 0.105925),
            (18.9261, 0.105925),
        ];

        for (a, eta) in &pts {
            println!("At a = {}, eta = {}:", a, eta);

            let n_max = (10.0 * (1.0 + a * a * a)) as i32;

            // CDF(n)
            let rates: Vec<f64> = (1..n_max)
                .map(|n| partial_rate(n, *a, *eta))
                .scan(0.0, |cs, r| {
                    *cs = *cs + r;
                    Some(*cs)
                })
                .collect();

            let total = rates.last().unwrap();
            println!("\t ... rates computed");

            let mut avg_err = [0.0; 3];
            let mut avg_err_sqd = [0.0; 3];
            let mut std_dev = [0.0; 3];
            let mut count = [0.0; 3];

            let n_1 = 1 + rates.iter().position(|elem| *elem > 0.5 * total).unwrap();
            let n_2 = 1 + rates.iter().position(|elem| *elem > 0.75 * total).unwrap();

            for _i in 0..1_000_000 {
                let frac = rng.gen::<f64>();

                // Using the tabulated CDF
                let n_interp = tables::invert(*a, *eta, frac);

                // Doing it directly
                let target = frac * total;
                let n_direct = rates.iter()
                    .position(|elem| *elem > target)
                    .map(|n| (n + 1) as i32)
                    .unwrap();

                // let err = ((n_direct - n_interp) as f64) / (n_direct as f64);
                let err = (n_direct - n_interp) as f64;
                let i = if frac < 0.5 {0} else if frac < 0.75 {1} else {2};
                avg_err[i] += err;
                avg_err_sqd[i] += err * err;
                count[i] += 1.0;
            }

            for i in 0..3 {
                avg_err[i] = avg_err[i] / count[i];
                avg_err_sqd[i] = avg_err_sqd[i] / count[i];
                std_dev[i] = (avg_err_sqd[i] - avg_err[i].powi(2)).sqrt();
            }

            println!("\t 0.00 < f < 0.50 | {:>4} < n <= {:<5} | ⟨Δn⟩ = {:.3} ± {:.3}", 1, n_1, avg_err[0], std_dev[0]);
            println!("\t 0.50 < f < 0.75 | {:>4} < n <= {:<5} | ⟨Δn⟩ = {:.3} ± {:.3}", n_1, n_2, avg_err[1], std_dev[1]);
            println!("\t 0.75 < f < 1.00 | {:>4} < n <= {:<5} | ⟨Δn⟩ = {:.3} ± {:.3}", n_2, n_max, avg_err[2], std_dev[2]);
        }
    }

    #[test]
    #[ignore]
    fn create_rate_table() {
        use indicatif::{ProgressStyle, ParallelProgressIterator};
        use crate::pwmci;

        const LOW_A_LIMIT: f64 = 0.02;
        const LOW_ETA_LIMIT: f64 = 1.0e-3;
        const A_DENSITY: usize = 20; // points per order of magnitude
        const ETA_DENSITY: usize = 20;
        const N_COLS: usize = 61; // pts in a0 direction
        const N_ROWS: usize = 70;
        let mut table = [[0.0; N_COLS]; N_ROWS];

        let num: usize = std::env::var("RAYON_NUM_THREADS")
            .map(|s| s.parse().unwrap_or(1))
            .unwrap_or(1);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num)
            .build()
            .unwrap();

        println!("Running on {:?}", pool);

        let mut pts: Vec<(usize, usize, f64, f64, i32)> = Vec::new();
        for i in 0..N_ROWS {
            let eta = LOW_ETA_LIMIT * 10.0f64.powf((i as f64) / (ETA_DENSITY as f64));
            for j in 0..N_COLS {
                let a = LOW_A_LIMIT * 10.0f64.powf((j as f64) / (A_DENSITY as f64));
                let n_max = (10.0 * (1.0 + a.powi(3))) as i32;
                pts.push((i, j, a, eta, n_max));
            }
        }

        let style = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})").unwrap();

        let pts: Vec<(usize, usize, f64, [[f64; 2]; 32])> = pool.install(|| pts.into_iter()
            .map(|(i, j, a, eta, n_max)| {
                let rates: Vec<[f64; 2]> = (1..(n_max+1))
                    .into_par_iter()
                    .progress_with_style(style.clone())
                    .map(|n| [n as f64, partial_rate(n, a, eta)])
                    .collect();

                // Cumulative sum
                let rates: Vec<[f64; 2]> = rates
                    .into_iter()
                    .scan(0.0, |cs, [n, r]| {
                        *cs = *cs + r;
                        Some([n, *cs])
                    })
                    .collect();

                // Total rate
                let rate: f64 = rates.last().unwrap()[1];

                let mut cdf: [[f64; 2]; 32] = [[0.0, 0.0]; 32];
                cdf[0] = [rates[0][0], rates[0][1] / rate];
                if n_max <= 32 {
                    // Write all the rates
                    for i in 1..=31 {
                        cdf[i] = rates.get(i)
                            .map(|r| [r[0], r[1] / rate])
                            .unwrap_or_else(|| [(i+1) as f64, 1.0]);
                    }
                } else if n_max < 100 {
                    // first 8 harmonics
                    for i in 1..=7 {
                        cdf[i] = [rates[i][0], rates[i][1] / rate];
                    }
                    // log-spaced for n >= 9
                    let delta = ((n_max as f64).ln() - 9_f64.ln()) / (31.0 - 8.0);
                    for i in 8..=31 {
                        let n = (9_f64.ln() + ((i - 8) as f64) * delta).exp();
                        let limit = rates.last().unwrap()[0];
                        let n = n.min(limit);
                        cdf[i][0] = n;
                        cdf[i][1] = pwmci::Interpolant::new(&rates[..]).evaluate(n).unwrap() / rate;
                    }
                } else {
                    // Sample CDF at 32 log-spaced points
                    let delta = (n_max as f64).ln() / 31.0;
                    for i in 1..=31 {
                        let n = ((i as f64) * delta).exp();
                        let limit = rates.last().unwrap()[0];
                        let n = n.min(limit);
                        cdf[i][0] = n;
                        cdf[i][1] = pwmci::Interpolant::new(&rates[..]).evaluate(n).unwrap() / rate;
                    }
                }

                println!("CP NLC: a = {:.3e}, eta = {:.3e}, ln(rate) = {:.6e}", a, eta, rate.ln());
                (i, j, rate, cdf)
            })
            .collect()
        );

        for (i, j, rate, _) in &pts {
            table[*i][*j] = *rate;
        }

        let mut file = File::create("output/nlc_cp_rate_table.rs").unwrap();
        writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
        writeln!(file, "pub const N_ROWS: usize = {};", N_ROWS).unwrap();
        writeln!(file, "pub const MIN: [f64; 2] = [{:.12e}, {:.12e}];", LOW_A_LIMIT.ln(), LOW_ETA_LIMIT.ln()).unwrap();
        writeln!(file, "pub const STEP: [f64; 2] = [{:.12e}, {:.12e}];", consts::LN_10 / (A_DENSITY as f64), consts::LN_10 / (ETA_DENSITY as f64)).unwrap();
        writeln!(file, "pub const TABLE: [[f64; {}]; {}] = [", N_COLS, N_ROWS).unwrap();
        for row in table.iter() {
            let val = row.first().unwrap().ln();
            if val.is_finite() {
                write!(file, "\t[{:>18.12e}", val).unwrap();
            } else {
                write!(file, "\t[{:>18}", "NEG_INFINITY").unwrap();
            }
            for val in row.iter().skip(1) {
                let tmp = val.ln();
                if tmp.is_finite() {
                    write!(file, ", {:>18.12e}", tmp).unwrap();
                } else {
                    write!(file, ", {:>18}", "NEG_INFINITY").unwrap();
                }
            }
            writeln!(file, "],").unwrap();
        }
        writeln!(file, "];").unwrap();

        let mut file = File::create("output/nlc_cp_cdf_table.rs").unwrap();
        writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
        writeln!(file, "pub const MIN: [f64; 2] = [{:.12e}, {:.12e}];", LOW_A_LIMIT.ln(), LOW_ETA_LIMIT.ln()).unwrap();
        writeln!(file, "pub const STEP: [f64; 2] = [{:.12e}, {:.12e}];", consts::LN_10 / (A_DENSITY as f64), consts::LN_10 / (ETA_DENSITY as f64)).unwrap();
        writeln!(file, "pub const TABLE: [[[f64; 2]; 32]; {}] = [", N_ROWS * N_COLS).unwrap();
        for (_, _, _, cdf) in &pts {
            write!(file, "\t[").unwrap();
            for entry in cdf.iter().take(31) {
                write!(file, "[{:>18.12e}, {:>18.12e}], ", entry[0], entry[1]).unwrap();
            }
            writeln!(file, "[{:>18.12e}, {:>18.12e}]],", cdf[31][0], cdf[31][1]).unwrap();
        }
        writeln!(file, "];").unwrap();
    }
}
