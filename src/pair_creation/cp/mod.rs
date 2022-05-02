//! Nonlinear Breit-Wheeler pair creation in CP backgrounds

use std::f64::consts;
use rand::prelude::*;
use crate::special_functions::*;

#[cfg(not(feature = "leading-order-only"))]
mod rate_table;

#[cfg(feature = "leading-order-only")]
mod leading_order;

#[cfg(feature = "leading-order-only")]
pub use leading_order::{rate, sample};

/// Evaluates the important part of the nonlinear Breit-Wheeler
/// differential rate, f,
/// either
///   `dP/(ds dϕ) = ⍺ f(n, a, η, s) / η`
/// or
///   `dP/(ds dt) = ⍺ m f(n, a, η, s) / γ`
/// where `0 < s < 1`.
///
/// The spectrum is symmetric about s = 1/2.
#[cfg(not(feature = "leading-order-only"))]
fn partial_spectrum(n: i32, a: f64, eta: f64, v: f64) -> f64 {
    let ell = n as f64;
    let sn = 2.0 * ell * eta / (1.0 + a * a);

    // equivalent to n > 2 (1 + a^2) / eta
    //if sn <= 4.0 {
    //    return 0.0;
    //}

    // limits on v come from requirement that z > 0
    //let (v_min, v_max) = (
    //    0.5 - (0.25 - 1.0 / sn).sqrt(),
    //    0.5 + (0.25 + 1.0 / sn).sqrt()
    //);

    //if v < v_min || v > v_max {
    //    return 0.0;
    //}

    let z = {
        let tmp = 1.0 / (sn * v * (1.0 - v));
        ((4.0 * ell * ell) * (a * a / (1.0 + a * a)) * tmp * (1.0 - tmp)).sqrt()
    };

    let (j_nm1, j_n, j_np1) = z.j_pm(n);

    j_n.powi(2)
    - 0.5 * a * a * (1.0 / (2.0 * v * (1.0 - v)) - 1.0)
    * (2.0 * j_n.powi(2) - j_np1.powi(2) - j_nm1.powi(2))
}

/// Integrates the important part of the nonlinear Breit-Wheeler
/// differential rate to give
///   `dP/dϕ = ⍺ F(n, a, η) / η`
/// or
///   `dP/dt = ⍺ m F(n, a, η) / γ`
/// where F = \int_0^1 f ds.
#[cfg(not(feature = "leading-order-only"))]
fn partial_rate(n: i32, a: f64, eta: f64) -> f64 {
    let ell = n as f64;
    let sn = 2.0 * ell * eta / (1.0 + a * a);

    // equivalent to n > 2 (1 + a^2) / eta
    if sn <= 4.0 {
        return 0.0;
    }

    // approx position at which probability is maximised
    let beta = a.powi(4) * (1.0/ell + 1.0).powi(2) + 16.0 * (a * a - 2.0).powi(2) / (sn * sn) - 8.0 * a * a * (a * a - 2.0) / sn;
    let beta = beta.sqrt() / (a * a - 2.0);
    let alpha = (a * a + 2.0 * ell) / (ell * (2.0 - a * a)) - 4.0 / sn;
    let tmp = alpha + beta;
    let s_peak = 0.5 * (1.0 - tmp.sqrt());

    let s_min = 0.5 - (0.25 - 1.0 / sn).sqrt();
    let s_max = 0.5;
    //println!("alpha = {}, beta = {}, tmp = {}, s_peak = {:.6}, s_min = {:.6e}", alpha, beta, tmp, s_peak, s_min);

    let pr = if s_peak.is_finite() {
        let s_mid = 2.0 * s_peak - s_min;
        if s_mid > s_max {
            // do integral in two stages, from s_min to s_peak and then
            // s_peak to s_max
            let lower: f64 = GAUSS_32_NODES.iter()
                .map(|x| 0.5 * (s_peak - s_min) * x + 0.5 * (s_min + s_peak))
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(s, w)| {
                    let sp = partial_spectrum(n, a, eta, s);
                    //println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                    0.5 * (s_peak - s_min) * w * sp
                })
                .sum();

            let upper: f64 = GAUSS_32_NODES.iter()
                .map(|x| 0.5 * (s_max - s_peak) * x + 0.5 * (s_peak + s_max))
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(s, w)| {
                    let sp = partial_spectrum(n, a, eta, s);
                    //println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                    0.5 * (s_max - s_peak) * w * sp
                })
                .sum();

            2.0 * (upper + lower)
        } else {
            // do integral in three stages, from s_min to s_peak,
            // s_peak to s_mid, and s_mid to s_max
            let lower: f64 = GAUSS_32_NODES.iter()
                .map(|x| 0.5 * (s_peak - s_min) * x + 0.5 * (s_min + s_peak))
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(s, w)| {
                    let sp = partial_spectrum(n, a, eta, s);
                    //println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                    0.5 * (s_peak - s_min) * w * sp
                })
                .sum();

            let middle: f64 = GAUSS_32_NODES.iter()
                .map(|x| 0.5 * (s_mid - s_peak) * x + 0.5 * (s_peak + s_mid))
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(s, w)| {
                    let sp = partial_spectrum(n, a, eta, s);
                    //println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                    0.5 * (s_mid - s_peak) * w * sp
                })
                .sum();

            let upper: f64 = GAUSS_32_NODES.iter()
                .map(|x| 0.5 * (s_max - s_mid) * x + 0.5 * (s_mid + s_max))
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(s, w)| {
                    let sp = partial_spectrum(n, a, eta, s);
                    //println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                    0.5 * (s_max - s_mid) * w * sp
                })
                .sum();

            2.0 * (lower + middle + upper)
        }
    } else {
        let total: f64 = GAUSS_32_NODES.iter()
            .map(|x| 0.5 * (s_max - s_min) * x + 0.5 * (s_min + s_max))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(s, w)| {
                let sp = partial_spectrum(n, a, eta, s);
                //println!("{} {:.3e} {:.3e} {:.6e} {:.6e}", n, a, eta, s, sp);
                0.5 * (s_max - s_min) * w * sp
            })
            .sum();
        2.0 * total
    };

    pr
}

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Breit-Wheeler rates, implemented as a table lookup.
#[allow(unused_parens)]
#[cfg(not(feature = "leading-order-only"))]
fn rate_by_lookup(a: f64, eta: f64) -> f64 {
    let (x, y) = (a.ln(), eta.ln());

    if x < rate_table::MIN[0] {
        panic!("NLBW rate lookup out of bounds: a = {:.3e}, eta = {:.3e}", a, eta);
    } else if y < rate_table::MIN[1] {
        0.0
    } else {
        let ix = ((x - rate_table::MIN[0]) / rate_table::STEP[0]) as usize;
        let iy = ((y - rate_table::MIN[1]) / rate_table::STEP[1]) as usize;
        if ix < rate_table::N_COLS - 1 && iy < rate_table::N_ROWS - 1 {
            if a * eta > 2.0 {
                // linear interpolation of: log y against log x, best for power law
                let dx = (x - rate_table::MIN[0]) / rate_table::STEP[0] - (ix as f64);
                let dy = (y - rate_table::MIN[1]) / rate_table::STEP[1] - (iy as f64);
                let f = (
                    (1.0 - dx) * (1.0 - dy) * rate_table::TABLE[iy][ix]
                    + dx * (1.0 - dy) * rate_table::TABLE[iy][ix+1]
                    + (1.0 - dx) * dy * rate_table::TABLE[iy+1][ix]
                    + dx * dy * rate_table::TABLE[iy+1][ix+1]
                );
                f.exp()
            } else {
                // linear interpolation of: 1 / log y against x, best for exp(-1/x)?
                let a_min = (rate_table::MIN[0] + (ix as f64) * rate_table::STEP[0]).exp();
                let a_max = (rate_table::MIN[0] + ((ix+1) as f64) * rate_table::STEP[0]).exp();
                let eta_min = (rate_table::MIN[1] + (iy as f64) * rate_table::STEP[1]).exp();
                let eta_max = (rate_table::MIN[1] + ((iy+1) as f64) * rate_table::STEP[1]).exp();
                let dx = (a - a_min) / (a_max - a_min);
                let dy = (eta - eta_min) / (eta_max - eta_min);
                let f = (
                    (1.0 - dx) * (1.0 - dy) / rate_table::TABLE[iy][ix]
                    + dx * (1.0 - dy) / rate_table::TABLE[iy][ix+1]
                    + (1.0 - dx) * dy / rate_table::TABLE[iy+1][ix]
                    + dx * dy / rate_table::TABLE[iy+1][ix+1]
                );
                (1.0 / f).exp()
            }
        } else {
            panic!("NLBW rate lookup out of bounds: a = {:.3e}, eta = {:.3e}", a, eta);
        }
    }
}

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Breit-Wheeler rates.
#[cfg(not(feature = "leading-order-only"))]
fn rate_by_summation(a: f64, eta: f64) -> f64 {
    let (n_min, n_max) = sum_limits(a, eta);
    (n_min..n_max).map(|n| partial_rate(n, a, eta)).sum()
}

/// Checks if a and eta are small enough such that the rate < exp(-200)
#[cfg(not(feature = "leading-order-only"))]
fn rate_too_small(a: f64, eta: f64) -> bool {
    eta.log10() < -1.0 - (a.log10() + 2.0).powi(2) / 4.5
}

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Breit-Wheeler rates. Equivalent to calling
/// ```
/// let rate = (n_min..=n_max).map(|n| partial_rate(n, a, eta)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
#[cfg(not(feature = "leading-order-only"))]
pub(super) fn rate(a: f64, eta: f64) -> Option<f64> {
    let f = if a < 0.02 || rate_too_small(a, eta) {
        0.0
    } else if a < rate_table::MIN[0].exp() {
        rate_by_summation(a, eta)
    } else {
        rate_by_lookup(a, eta)
    };

    Some(f)
}

/// Returns a pseudorandomly sampled n (harmonic order), s (lightfront momentum
/// transfer) and theta (azimuthal angle in the ZMF) for a pair creatione event that
/// occurs at normalized amplitude a and energy parameter eta.
#[cfg(not(feature = "leading-order-only"))]
pub(super) fn sample<R: Rng>(a: f64, eta: f64, rng: &mut R) -> (i32, f64, f64) {
    let n = {
        let (n_min, n_max) = sum_limits(a, eta);
        let target = if a < rate_table::MIN[0].exp() {
            rate_by_summation(a, eta)
        } else {
            rate_by_lookup(a, eta)
        };
        let target = target * rng.gen::<f64>();
        let mut cumsum: f64 = 0.0;
        let mut index: i32 = -1; // invalid harmonic order
        for k in n_min..n_max {
            cumsum += partial_rate(k, a, eta);
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
        assert!(index >= n_min && index < n_max);
        index
    };

    let j = n as f64;
    let sn = 2.0 * j * eta / (1.0 + a * a);
    let s_min = 0.5 - (0.25 - 1.0 / sn).sqrt();
    let s_max = 0.5;

    // Approximate maximum value of the probability density:
    let max: f64 = GAUSS_32_NODES.iter()
        .map(|x| 0.5 * (s_max - s_min) * x + 0.5 * (s_min + s_max)) // from x in [-1,1] to s in [s_min, smax]
        .map(|s| partial_spectrum(n, a, eta, s))
        .fold(0.0f64 / 0.0f64, |a: f64, b: f64| a.max(b));
    let max = 1.5 * max;

    // Rejection sampling for s = k.p / k.ell
    let s = loop {
        let s = s_min + (1.0 - 2.0 * s_min) * rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = partial_spectrum(n, a, eta, s);
        if u <= f / max {
            break s;
        }
    };

    let cphi_zmf = 2.0 * consts::PI * rng.gen::<f64>();

    (n, s, cphi_zmf)
}

#[cfg(not(feature = "leading-order-only"))]
fn sum_limits(a: f64, eta: f64) -> (i32, i32) {
    let n_min = (2.0 * (1.0 + a * a) / eta).ceil() as i32;
    //let delta = (1.671 * (1.0 + 1.226 * a * a) * (1.0 + 7.266 * eta) / eta).ceil() as i32;
    let delta = (0.25 * (1.0 + 3.3 * a.sqrt() + 8.0 * a * a) * (1.0 + 7.3 * eta) / eta).ceil() as i32;
    let n_max = n_min + delta;
    (n_min, n_max)
}

#[cfg(not(feature = "leading-order-only"))]
#[allow(dead_code)]
fn find_sum_limits(a: f64, eta: f64, max_error: f64) -> (i32, i32, i32, f64) {
    let n_min = (2.0f64 * (1.0 + a * a) / eta).ceil() as i32;

    let mut total = partial_rate(n_min, a, eta) + partial_rate(n_min + 1, a, eta) + partial_rate(n_min + 2, a, eta);
    let mut prev = partial_rate(n_min + 2, a, eta);
    let mut n = n_min + 3;
    let mut n_peak = n_min;

    loop {
        let next = partial_rate(n, a, eta);
        if next > prev {
            n_peak = n;
        }
        prev = next;
        total += next;

        if next / total < max_error {
            break;
        }

        n += 1;
    }

    (n_min, n_peak, n, total)
}

#[cfg(not(feature = "leading-order-only"))]
#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
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
            let result = partial_rate(*n, *a, *eta);
            let error = (target - result).abs() / target;
            println!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn extra_check() {
        let eta = 0.2;
        let a = [0.5, 1.0, 2.5];
        for a in &a {
            let target = rate_by_summation(*a, eta);
            let result = rate_by_lookup(*a, eta);
            let error = (target - result).abs() / target;
            println!("a = {}, eta = {}, result = {:.6e}, target = {:.6e}, error = {:.6e}", a, eta, result, target, error);
        }
    }

    #[test]
    fn total_rates() {
        let max_error = 0.01;

        for i in 0..9 {
            for j in 0..14 {
                let a = 0.305 * 10.0f64.powf((i as f64) / 5.0);
                let eta = 0.003 * 10.0f64.powf((j as f64) / 5.0);
                if rate_too_small(0.9 * a, 0.9 * eta) {
                    continue;
                }
                let target = rate_by_summation(a, eta);
                let result = rate_by_lookup(a, eta);
                let error = (target - result).abs() / target;
                //println!("a = {}, eta = {}, result = {:.6e}, target = {:.6e}, error = {:.6e}", a, eta, result, target, error);
                println!("{:.6e} {:.6e} {:.6e} {:.6e} {:.6e}", a, eta, error, target, result);
                assert!(error < max_error);
            }
        }
    }

    #[test]
    #[ignore]
    fn summation_limits() {
        let max_error = 1.0e-4;
        let pts = [
            (0.1, 1.0), (1.0, 0.1), (10.0, 0.01), (1.0, 1.0), (10.0, 0.1),
            (10.0, 1.0), (0.1, 0.1), (1.0, 0.01), (0.2, 0.1), (0.5, 0.1),
            (2.0, 0.1), (5.0, 0.1), (0.2, 1.0), (0.5, 1.0), (2.0, 1.0),
            (5.0, 1.0), (0.5, 0.01), (2.0, 0.01), (5.0, 0.01), (0.3, 0.01),
            (0.3, 0.1), (0.3, 1.0), (0.05, 0.1), (0.05, 1.0), (0.25, 0.01),
        ];

        for (a, eta) in pts.iter() {
            let (n_min, n_peak, n_max, total) = find_sum_limits(*a, *eta, max_error);
            let (_, expected) = sum_limits(*a, *eta);
            println!("{:.6e} {:.6e} {} {} {} {:.6e} {}", a, eta, n_min, n_peak, n_max, total, expected);
        }
    }

    #[test]
    #[ignore]
    fn create_rate_table() {
        const LOW_ETA_LIMIT: f64 = 0.002;
        const LOW_A_LIMIT: f64 = 0.3;
        const A_DENSITY: usize = 40;
        const ETA_DENSITY: usize = 40;
        const N_COLS: usize = 70; // 2 * DENSITY;
        const N_ROWS: usize = 3 * ETA_DENSITY + 5;
        let mut table = [[0.0; N_COLS]; N_ROWS];

        for i in 0..N_ROWS {
            let eta = LOW_ETA_LIMIT * 10.0f64.powf((i as f64) / (ETA_DENSITY as f64));
            for j in 0..N_COLS {
                let a = LOW_A_LIMIT * 10.0f64.powf((j as f64) / (A_DENSITY as f64));
                let rate = if rate_too_small(a, eta) {
                    0.0
                } else {
                    rate_by_summation(a, eta)
                };
                table[i][j] = rate;
                println!("NBW: eta = {:.3e}, a = {:.3e}, ln(rate) = {:.6e}", eta, a, table[i][j].ln());
            }
        }

        let mut file = File::create("output/rate_table.rs").unwrap();
        writeln!(file, "use std::f64::NEG_INFINITY;").unwrap();
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
    }
}

static GAUSS_32_NODES: [f64; 32] = [
    -9.972638618494816e-1,
    -9.856115115452683e-1,
    -9.647622555875064e-1,
    -9.349060759377397e-1,
    -8.963211557660521e-1,
    -8.493676137325700e-1,
    -7.944837959679424e-1,
    -7.321821187402897e-1,
    -6.630442669302152e-1,
    -5.877157572407623e-1,
    -5.068999089322294e-1,
    -4.213512761306353e-1,
    -3.318686022821276e-1,
    -2.392873622521371e-1,
    -1.444719615827965e-1,
    -4.830766568773832e-2,
    4.830766568773832e-2,
    1.444719615827965e-1,
    2.392873622521371e-1,
    3.318686022821276e-1,
    4.213512761306353e-1,
    5.068999089322294e-1,
    5.877157572407623e-1,
    6.630442669302152e-1,
    7.321821187402897e-1,
    7.944837959679424e-1,
    8.493676137325700e-1,
    8.963211557660521e-1,
    9.349060759377397e-1,
    9.647622555875064e-1,
    9.856115115452683e-1,
    9.972638618494816e-1,
];

static GAUSS_32_WEIGHTS: [f64; 32] = [
    7.018610000000000e-3,
    1.627439500000000e-2,
    2.539206500000000e-2,
    3.427386300000000e-2,
    4.283589800000000e-2,
    5.099805900000000e-2,
    5.868409350000000e-2,
    6.582222280000000e-2,
    7.234579411000000e-2,
    7.819389578700000e-2,
    8.331192422690000e-2,
    8.765209300440000e-2,
    9.117387869576400e-2,
    9.384439908080460e-2,
    9.563872007927486e-2,
    9.654008851472780e-2,
    9.654008851472780e-2,
    9.563872007927486e-2,
    9.384439908080460e-2,
    9.117387869576400e-2,
    8.765209300440000e-2,
    8.331192422690000e-2,
    7.819389578700000e-2,
    7.234579411000000e-2,
    6.582222280000000e-2,
    5.868409350000000e-2,
    5.099805900000000e-2,
    4.283589800000000e-2,
    3.427386300000000e-2,
    2.539206500000000e-2,
    1.627439500000000e-2,
    7.018610000000000e-3,
];
