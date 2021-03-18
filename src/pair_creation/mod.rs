//! Nonlinear Breit-Wheeler pair creation

use std::f64::consts;
use rand::prelude::*;
use crate::constants::*;
use crate::geometry::*;
use crate::special_functions::*;

/// Evaluates the important part of the nonlinear Breit-Wheeler
/// differential rate, f,
/// either
///   `dP/(ds dϕ) = ⍺ f(n, a, η, s) / η`
/// or
///   `dP/(ds dt) = ⍺ m f(n, a, η, s) / γ`
/// where `0 < s < 1`.
///
/// The spectrum is symmetric about s = 1/2.
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
fn partial_rate(n: i32, a: f64, eta: f64) -> f64 {
    let ell = n as f64;
    let sn = 2.0 * ell * eta / (1.0 + a * a);

    // equivalent to n > 2 (1 + a^2) / eta
    assert!(sn > 4.0);

    // approx position at which probability is maximised
    let beta = a.powi(4) * (1.0/ell + 1.0).powi(2) + 16.0 * (a * a - 2.0).powi(2) / (sn * sn) - 8.0 * a * a * (a * a - 2.0) / sn;
    let beta = beta.sqrt() / (a * a - 2.0);
    let alpha = (a * a + 2.0 * ell) / (ell * (2.0 - a * a)) - 4.0 / sn;
    let tmp = alpha + beta;
    let s_peak = 0.5 * (1.0 - tmp.sqrt());
    //eprintln!("alpha = {}, beta = {}, tmp = {}, s_peak = {:.6}", alpha, beta, tmp, s_peak);

    let s_min = 0.5 - (0.25 - 1.0 / sn).sqrt();
    let s_max = 0.5;

    if s_peak.is_finite() {
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
    }
}

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Breit-Wheeler rates, implemented as a table lookup.
fn rate_by_lookup(a: f64, eta: f64) -> f64 {
    unimplemented!()
}

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Breit-Wheeler rates.
fn rate_by_summation(a: f64, eta: f64) -> f64 {
    let n_min = (2.0 * (1.0 + a * a) / eta).ceil() as i32;
    let delta = (1.671 * (1.0 + 1.226 * a * a) * (1.0 + 7.266 * eta) / eta).ceil() as i32;
    let n_max = n_min + delta;
    (n_min..n_max).map(|n| partial_rate(n, a, eta)).sum()
}

#[allow(dead_code)]
fn find_sum_limits(a: f64, eta: f64, max_error: f64) -> (i32, i32, i32, f64) {
    let n_min = (2.0f64 * (1.0 + a * a) / eta).ceil() as i32;

    let mut total = 0.0;
    let mut n_peak = n_min;
    let mut partial = partial_rate(n_min, a, eta);
    let mut n = n_min + 1;
    loop {
        total += partial;
        let tmp = partial_rate(n, a, eta);
        if tmp > partial {
            n_peak = n;
        }
        partial = tmp;
        n += 1;

        if n < 2 * n_min {
            continue;
        } else if partial / total < max_error {
            break;
        } else if total == 0.0 {
            break;
        }
    }

    (n_min, n_peak, n, total)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::*;

    #[test]
    fn partial_rates() {
        let max_error = 1.0e-6;

        // At chi = a eta = 0.1:

        let (n, a, eta) = (50, 1.0, 0.1);
        let target = 1.4338498960396018931e-18;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (100, 1.0, 0.1);
        let target = 1.3654528056555865291e-18;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (200, 1.0, 0.1);
        let target = 2.0884399327604375975e-34;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (4000, 5.0, 0.02);
        let target = 9.2620552570985535880e-61;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (5000, 5.0, 0.02);
        let target = 5.9413657401296979089e-17;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (6000, 5.0, 0.02);
        let target = 1.1629168979497463847e-18;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (8000, 5.0, 0.02);
        let target = 1.6722921069034930599e-23;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (3, 0.1, 1.0);
        let target = 2.7926363804797338348e-7;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (5, 0.1, 1.0);
        let target = 1.4215822192894496185e-10;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (20, 0.1, 1.0);
        let target = 1.1028787567587238051e-37;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        // At chi = a eta = 1

        let (n, a, eta) = (3000, 10.0, 0.1);
        let target = 2.8532353822421244101e-48;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (4000, 10.0, 0.1);
        let target = 1.4356925580571873594e-5;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (8000, 10.0, 0.1);
        let target = 1.5847977567888444504e-7;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (6, 1.0, 1.0);
        let target = 0.0031666996194000280745;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (20, 1.0, 1.0);
        let target = 1.2280171339973043031e-5;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (50, 1.0, 1.0);
        let target = 7.7268893728561315057e-11;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        // At chi = a eta = 10

        let (n, a, eta) = (300, 10.0, 1.0);
        let target = 1.6230747656967905300e-7;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (400, 10.0, 1.0);
        let target = 0.0031791285538880908649;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (2000, 10.0, 1.0);
        let target = 5.9533991784012215406e-5;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        // At chi = a eta = 0.01

        let (n, a, eta) = (640, 1.0, 0.01);
        let target = 9.7351336009776642509e-115;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (1000, 1.0, 0.01);
        let target = 2.3257373741363993054e-156;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (25, 0.1, 0.1);
        let target = 5.7778053795802739886e-52;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);

        let (n, a, eta) = (50, 0.1, 0.1);
        let target = 3.3444371706672986244e-90;
        let result = partial_rate(n, a, eta);
        let error = (target - result).abs() / target;
        eprintln!("n = {}, a = {}, eta = {}, result = {:.6e}, error = {:.6e}", n, a, eta, result, error);
        assert!(error < max_error);
    }

    #[test]
    #[ignore]
    fn summation_limits() {
        let max_error = 1.0e-4;
        let pts = [
            (0.1, 1.0), (1.0, 0.1), (10.0, 0.01), (1.0, 1.0), (10.0, 0.1),
            (10.0, 1.0), (0.1, 0.1), (1.0, 0.01), (0.2, 0.1), (0.5, 0.1),
            (2.0, 0.1), (5.0, 0.1), (0.2, 1.0), (0.5, 1.0), (2.0, 1.0),
            (5.0, 1.0), (0.5, 0.01), (2.0, 0.01), (5.0, 0.01)
        ];

        for (a, eta) in pts.iter() {
            let (n_min, n_peak, n_max, total) = find_sum_limits(*a, *eta, max_error);
            println!("{:.6e} {:.6e} {} {} {} {:.6e}", a, eta, n_min, n_peak, n_max, total);
        }
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
