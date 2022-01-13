//! Rates and spectra for circularly polarized backgrounds
use std::f64::consts;
use crate::special_functions::*;
use super::{GAUSS_32_NODES, GAUSS_32_WEIGHTS};

/// Rate, differential in s (fractional lightfront momentum transfer)
/// and theta (polar angle).
/// Result valid only for 0 < s < s_max and 0 < theta < pi/2.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
fn double_diff_partial_rate(a: f64, eta: f64, s: f64, theta: f64, dj: &mut DoubleBessel) -> f64 {
    let n = dj.n();

    // opposite sign! cos theta > 0 in 0 < theta < pi/2
    let alpha = {
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let wn = s / (sn * (1.0 - s));
        2.0 * (n as f64) * a * theta.cos() * (wn * (1.0 - wn)).sqrt() / (1.0 + 0.5 * a * a).sqrt()
    };

    let beta = a * a * s / (8.0 * eta * (1.0 - s));

    // assert!(alpha < (n as f64) * consts::SQRT_2);
    // assert!(beta < (n as f64) * 0.5);

    // need to correct for alpha being negative, using
    // J_n(-|alpha|, beta) = (-1)^n J_n(|alpha|, beta)
    let j = dj.evaluate(alpha.abs(), beta); // n-2, n-1, n, n+1, n+2

    let gamma = if n % 2 == 0 {
        // j[1] and j[3] change sign
        [j[2], -0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])]
    } else {
        // j[0], j[2] and j[4] change sign
        [-j[2], 0.5 * (j[1] + j[3]), -0.25 * (j[0] + 2.0 * j[2] + j[4])]
    };

    (-gamma[0] * gamma[0] - a * a * (1.0 + 0.5 * s * s / (1.0 - s)) * (gamma[0] * gamma[2] - gamma[1] * gamma[1])) / (2.0 * consts::PI)
}

/// `double_diff_partial_rate` integrated over s and theta.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
fn partial_rate(n: i32, a: f64, eta: f64) -> f64 {
    let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
    let smax = sn / (1.0 + sn);
    // approximate s where rate is maximised
    let s_peak = sn / (2.0 + sn);

    // allocate once and reuse
    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);

    let integral: f64 = GAUSS_32_NODES.iter()
        .map(|x| 0.5 * (x + 1.0) * smax)
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(s, ws)| {
            let theta_integral: f64 = GAUSS_32_NODES.iter()
                // integrate over 0 to pi/2, then multiply by 4
                .map(|x| 0.5 * (x + 1.0) * consts::FRAC_PI_2)
                .zip(GAUSS_32_WEIGHTS.iter())
                .map(|(theta, wt)|
                     4.0 * (0.5 * consts::FRAC_PI_2) * wt * double_diff_partial_rate(a, eta, s, theta, &mut dj)
                )
                .sum();
            ws * (0.5 * smax) * theta_integral
        })
        .sum();

    integral
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use super::*;

    #[test]
    fn integration_domain() {
        let (n, a, eta) = (20, 1.0, 0.1);
        // bounds on s
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let smax = sn / (1.0 + sn);

        let nodes: Vec<f64> = (0..100).map(|i| -1.0 + 2.0 * (i as f64) / 100.0).collect();
        let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);

        let filename = format!("output/nlc_lp_dd_rate_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        for s in nodes.iter().map(|x| 0.5 * (x + 1.0) * smax) {
            for theta in nodes.iter().map(|x| 0.25 * (x + 1.0) * consts::PI) {
                let rate = double_diff_partial_rate(a, eta, s, theta, &mut dj);
                writeln!(file, "{:.6e} {:.6e} {:.6e}", s, theta, rate).unwrap();
            }
        }
    }

    #[test]
    fn partial_rates() {
        let (a, eta): (f64, f64) = (10.0, 0.1);
        let nmax = (10.0 * (1.0 + a * a)).ceil() as i32;
        let filename = format!("output/nlc_lp_rates_{}_{}.dat", a, eta);
        let mut file = File::create(&filename).unwrap();
        for n in 1..=nmax {
            let rate = partial_rate(n, a, eta);
            println!("n = {}, rate = {:.6e}", n, rate);
            writeln!(file, "{:.6e}", rate).unwrap();
        }
    }

    #[test]
    fn harmonic_limit() {
        let a_s: [f64; 9] = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 7.0, 10.0];
        let eta = 0.001;
        for &a in &a_s {
            let mut sum = 0.0;
            let mut n = 1;
            let nstop = loop {
                let rate = partial_rate(n, a, eta);
                sum += rate;
                if rate / sum < 1.0e-4 {
                    break n;
                }
                n += 1;
            };
            println!("a = {:.3e}, stopped at n = {}", a, nstop);
        }
    }

}