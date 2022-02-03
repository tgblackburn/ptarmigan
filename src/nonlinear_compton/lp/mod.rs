//! Rates and spectra for circularly polarized backgrounds
use std::f64::consts;
use num_complex::Complex;
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
        let wn = wn.min(1.0);
        2.0 * (n as f64) * a * theta.cos() * (wn * (1.0 - wn)).sqrt() / (1.0 + 0.5 * a * a).sqrt()
    };

    let beta = a * a * s / (8.0 * eta * (1.0 - s));

    // assert!(alpha < (n as f64) * consts::SQRT_2);
    // assert!(beta < (n as f64) * 0.5);

    // need to correct for alpha being negative, using
    // J_n(-|alpha|, beta) = (-1)^n J_n(|alpha|, beta)
    let j = dj.around(alpha, beta); // n-2, n-1, n, n+1, n+2

    let gamma = if n % 2 == 0 {
        // j[1] and j[3] change sign
        [j[2], -0.5 * (j[1] + j[3]), 0.25 * (j[0] + 2.0 * j[2] + j[4])]
    } else {
        // j[0], j[2] and j[4] change sign
        [-j[2], 0.5 * (j[1] + j[3]), -0.25 * (j[0] + 2.0 * j[2] + j[4])]
    };

    (-gamma[0] * gamma[0] - a * a * (1.0 + 0.5 * s * s / (1.0 - s)) * (gamma[0] * gamma[2] - gamma[1] * gamma[1])) / (2.0 * consts::PI)
}

#[derive(Debug)]
struct ThetaBound {
    s: [f64; 16],
    f: [f64; 16],
}

impl ThetaBound {
    fn for_harmonic(n: i32, a: f64, eta: f64) -> Self {
        let mut s = [0.0; 16];
        let mut f = [0.0; 16];

        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);

        for i in 0..16 {
            let z = (consts::PI * (i as f64) / 15.0).cos();
            s[i] = 0.5 * (z + 1.0) * sn / (1.0 + sn);

            // Coordinates in (x,y) space where integration over theta begins
            let x = {
                let wn = s[i] / (sn * (1.0 - s[i]));
                let wn = wn.min(1.0);
                2.0 * (n as f64) * a * (wn * (1.0 - wn)).sqrt() / (1.0 + 0.5 * a * a).sqrt()
            };

            let y = a * a * s[i] / (8.0 * eta * (1.0 - s[i]));

            // At fixed y, J(n,x,y) is maximised at
            let x_crit = if y > (n as f64) / 6.0 {
                4.0 * y.sqrt() * ((n as f64) - 2.0 * y).sqrt()
            } else {
                (n as f64) + 2.0 * y
            };

            // with value approx J_n(n) ~ 0.443/n^(1/3), here log-scaled
            let ln_j_crit = 0.443f64.ln() - (n as f64).ln() / 3.0;

            // The value of J(n,0,y) is approximately
            let ln_j_bdy = if n % 2 == 0 {
                Self::ln_double_bessel_x_zero(n, y)
            } else {
                Self::ln_double_bessel_x_zero(n + 1, y)
            };
            let ln_j_bdy = ln_j_bdy.min(ln_j_crit);

            // Exponential fit between x = 0 and x = x_crit, looking for the
            // x which is 100x smaller than at the starting point
            let cos_theta = 1.0 - x_crit * 0.01f64.ln() / (x * (ln_j_bdy - ln_j_crit));
            let cos_theta = cos_theta.max(-1.0);

            f[i] = cos_theta.acos();
            if f[i].is_nan() {
                f[i] = consts::PI;
            }

            //println!("(x, y) = ({:.3e}, {:.3e}), (x_crit, j_crit) = ({:.3e}, {:.3e}), j_bdy = {:.3e}, cos_theta = {:.3e}, theta = {:.3e}", x, y, x_crit, ln_j_crit.exp(), ln_j_bdy.exp(), cos_theta, f[i]);
        }

        let mut f_min = f[0];
        for i in 1..16 {
            if f[i] < f_min {
                f_min = f[i];
            } else {
                f[i] = f_min;
            }
        }

        Self {s, f}
    }

    /// Approximate value for the double Bessel function J(n,0,y)
    /// along x = 0, using a saddle point approximation for small y
    /// and a Taylor series for y near n/2.
    fn ln_double_bessel_x_zero(n: i32, y: f64) -> f64 {
        let n = n as f64;
        let z = (n / (4.0 * y) - 0.5).sqrt();
        let theta = Complex::new(
            consts::FRAC_PI_2,
            ((1.0 + z * z).sqrt() - z).ln()
        );
        let f = Complex::<f64>::i() * (-n * theta - y * (2.0 * theta).sin());
        let f2 = Complex::<f64>::i() * 4.0 * y * (2.0 * theta).sin();
        let e = 0.5 * (consts::PI - f2.arg());
        let phase = f + Complex::i() * e;
        (2.0 / (consts::PI * f2.norm())).sqrt().ln() + phase.re + Complex::new(phase.im, 0.0).cos().ln().re
    }

    fn at(&self, s: f64) -> f64 {
        let mut val = self.f[15];

        for i in 1..16 {
            // s[i] is stored backwards, decreasing from s_max
            if s > self.s[i] {
                let weight = (s - self.s[i-1]) / (self.s[i] - self.s[i-1]);
                val = weight * self.f[i] + (1.0 - weight) * self.f[i-1];
                break;
            }
        }

        val.min(consts::FRAC_PI_2)
    }
}

/// `double_diff_partial_rate` integrated over 0 < theta < 2 pi
fn single_diff_partial_rate(a: f64, eta: f64, s: f64, theta_max: f64, dj: &mut DoubleBessel, nodes: &[f64], weights: &[f64]) -> f64 {
    nodes.iter()
        // integrate over 0 to pi/2, then multiply by 4
        .map(|x| 0.5 * (x + 1.0) * theta_max)
        .zip(weights.iter())
        .map(|(theta, w)|
            4.0 * (0.5 * theta_max) * w * double_diff_partial_rate(a, eta, s, theta, dj)
        )
        .sum()
}

/// `double_diff_partial_rate` integrated over s and theta.
/// Multiply by alpha / eta to get dP/(ds dtheta dphase)
fn partial_rate(n: i32, a: f64, eta: f64, nodes: &[f64], weights: &[f64]) -> f64 {
    let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
    let smax = sn / (1.0 + sn);
    // approximate s where rate is maximised
    let s_peak = sn / (2.0 + sn);

    // allocate once and reuse
    let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
    let theta_max = ThetaBound::for_harmonic(n, a, eta);

    let integral: f64 = if sn < 1.0 {
        // if s_peak < 2/3 * smax
        // split integral in two: 0 to s_peak
        let lower: f64 = nodes.iter()
            .map(|x| 0.5 * (x + 1.0) * s_peak)
            .zip(weights.iter())
            .map(|(s, w)|
                w * (0.5 * s_peak) * single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)
            )
            .sum();

        // and then s_peak to s_max:
        let upper: f64 = nodes.iter()
            .map(|x| s_peak + 0.5 * (smax - s_peak) * (x + 1.0))
            .zip(weights.iter())
            .map(|(s, w)|
                w * 0.5 * (smax - s_peak) * single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)
            )
            .sum();

        upper + lower
    } else {
        // split domain into three: 0 to sm-2d, sm-2d to sp, sp to sm
        // where d = sm - sp
        let (s0, s1) = (0.0, smax - 2.0 * (smax - s_peak));
        let lower: f64 = nodes.iter()
            .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
            .zip(weights.iter())
            .map(|(s, w)|
                w * 0.5 * (s1 - s0) * single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)
            )
            .sum();

        let (s0, s1) = (smax - 2.0 * (smax - s_peak), s_peak);
        let mid: f64 = nodes.iter()
            .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
            .zip(weights.iter())
            .map(|(s, w)|
                w * 0.5 * (s1 - s0) * single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)
            )
            .sum();

        let (s0, s1) = (s_peak, smax);
        let upper: f64 = nodes.iter()
            .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
            .zip(weights.iter())
            .map(|(s, w)|
                w * 0.5 * (s1 - s0) * single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)
            )
            .sum();

        lower + mid + upper
    };

    integral
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use super::*;

    #[test]
    fn integration_domain() {
        let (n, a, eta) = (100, 10.0, 0.1);
        // bounds on s
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let smax = sn / (1.0 + sn);

        let nodes: Vec<f64> = (0..=1000).map(|i| (i as f64) / 1000.0).collect();
        let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);

        let max_theta = ThetaBound::for_harmonic(n, a, eta);

        let filename = format!("output/nlc_lp_dd_rate_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        for s in nodes.iter().map(|x| x * smax) {//.filter(|&s| s > 0.5) {
            for theta in nodes.iter().map(|x| x * consts::FRAC_PI_2) {//.filter(|&theta| theta < 0.3) {
                if theta > max_theta.at(s) {
                    continue;
                }
                let rate = double_diff_partial_rate(a, eta, s, theta, &mut dj);
                writeln!(file, "{:.6e} {:.6e} {:.6e}", s, theta, rate).unwrap();
            }
        }
    }

    #[test]
    fn theta_bounds() {
        let (n, a, eta) = (1, 10.0, 0.1);
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let smax = sn / (1.0 + sn);
        let bound = ThetaBound::for_harmonic(n, a, eta);
        println!("{:?}", bound);

        for i in 0..10 {
            let s = (i as f64) * smax / 10.0;
            let theta = bound.at(s);
            println!("{} {}", s, theta);
        }
    }

    #[test]
    fn partial_spectra() {
        let (n, a, eta): (i32, f64, f64) = (15, 10.0, 0.1);
        let sn = 2.0 * (n as f64) * eta / (1.0 + 0.5 * a * a);
        let s_peak = sn / (2.0 + sn);
        let smax = sn / (1.0 + sn);
        let mut dj = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, (n as f64) * 0.5);
        let theta_max = ThetaBound::for_harmonic(n, a, eta);
        let nodes = &GAUSS_128_NODES;
        let weights = &GAUSS_128_WEIGHTS;

        let pts: Vec<(f64, f64)> = if sn < 1.0 {
            // if s_peak < 2/3 * smax
            // split integral in two: 0 to s_peak
            let mut lower: Vec<(f64, f64)> = nodes.iter()
                .map(|x| 0.5 * (x + 1.0) * s_peak)
                .map(|s| (s, single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)))
                .collect();

            // and then s_peak to s_max:
            let mut upper = nodes.iter()
                .map(|x| s_peak + 0.5 * (smax - s_peak) * (x + 1.0))
                .map(|s| (s, single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)))
                .collect();

            lower.append(&mut upper);
            lower
        } else {
            // split domain into three: 0 to sm-2d, sm-2d to sp, sp to sm
            // where d = sm - sp
            let (s0, s1) = (0.0, smax - 2.0 * (smax - s_peak));
            let mut lower: Vec<(f64, f64)> = nodes.iter()
                .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
                .map(|s| (s, single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)))
                .collect();

            let (s0, s1) = (smax - 2.0 * (smax - s_peak), s_peak);
            let mut mid = nodes.iter()
                .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
                .map(|s| (s, single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)))
                .collect();

            let (s0, s1) = (s_peak, smax);
            let mut upper = nodes.iter()
                .map(|x| s0 + 0.5 * (s1 - s0) * (x + 1.0))
                .map(|s| (s, single_diff_partial_rate(a, eta, s, theta_max.at(s), &mut dj, nodes, weights)))
                .collect();

            lower.append(&mut mid);
            lower.append(&mut upper);
            lower
        };

        let filename = format!("output/nlc_lp_sd_rate_{}_{}_{}.dat", n, a, eta);
        let mut file = File::create(&filename).unwrap();
        for pt in &pts {
            writeln!(file, "{:.6e} {:.6e}", pt.0, pt.1).unwrap();
        }
    }

    #[test]
    fn partial_rates() {
        // let (a, eta): (f64, f64) = (1.0, 0.1);
        // let nmax = (10.0 * (1.0 + a * a)).ceil() as i32;
        // let filename = format!("output/nlc_lp_rates_{}_{}.dat", a, eta);
        // let mut file = File::create(&filename).unwrap();
        // for n in 1..=nmax {
        //     let rate = partial_rate(n, a, eta, &GAUSS_32_NODES, &GAUSS_32_WEIGHTS);
        //     println!("n = {}, rate = {:.6e}", n, rate);
        //     writeln!(file, "{:.6e}", rate).unwrap();
        // }

        let (a, eta): (f64, f64) = (0.1, 0.1);
        for n in [1, 2, 5, 10] {
            let rate16 = partial_rate(n, a, eta, &GAUSS_16_NODES, &GAUSS_16_WEIGHTS);
            let rate32 = partial_rate(n, a, eta, &GAUSS_32_NODES, &GAUSS_32_WEIGHTS);
            let rate64 = partial_rate(n, a, eta, &GAUSS_64_NODES, &GAUSS_64_WEIGHTS);
            let rate128 = partial_rate(n, a, eta, &GAUSS_128_NODES, &GAUSS_128_WEIGHTS);
            println!(
                "a = {}, n = {}: rate = {:.3e} [128], error = {:.3e} [16], {:.3e} [32], {:.3e} [64]",
                a, n, rate128,
                (rate16 - rate128).abs() / rate128,
                (rate32 - rate128).abs() / rate128,
                (rate64 - rate128).abs() / rate128,
            );
        }

        let (a, eta): (f64, f64) = (1.0, 0.1);
        for n in [1, 5, 10, 20] {
            let rate16 = partial_rate(n, a, eta, &GAUSS_16_NODES, &GAUSS_16_WEIGHTS);
            let rate32 = partial_rate(n, a, eta, &GAUSS_32_NODES, &GAUSS_32_WEIGHTS);
            let rate64 = partial_rate(n, a, eta, &GAUSS_64_NODES, &GAUSS_64_WEIGHTS);
            let rate128 = partial_rate(n, a, eta, &GAUSS_128_NODES, &GAUSS_128_WEIGHTS);
            println!(
                "a = {}, n = {}: rate = {:.3e} [128], error = {:.3e} [16], {:.3e} [32], {:.3e} [64]",
                a, n, rate128,
                (rate16 - rate128).abs() / rate128,
                (rate32 - rate128).abs() / rate128,
                (rate64 - rate128).abs() / rate128,
            );
        }

        let (a, eta): (f64, f64) = (10.0, 0.1);
        for n in [1, 10, 100, 1000] {
            let rate16 = partial_rate(n, a, eta, &GAUSS_16_NODES, &GAUSS_16_WEIGHTS);
            let rate32 = partial_rate(n, a, eta, &GAUSS_32_NODES, &GAUSS_32_WEIGHTS);
            let rate64 = partial_rate(n, a, eta, &GAUSS_64_NODES, &GAUSS_64_WEIGHTS);
            let rate128 = partial_rate(n, a, eta, &GAUSS_128_NODES, &GAUSS_128_WEIGHTS);
            println!(
                "a = {}, n = {}: rate = {:.3e} [128], error = {:.3e} [16], {:.3e} [32], {:.3e} [64]",
                a, n, rate128,
                (rate16 - rate128).abs() / rate128,
                (rate32 - rate128).abs() / rate128,
                (rate64 - rate128).abs() / rate128,
            );
        }

        let (a, eta): (f64, f64) = (0.1, 0.01);
        for n in [1, 5, 10, 20] {
            let rate16 = partial_rate(n, a, eta, &GAUSS_16_NODES, &GAUSS_16_WEIGHTS);
            let rate32 = partial_rate(n, a, eta, &GAUSS_32_NODES, &GAUSS_32_WEIGHTS);
            let rate64 = partial_rate(n, a, eta, &GAUSS_64_NODES, &GAUSS_64_WEIGHTS);
            let rate128 = partial_rate(n, a, eta, &GAUSS_128_NODES, &GAUSS_128_WEIGHTS);
            println!(
                "a = {}, n = {}: rate = {:.3e} [128], error = {:.3e} [16], {:.3e} [32], {:.3e} [64]",
                a, n, rate128,
                (rate16 - rate128).abs() / rate128,
                (rate32 - rate128).abs() / rate128,
                (rate64 - rate128).abs() / rate128,
            );
        }

        let (a, eta): (f64, f64) = (1.0, 0.01);
        for n in [1, 5, 10, 20] {
            let rate16 = partial_rate(n, a, eta, &GAUSS_16_NODES, &GAUSS_16_WEIGHTS);
            let rate32 = partial_rate(n, a, eta, &GAUSS_32_NODES, &GAUSS_32_WEIGHTS);
            let rate64 = partial_rate(n, a, eta, &GAUSS_64_NODES, &GAUSS_64_WEIGHTS);
            let rate128 = partial_rate(n, a, eta, &GAUSS_128_NODES, &GAUSS_128_WEIGHTS);
            println!(
                "a = {}, n = {}: rate = {:.3e} [128], error = {:.3e} [16], {:.3e} [32], {:.3e} [64]",
                a, n, rate128,
                (rate16 - rate128).abs() / rate128,
                (rate32 - rate128).abs() / rate128,
                (rate64 - rate128).abs() / rate128,
            );
        }

        let (a, eta): (f64, f64) = (10.0, 0.01);
        for n in [1, 10, 100, 1000] {
            let rate16 = partial_rate(n, a, eta, &GAUSS_16_NODES, &GAUSS_16_WEIGHTS);
            let rate32 = partial_rate(n, a, eta, &GAUSS_32_NODES, &GAUSS_32_WEIGHTS);
            let rate64 = partial_rate(n, a, eta, &GAUSS_64_NODES, &GAUSS_64_WEIGHTS);
            let rate128 = partial_rate(n, a, eta, &GAUSS_128_NODES, &GAUSS_128_WEIGHTS);
            println!(
                "a = {}, n = {}: rate = {:.3e} [128], error = {:.3e} [16], {:.3e} [32], {:.3e} [64]",
                a, n, rate128,
                (rate16 - rate128).abs() / rate128,
                (rate32 - rate128).abs() / rate128,
                (rate64 - rate128).abs() / rate128,
            );
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
                let rate = partial_rate(n, a, eta, &GAUSS_32_NODES, &GAUSS_32_WEIGHTS);
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

static GAUSS_16_NODES: [f64; 16] = [
    -9.894009349916499e-1,
    -9.445750230732326e-1,
    -8.656312023878317e-1,
    -7.554044083550030e-1,
    -6.178762444026437e-1,
    -4.580167776572274e-1,
    -2.816035507792589e-1,
    -9.501250983763744e-2,
    9.501250983763744e-2,
    2.816035507792589e-1,
    4.580167776572274e-1,
    6.178762444026437e-1,
    7.554044083550030e-1,
    8.656312023878317e-1,
    9.445750230732326e-1,
    9.894009349916499e-1,
];

static GAUSS_16_WEIGHTS: [f64; 16] = [
    2.715245941175400e-2,
    6.225352393864800e-2,
    9.515851168249300e-2,
    1.246289712555340e-1,
    1.495959888165770e-1,
    1.691565193950025e-1,
    1.826034150449236e-1,
    1.894506104550685e-1,
    1.894506104550685e-1,
    1.826034150449236e-1,
    1.691565193950025e-1,
    1.495959888165770e-1,
    1.246289712555340e-1,
    9.515851168249300e-2,
    6.225352393864800e-2,
    2.715245941175400e-2,
];

static GAUSS_64_NODES: [f64; 64] = [
    -9.993050417357721e-1,
    -9.963401167719553e-1,
    -9.910133714767443e-1,
    -9.833362538846260e-1,
    -9.733268277899110e-1,
    -9.610087996520537e-1,
    -9.464113748584028e-1,
    -9.295691721319396e-1,
    -9.105221370785028e-1,
    -8.893154459951141e-1,
    -8.659993981540928e-1,
    -8.406292962525804e-1,
    -8.132653151227976e-1,
    -7.839723589433414e-1,
    -7.528199072605319e-1,
    -7.198818501716108e-1,
    -6.852363130542332e-1,
    -6.489654712546573e-1,
    -6.111553551723933e-1,
    -5.718956462026340e-1,
    -5.312794640198945e-1,
    -4.894031457070530e-1,
    -4.463660172534641e-1,
    -4.022701579639916e-1,
    -3.572201583376681e-1,
    -3.113228719902110e-1,
    -2.646871622087674e-1,
    -2.174236437400071e-1,
    -1.696444204239928e-1,
    -1.214628192961206e-1,
    -7.299312178779904e-2,
    -2.435029266342443e-2,
    2.435029266342443e-2,
    7.299312178779904e-2,
    1.214628192961206e-1,
    1.696444204239928e-1,
    2.174236437400071e-1,
    2.646871622087674e-1,
    3.113228719902110e-1,
    3.572201583376681e-1,
    4.022701579639916e-1,
    4.463660172534641e-1,
    4.894031457070530e-1,
    5.312794640198945e-1,
    5.718956462026340e-1,
    6.111553551723933e-1,
    6.489654712546573e-1,
    6.852363130542332e-1,
    7.198818501716108e-1,
    7.528199072605319e-1,
    7.839723589433414e-1,
    8.132653151227976e-1,
    8.406292962525804e-1,
    8.659993981540928e-1,
    8.893154459951141e-1,
    9.105221370785028e-1,
    9.295691721319396e-1,
    9.464113748584028e-1,
    9.610087996520537e-1,
    9.733268277899110e-1,
    9.833362538846260e-1,
    9.910133714767443e-1,
    9.963401167719553e-1,
    9.993050417357721e-1,
];

static GAUSS_64_WEIGHTS: [f64; 64] = [
    1.783280721696433e-3,
    4.147033260562468e-3,
    6.504457968978363e-3,
    8.846759826363948e-3,
    1.116813946013113e-2,
    1.346304789671864e-2,
    1.572603047602472e-2,
    1.795171577569734e-2,
    2.013482315353021e-2,
    2.227017380838325e-2,
    2.435270256871087e-2,
    2.637746971505466e-2,
    2.833967261425948e-2,
    3.023465707240248e-2,
    3.205792835485155e-2,
    3.380516183714161e-2,
    3.547221325688238e-2,
    3.705512854024005e-2,
    3.855015317861563e-2,
    3.995374113272034e-2,
    4.126256324262353e-2,
    4.247351512365359e-2,
    4.358372452932345e-2,
    4.459055816375656e-2,
    4.549162792741814e-2,
    4.628479658131442e-2,
    4.696818281621002e-2,
    4.754016571483031e-2,
    4.799938859645831e-2,
    4.834476223480296e-2,
    4.857546744150343e-2,
    4.869095700913972e-2,
    4.869095700913972e-2,
    4.857546744150343e-2,
    4.834476223480296e-2,
    4.799938859645831e-2,
    4.754016571483031e-2,
    4.696818281621002e-2,
    4.628479658131442e-2,
    4.549162792741814e-2,
    4.459055816375656e-2,
    4.358372452932345e-2,
    4.247351512365359e-2,
    4.126256324262353e-2,
    3.995374113272034e-2,
    3.855015317861563e-2,
    3.705512854024005e-2,
    3.547221325688238e-2,
    3.380516183714161e-2,
    3.205792835485155e-2,
    3.023465707240248e-2,
    2.833967261425948e-2,
    2.637746971505466e-2,
    2.435270256871087e-2,
    2.227017380838325e-2,
    2.013482315353021e-2,
    1.795171577569734e-2,
    1.572603047602472e-2,
    1.346304789671864e-2,
    1.116813946013113e-2,
    8.846759826363948e-3,
    6.504457968978363e-3,
    4.147033260562468e-3,
    1.783280721696433e-3,
];

static GAUSS_128_NODES: [f64; 128] = [
    -9.998248879471319e-1,
    -9.990774599773759e-1,
    -9.977332486255140e-1,
    -9.957927585349812e-1,
    -9.932571129002129e-1,
    -9.901278184917344e-1,
    -9.864067427245862e-1,
    -9.820961084357185e-1,
    -9.771984914639074e-1,
    -9.717168187471366e-1,
    -9.656543664319653e-1,
    -9.590147578536999e-1,
    -9.518019613412644e-1,
    -9.440202878302202e-1,
    -9.356743882779164e-1,
    -9.267692508789478e-1,
    -9.173101980809605e-1,
    -9.073028834017568e-1,
    -8.967532880491582e-1,
    -8.856677173453972e-1,
    -8.740527969580318e-1,
    -8.619154689395485e-1,
    -8.492629875779690e-1,
    -8.361029150609068e-1,
    -8.224431169556438e-1,
    -8.082917575079137e-1,
    -7.936572947621933e-1,
    -7.785484755064120e-1,
    -7.629743300440947e-1,
    -7.469441667970620e-1,
    -7.304675667419088e-1,
    -7.135543776835874e-1,
    -6.962147083695143e-1,
    -6.784589224477193e-1,
    -6.602976322726461e-1,
    -6.417416925623076e-1,
    -6.228021939105849e-1,
    -6.034904561585486e-1,
    -5.838180216287631e-1,
    -5.637966482266181e-1,
    -5.434383024128104e-1,
    -5.227551520511755e-1,
    -5.017595591361445e-1,
    -4.804640724041720e-1,
    -4.588814198335522e-1,
    -4.370245010371042e-1,
    -4.149063795522750e-1,
    -3.925402750332674e-1,
    -3.699395553498590e-1,
    -3.471177285976355e-1,
    -3.240884350244134e-1,
    -3.008654388776772e-1,
    -2.774626201779044e-1,
    -2.538939664226943e-1,
    -2.301735642266600e-1,
    -2.063155909020792e-1,
    -1.823343059853372e-1,
    -1.582440427142249e-1,
    -1.340591994611878e-1,
    -1.097942311276437e-1,
    -8.546364050451550e-2,
    -6.108196960413957e-2,
    -3.666379096873349e-2,
    -1.222369896061576e-2,
    1.222369896061576e-2,
    3.666379096873349e-2,
    6.108196960413957e-2,
    8.546364050451550e-2,
    1.097942311276437e-1,
    1.340591994611878e-1,
    1.582440427142249e-1,
    1.823343059853372e-1,
    2.063155909020792e-1,
    2.301735642266600e-1,
    2.538939664226943e-1,
    2.774626201779044e-1,
    3.008654388776772e-1,
    3.240884350244134e-1,
    3.471177285976355e-1,
    3.699395553498590e-1,
    3.925402750332674e-1,
    4.149063795522750e-1,
    4.370245010371042e-1,
    4.588814198335522e-1,
    4.804640724041720e-1,
    5.017595591361445e-1,
    5.227551520511755e-1,
    5.434383024128104e-1,
    5.637966482266181e-1,
    5.838180216287631e-1,
    6.034904561585486e-1,
    6.228021939105849e-1,
    6.417416925623076e-1,
    6.602976322726461e-1,
    6.784589224477193e-1,
    6.962147083695143e-1,
    7.135543776835874e-1,
    7.304675667419088e-1,
    7.469441667970620e-1,
    7.629743300440947e-1,
    7.785484755064120e-1,
    7.936572947621933e-1,
    8.082917575079137e-1,
    8.224431169556438e-1,
    8.361029150609068e-1,
    8.492629875779690e-1,
    8.619154689395485e-1,
    8.740527969580318e-1,
    8.856677173453972e-1,
    8.967532880491582e-1,
    9.073028834017568e-1,
    9.173101980809605e-1,
    9.267692508789478e-1,
    9.356743882779164e-1,
    9.440202878302202e-1,
    9.518019613412644e-1,
    9.590147578536999e-1,
    9.656543664319653e-1,
    9.717168187471366e-1,
    9.771984914639074e-1,
    9.820961084357185e-1,
    9.864067427245862e-1,
    9.901278184917344e-1,
    9.932571129002129e-1,
    9.957927585349812e-1,
    9.977332486255140e-1,
    9.990774599773759e-1,
    9.998248879471319e-1,
];

static GAUSS_128_WEIGHTS: [f64; 128] = [
    4.493809602920904e-4,
    1.045812679340349e-3,
    1.642503018669030e-3,
    2.238288430962619e-3,
    2.832751471457991e-3,
    3.425526040910216e-3,
    4.016254983738642e-3,
    4.604584256702955e-3,
    5.190161832676330e-3,
    5.772637542865699e-3,
    6.351663161707189e-3,
    6.926892566898814e-3,
    7.497981925634729e-3,
    8.064589890486058e-3,
    8.626377798616750e-3,
    9.183009871660874e-3,
    9.734153415006806e-3,
    1.027947901583216e-2,
    1.081866073950308e-2,
    1.135137632408042e-2,
    1.187730737274028e-2,
    1.239613954395092e-2,
    1.290756273926735e-2,
    1.341127128861633e-2,
    1.390696413295199e-2,
    1.439434500416685e-2,
    1.487312260214731e-2,
    1.534301076886514e-2,
    1.580372865939935e-2,
    1.625500090978519e-2,
    1.669655780158920e-2,
    1.712813542311138e-2,
    1.754947582711770e-2,
    1.796032718500869e-2,
    1.836044393733134e-2,
    1.874958694054471e-2,
    1.912752360995095e-2,
    1.949402805870660e-2,
    1.984888123283086e-2,
    2.019187104213004e-2,
    2.052279248696007e-2,
    2.084144778075115e-2,
    2.114764646822135e-2,
    2.144120553920846e-2,
    2.172194953805208e-2,
    2.198971066846049e-2,
    2.224432889379977e-2,
    2.248565203274497e-2,
    2.271353585023646e-2,
    2.292784414368685e-2,
    2.312844882438703e-2,
    2.331522999406276e-2,
    2.348807601653591e-2,
    2.364688358444762e-2,
    2.379155778100340e-2,
    2.392201213670346e-2,
    2.403816868102405e-2,
    2.413995798901928e-2,
    2.422731922281525e-2,
    2.430020016797187e-2,
    2.435855726469063e-2,
    2.440235563384958e-2,
    2.443156909785005e-2,
    2.444618019626252e-2,
    2.444618019626252e-2,
    2.443156909785005e-2,
    2.440235563384958e-2,
    2.435855726469063e-2,
    2.430020016797187e-2,
    2.422731922281525e-2,
    2.413995798901928e-2,
    2.403816868102405e-2,
    2.392201213670346e-2,
    2.379155778100340e-2,
    2.364688358444762e-2,
    2.348807601653591e-2,
    2.331522999406276e-2,
    2.312844882438703e-2,
    2.292784414368685e-2,
    2.271353585023646e-2,
    2.248565203274497e-2,
    2.224432889379977e-2,
    2.198971066846049e-2,
    2.172194953805208e-2,
    2.144120553920846e-2,
    2.114764646822135e-2,
    2.084144778075115e-2,
    2.052279248696007e-2,
    2.019187104213004e-2,
    1.984888123283086e-2,
    1.949402805870660e-2,
    1.912752360995095e-2,
    1.874958694054471e-2,
    1.836044393733134e-2,
    1.796032718500869e-2,
    1.754947582711770e-2,
    1.712813542311138e-2,
    1.669655780158920e-2,
    1.625500090978519e-2,
    1.580372865939935e-2,
    1.534301076886514e-2,
    1.487312260214731e-2,
    1.439434500416685e-2,
    1.390696413295199e-2,
    1.341127128861633e-2,
    1.290756273926735e-2,
    1.239613954395092e-2,
    1.187730737274028e-2,
    1.135137632408042e-2,
    1.081866073950308e-2,
    1.027947901583216e-2,
    9.734153415006806e-3,
    9.183009871660874e-3,
    8.626377798616750e-3,
    8.064589890486058e-3,
    7.497981925634729e-3,
    6.926892566898814e-3,
    6.351663161707189e-3,
    5.772637542865699e-3,
    5.190161832676330e-3,
    4.604584256702955e-3,
    4.016254983738642e-3,
    3.425526040910216e-3,
    2.832751471457991e-3,
    2.238288430962619e-3,
    1.642503018669030e-3,
    1.045812679340349e-3,
    4.493809602920904e-4,
];
