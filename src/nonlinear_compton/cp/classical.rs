//! Rates and spectra for classically polarized backgrounds,
//! in the classical regime eta -> 0

use std::f64::consts;
use rand::prelude::*;
use crate::pwmci;
use crate::special_functions::*;
use crate::geometry::StokesVector;
use super::{
    spectrum_low_eta,
};

mod rate_table;
mod cdf_table;

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Compton rates. Equivalent to calling
/// ```
/// let nmax = (10.0 * (1.0 + a.powi(3))) as i32;
/// let rate = eta * (1..=nmax).map(|n| integrated_spectrum_low_eta(n, a)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
#[allow(unused_parens)]
pub fn rate(a: f64, eta: f64) -> Option<f64> {
    let x = a.ln();

    if x < rate_table::MIN {
        // linear Thomson scattering
        Some(2.0 * a * a * eta / 3.0)
    } else {
        let ix = ((x - rate_table::MIN) / rate_table::STEP) as usize;

        if ix < rate_table::N_COLS - 1 {
            let dx = (x - rate_table::MIN) / rate_table::STEP - (ix as f64);
            let f = (
                (1.0 - dx) * rate_table::TABLE[ix]
                + dx * rate_table::TABLE[ix+1]
            );
            Some(eta * f.exp())
        } else {
            eprintln!("NLC (classical CP) rate lookup out of bounds: a = {:.3e}", a);
            None
        }
    }
}

/// Returns the largest value of spectrum_low_eta, multiplied by a small safety factor.
fn ceiling_spectrum_low_eta(n: i32, a: f64) -> f64 {
    let n_switch = (3.0 * (1.0 + 0.54 * a.powf(2.75))).round();

    let max = if n == 1 {
        spectrum_low_eta(n, a, 0.99)
    } else if a < 0.15 || n > n_switch as i32 {
        spectrum_low_eta(n, a, 0.5)
    } else {
        let b = if a < 0.5 {
            2.3 * a
        } else {
            0.8 * (-(a - 0.5) / 3.5).exp() + 0.35
        };
        let v = 1.0 - 0.5 * (n as f64).ln().powf(b) / n_switch.ln().powf(b);
        let delta = 4;
        (-delta..delta)
            .map(|i| (v + 0.1 * (i as f64) / (delta as f64)).min(1.0))
            .map(|v| spectrum_low_eta(n, a, v))
            .reduce(f64::max)
            .unwrap()
    };

    1.1 * max
}

fn rescale(frac: f64, table: &[[f64; 2]; 32]) -> (f64, [[f64; 2]; 31]) {
    let mut output = [[0.0; 2]; 31];
    for i in 0..31 {
        output[i][0] = table[i][0].ln();
        output[i][1] = (-1.0 * (1.0 - table[i][1]).ln()).ln();
    }
    let frac2 = (-1.0 * (1.0 - frac).ln()).ln();
    (frac2, output)
}

/// Obtain harmonic index by inverting frac = cdf(n), where 0 <= frac < 1 and
/// the cdf is tabulated.
fn get_harmonic_index(a: f64, frac: f64) -> i32 {
    if a.ln() <= cdf_table::MIN {
        // first harmonic only
       1
    } else {
        let ix = ((a.ln() - cdf_table::MIN) / cdf_table::STEP) as usize;
        let dx = (a.ln() - cdf_table::MIN) / cdf_table::STEP - (ix as f64);

        let index = [
            ix,
            ix + 1,
        ];

        let weight = [
            1.0 - dx,
            dx,
        ];

        let n_alt: f64 = index.iter()
            .zip(weight.iter())
            .map(|(i, w)| {
                let table = &cdf_table::TABLE[*i];
                let n = if frac <= table[0][1] {
                    0.9
                } else {
                    let (rs_frac, rs_table) = rescale(frac, table);
                    pwmci::Interpolant::new(&rs_table)
                        .extrapolate(true)
                        .invert(rs_frac)
                        .map(f64::exp)
                        .unwrap()
                };
                n * w
            })
            .sum();

        n_alt.ceil() as i32
    }
}

/// Returns a pseudorandomly sampled n (harmonic order), s (lightfront momentum
/// transfer) and theta (azimuthal angle in the ZMF) for a photon emission that
/// occurs at normalized amplitude a and energy parameter eta.
pub fn sample<R: Rng>(a: f64, eta: f64, rng: &mut R, fixed_n: Option<i32>) -> (i32, f64, f64, StokesVector) {
    let n = fixed_n.unwrap_or_else(|| {
        let frac = rng.gen::<f64>();
        get_harmonic_index(a, frac) // via lookup of cdf
    });

    // Approximate maximum value of the probability density:
    let max = ceiling_spectrum_low_eta(n, a);

    // Rejection sampling
    let v = loop {
        let v = rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = spectrum_low_eta(n, a, v);
        if u <= f / max {
            break v;
        }
    };

    // Four-momentum transfer s = k.l / k.q
    let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
    let s = v * sn;

    // Azimuthal angle in ZMF
    let theta = 2.0 * consts::PI * rng.gen::<f64>();

    // Stokes parameters
    let z = (
        ((4 * n * n) as f64)
        * (a * a / (1.0 + a * a))
        * v * (1.0 - v)
    ).sqrt();

    let (j_nm1, j_n, j_np1) = z.j_pm(n);

    // Defined w.r.t. rotated basis: x || k x k' and y || k'_perp
    let sv: StokesVector = {
        let xi_0 = 2.0 * (j_nm1 * j_nm1 + j_np1 * j_np1 - 2.0 * j_n * j_n) - (2.0 * j_n / a).powi(2);
        let xi_1 = 2.0 * (j_nm1 * j_nm1 + j_np1 * j_np1 - 2.0 * j_n * j_n) + 8.0 * j_n * j_n * (1.0 - ((n as f64) / z).powi(2) - 0.5 / (a * a));
        let xi_2 = 0.0;
        let xi_3 = 2.0 * (1.0 - 2.0 * v) * (j_nm1 * j_nm1 - j_np1 * j_np1); // +/-1 depending on wave handedness
        [1.0, xi_1 / xi_0, xi_2 / xi_0, xi_3 / xi_0].into()
    };

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
    use super::*;

    #[test]
    fn rate_ceiling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _i in 0..100 {
            let a = (0.2_f64.ln() + (20_f64.ln() - 0.2_f64.ln()) * rng.gen::<f64>()).exp();
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
                let true_max = (0..100)
                    .map(|i| 0.5 * (1.0 + (i as f64) / 100.0))
                    .map(|v| spectrum_low_eta(*n, a, v))
                    .reduce(f64::max)
                    .unwrap();
                let max = ceiling_spectrum_low_eta(*n, a);
                let err = (true_max - max) / true_max;
                println!(
                    "a = {:>9.3e}, n = {:>4} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                    a, n, true_max, max, 100.0 * err,
                );
                assert!(err < 0.0 || true_max == 0.0);
            }
        }
    }

    #[test]
    #[ignore]
    fn create_rate_table() {
        use super::super::integrated_spectrum_low_eta;

        const LOW_A_LIMIT: f64 = 0.02;
        const A_DENSITY: usize = 20; // points per order of magnitude
        const N_COLS: usize = 61; // pts in a0 direction
        let mut table = [0.0; N_COLS];

        let mut pts: Vec<(usize, f64, i32)> = Vec::new();
        for j in 0..N_COLS {
            let a = LOW_A_LIMIT * 10.0f64.powf((j as f64) / (A_DENSITY as f64));
            let n_max = (10.0 * (1.0 + a.powi(3))) as i32;
            pts.push((j, a, n_max));
        }

        let pts: Vec<(usize, f64, [[f64; 2]; 32])> = pts.into_iter()
            .map(|(j, a, n_max)| {
                let rates: Vec<[f64; 2]> = (1..(n_max+1))
                    .map(|n| [n as f64, integrated_spectrum_low_eta(n, a)])
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

                println!("CP classical NLC: a = {:.3e}, ln(rate) = {:.6e}", a, rate.ln());
                (j, rate, cdf)
            })
            .collect();

        for (j, rate, _) in &pts {
            table[*j] = *rate;
        }

        let mut file = File::create("output/nlc_cp_cls_rate_table.rs").unwrap();
        writeln!(file, "pub const N_COLS: usize = {};", N_COLS).unwrap();
        writeln!(file, "pub const MIN: f64 = {:.12e};", LOW_A_LIMIT.ln()).unwrap();
        writeln!(file, "pub const STEP: f64 = {:.12e};", consts::LN_10 / (A_DENSITY as f64)).unwrap();
        writeln!(file, "pub const TABLE: [f64; {}] = [", N_COLS).unwrap();
        for row in table.iter() {
            let val = row.ln();
            if val.is_finite() {
                writeln!(file, "\t{:>18.12e},", val).unwrap();
            } else {
                writeln!(file, "\t{:>18},", "NEG_INFINITY").unwrap();
            }
        }
        writeln!(file, "];").unwrap();

        let mut file = File::create("output/nlc_cp_cls_cdf_table.rs").unwrap();
        writeln!(file, "pub const MIN: f64 = {:.12e};", LOW_A_LIMIT.ln()).unwrap();
        writeln!(file, "pub const STEP: f64 = {:.12e};", consts::LN_10 / (A_DENSITY as f64)).unwrap();
        writeln!(file, "pub const TABLE: [[[f64; 2]; 32]; {}] = [", N_COLS).unwrap();
        for (_, _, cdf) in &pts {
            write!(file, "\t[").unwrap();
            for entry in cdf.iter().take(31) {
                write!(file, "[{:>18.12e}, {:>18.12e}], ", entry[0], entry[1]).unwrap();
            }
            writeln!(file, "[{:>18.12e}, {:>18.12e}]],", cdf[31][0], cdf[31][1]).unwrap();
        }
        writeln!(file, "];").unwrap();
    }
}