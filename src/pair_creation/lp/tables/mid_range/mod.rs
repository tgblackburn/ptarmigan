//! For 0.05 < a < 1.0 and 0.1 < eta < 2

use crate::pwmci;
use super::PolDep;

mod total;
mod cdf;

pub fn contains(a: f64, eta: f64) -> bool {
    let ln_a = a.ln();
    let ln_eta_prime = (eta / (1.0 + 0.5 * a * a)).ln();
    ln_a > total::LN_MIN_A
        && a < 2.0
        && ln_eta_prime > total::LN_MIN_ETA_PRIME
        && ln_eta_prime < total::LN_MAX_ETA_PRIME
}

fn interpolate_in_eta_prime(eta_prime: f64, ia: usize) -> PolDep {
    use total::*;

    // eta' = 2 / (1 + i/r)
    let ie = (ETA_PRIME_DENSITY * (LN_MAX_ETA_PRIME.exp() / eta_prime - 1.0)) as usize;

    // fix ie if it's next to a harmonic boundary
    let ie = if ie % (ETA_PRIME_DENSITY as usize) == (ETA_PRIME_DENSITY as usize) - 1 {
        ie - 1
    } else {
        ie
    };

    // remember increasing ie reduces eta
    let lower = (
        2.0 / (1.0 + ((ie+2) as f64) / ETA_PRIME_DENSITY),
        PolDep::from(TABLE[ie+2][ia])
    );

    let mid = (
        2.0 / (1.0 + ((ie+1) as f64) / ETA_PRIME_DENSITY),
        PolDep::from(TABLE[ie+1][ia])
    );

    let upper = (
        2.0 / (1.0 + (ie as f64) / ETA_PRIME_DENSITY),
        PolDep::from(TABLE[ie][ia])
    );

    // println!("lower = {:?}", lower);
    // println!("mid = {:?}", mid);
    // println!("upper = {:?}", upper);

    // fit a power law y = alpha (e - e_0)^m + b
    let m = ((upper.1 - lower.1) / (mid.1 - lower.1)).ln() / ((upper.0 - lower.0) / (mid.0 - lower.0)).ln();
    let alpha = (mid.1 - lower.1) / PolDep::from(mid.0 - lower.0).powf(m);
    //println!("alpha = {}, m = {}, returning {}", alpha, m, alpha * (eta_prime - lower.0).powf(m) + lower.1);

    let mut interp = alpha * PolDep::from(eta_prime - lower.0).powf(m) + lower.1;

    // Power-law fit fails if !(lower < mid < upper), fall back to linear interpolation
    if !interp.par().is_finite() {
        let par = if eta_prime < mid.0 {
            let w = (eta_prime.ln() - lower.0.ln()) / (mid.0.ln() - lower.0.ln());
            (1.0 - w) * lower.1.par() + w * mid.1.par()
        } else {
            let w = (eta_prime.ln() - mid.0.ln()) / (upper.0.ln() - mid.0.ln());
            (1.0 - w) * mid.1.par() + w * upper.1.par()
        };
        interp.with_par(par);
    }

    if !interp.perp().is_finite() {
        let perp = if eta_prime < mid.0 {
            let w = (eta_prime.ln() - lower.0.ln()) / (mid.0.ln() - lower.0.ln());
            (1.0 - w) * lower.1.perp() + w * mid.1.perp()
        } else {
            let w = (eta_prime.ln() - mid.0.ln()) / (upper.0.ln() - mid.0.ln());
            (1.0 - w) * mid.1.perp() + w * upper.1.perp()
        };
        interp.with_perp(perp);
    }

    interp
}

pub fn interpolate(a: f64, eta: f64) -> [f64; 2] {
    use total::*;
    let ia = ((a.ln() - LN_MIN_A) / LN_A_STEP) as usize;
    let eta_prime = eta / (1.0 + 0.5 * a * a);

    let f = if a < 0.5 {
        let da = (a.ln() - LN_MIN_A) / LN_A_STEP - (ia as f64);
        (1.0 - da) * interpolate_in_eta_prime(eta_prime, ia) + da * interpolate_in_eta_prime(eta_prime, ia+1)
    } else {
        let a_min = (LN_MIN_A + (ia as f64) * LN_A_STEP).exp();
        let a_max = (LN_MIN_A + ((ia + 1) as f64) * LN_A_STEP).exp();
        let da = (a - a_min) / (a_max - a_min);
        ((1.0 - da) * a_min * interpolate_in_eta_prime(eta_prime, ia) + da * a_max * interpolate_in_eta_prime(eta_prime, ia+1)) / a
    };

    f.exp().into_inner()
}

fn interpolate_and_project(a: f64, eta: f64, sv1: f64) -> f64 {
    let [a, b] = interpolate(a, eta);
    0.5 * ((a + b) + sv1 * (a - b))
}

/// Estimates the pair-creation rate for the lowest accessible harmonic,
/// returning the harmonic index and the interpolated rate
fn interpolate_lowest_harmonic(a: f64, eta: f64, sv1: f64) -> (i32, f64) {
    use cdf::*;
    use super::TotalRate;

    let ia = ((a.ln() - LN_MIN_A) / LN_A_STEP) as usize;
    let da = (a.ln() - LN_MIN_A) / LN_A_STEP - (ia as f64);

    let eta_prime = eta / (1.0 + 0.5 * a * a);
    // eta' = 2 / (1 + i/r)
    let ie = (total::ETA_PRIME_DENSITY * (LN_MAX_ETA_PRIME.exp() / eta_prime - 1.0)) as usize;

    // 2/(1 + ie/r) >= eta_prime > 2/[1 + (ie+1)/r]
    let de = {
        let lower = LN_MAX_ETA_PRIME.exp() / (1.0 + ((ie+1) as f64) / total::ETA_PRIME_DENSITY);
        let upper = LN_MAX_ETA_PRIME.exp() / (1.0 + (ie as f64) / total::ETA_PRIME_DENSITY);
        (eta_prime.ln() - lower.ln()) / (upper.ln() - lower.ln())
    };

    let index = [
        N_COLS * ie + ia,
        N_COLS * ie + ia + 1,
        N_COLS * (ie + 1) + ia,
        N_COLS * (ie + 1) + (ia + 1),
    ];

    let weight = [
        (1.0 - da) * de,
        da * de,
        (1.0 - da) * (1.0 - de),
        da * (1.0 - de),
    ];

    let (n_min, _) = TotalRate::new(a, eta).sum_limits();

    let total_rate = interpolate_and_project(a, eta, sv1);

    let rate = index.iter()
        .zip(weight.iter())
        .map(|(i, w)| {
            let table = &TABLE[*i];
            let frac = if table[0][0] > (n_min as f64) {
                0.0
            } else {
                let f = PolDep::new(table[0][1], table[0][2]).interp(sv1);
                let f_max = PolDep::new(table[15][1], table[15][2]).interp(sv1);
                f / f_max
            };
            w * (frac * total_rate).powi(3)
        })
        .sum::<f64>()
        .cbrt();

    (n_min, rate)
}

pub fn invert(a: f64, eta: f64, sv1: f64, frac: f64) -> i32 {
    use cdf::*;
    let ia = ((a.ln() - LN_MIN_A) / LN_A_STEP) as usize;
    let da = (a.ln() - LN_MIN_A) / LN_A_STEP - (ia as f64);

    let eta_prime = eta / (1.0 + 0.5 * a * a);
    // eta' = 2 / (1 + i/r)
    let ie = (total::ETA_PRIME_DENSITY * (LN_MAX_ETA_PRIME.exp() / eta_prime - 1.0)) as usize;

    // fix ie if it's next to a harmonic boundary
    let next_to_bdy = ie % (total::ETA_PRIME_DENSITY as usize) == (total::ETA_PRIME_DENSITY as usize) - 1;

    let ie = if next_to_bdy {
        ie - 1
    } else {
        ie
    };

    // 2/(1 + ie/r) >= eta_prime > 2/[1 + (ie+1)/r]
    let de = {
        let lower = LN_MAX_ETA_PRIME.exp() / (1.0 + ((ie+1) as f64) / total::ETA_PRIME_DENSITY);
        let upper = LN_MAX_ETA_PRIME.exp() / (1.0 + (ie as f64) / total::ETA_PRIME_DENSITY);
        (eta_prime.ln() - lower.ln()) / (upper.ln() - lower.ln())
    };

    let index = [
        N_COLS * ie + ia,
        N_COLS * ie + ia + 1,
        N_COLS * (ie + 1) + ia,
        N_COLS * (ie + 1) + (ia + 1),
    ];

    let weight = [
        (1.0 - da) * de,
        da * de,
        (1.0 - da) * (1.0 - de),
        da * (1.0 - de),
    ];

    if next_to_bdy {
        let (n_min, rate_lowest) = interpolate_lowest_harmonic(a, eta, sv1);
        let total_rate = interpolate_and_project(a, eta, sv1);
        let p_lowest = rate_lowest / total_rate;
        // println!("n_min = {}, frac = {:.3e}, p_lowest = {:.3e}", n_min, frac, p_lowest);
        if frac <= p_lowest {
            n_min
        } else {
            let n: f64 = index.iter()
                .zip(weight.iter())
                .map(|(i, w)| {
                    let src = &TABLE[*i];
                    let f_max = PolDep::new(src[15][1], src[15][2]).interp(sv1);
                    let mut table = [[0.0; 2]; 16];

                    for i in 0..16 {
                        table[i][0] = src[i][0];
                        let f = PolDep::new(src[i][1], src[i][2]).interp(sv1);
                        table[i][1] = f / f_max;
                    }

                    // if lowest harmonic in table is equal to n_min, ignore it
                    // we've already taken care of this
                    if table[0][0] == n_min as f64 {
                        // adjust cdf values so that table[0][1] = p_lowest,
                        // so that it's never generated
                        let t01 = table[0][1];
                        for t in table.iter_mut() {
                            t[1] = (t01 - t[1] + p_lowest * (t[1] - 1.0)) / (t01 - 1.0);
                        }
                    }

                    let n = pwmci::Interpolant::new(&table).invert(frac).unwrap();
                    n * w
                })
                .sum();

            n.ceil() as i32
        }
    } else {
        let n: f64 = index.iter()
            .zip(weight.iter())
            .map(|(i, w)| {
                let src = &TABLE[*i];
                let f_max = PolDep::new(src[15][1], src[15][2]).interp(sv1);
                let mut table = [[0.0; 2]; 16];

                for i in 0..16 {
                    table[i][0] = src[i][0];
                    let f = PolDep::new(src[i][1], src[i][2]).interp(sv1);
                    table[i][1] = f / f_max;
                }

                let n = if frac <= table[0][1] {
                    table[0][0] - 0.1
                } else {
                    pwmci::Interpolant::new(&table).invert(frac).unwrap()
                };
                n * w
            })
            .sum();

        n.ceil() as i32
    }

}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_xoshiro::*;
    use crate::pair_creation::lp::{DoubleDiffPartialRate, TotalRate};
    use super::*;

    #[test]
    fn inversion() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let sv1 = 0.0;
        let pts = [
            (0.075, 0.6),
            (0.075, 0.13),
            (0.75, 0.6),
            (0.75, 0.13),
            (0.25, 0.34), // across bdy
            (0.25, 0.35),
            (0.45, 0.2),
            (0.445, 0.2),
            (0.43, 0.2),
            (0.4, 0.2),
            (0.1, 0.2),
            (0.5, 0.1),
            (0.8, 0.1),
        ];

        for (a, eta) in &pts {
            let rate = TotalRate::new(*a, *eta);
            let (n_min, n_max) = rate.sum_limits();
            let harmonics: Vec<_> = ((n_min-1)..=n_max).collect();
            let rates: Vec<_> = harmonics.iter().map(|n| {
                let pr = DoubleDiffPartialRate::new(*n, *a, *eta).integrate();
                0.5 * ((pr.re + pr.im) + sv1 * (pr.re - pr.im))
            }).collect();
            let total: f64 = rates.iter().sum();

            println!("a = {}, eta = {}, r({}) = {:.3e}:", a, eta, n_min, interpolate_lowest_harmonic(*a, *eta, sv1).1 / interpolate_and_project(*a, *eta, sv1));

            let mut counts = [0.0; 16];
            for _ in 0..1_000_000 {
                let n = invert(*a, *eta, sv1, rng.gen());
                for i in 0..harmonics.len() {
                    if n == harmonics[i] as i32 && i < 16 {
                        counts[i] += 1.0 / 1_000_000.0;
                    }
                }
            }

            for ((n, rate), count) in harmonics.iter().zip(rates.iter()).zip(counts.iter()) {
                let target = rate / total;
                if target == 0.0 && *count == 0.0 {
                    continue;
                }
                let error = (target - count) / target;
                println!("\t{:>4} {:>9.3e} {:>9.3e} [{:>6.2}%]", n, target, count, 100.0 * error);
            };
        }
    }

    #[test]
    fn lowest_harmonic() {
        let sv1 = 0.0;
        let a_min = 0.4_f64;
        let a_max = 2.0_f64;
        let pts: Vec<_> = (0..20)
            .map(|i| (
                (a_min.ln() + (a_max.ln() - a_min.ln()) * (i as f64) / 20.0).exp(),
                0.2
            ))
            .collect();

        for (a, eta) in &pts {
            let (n_min, value) = interpolate_lowest_harmonic(*a, *eta, sv1);
            let target = DoubleDiffPartialRate::new(n_min, *a, *eta).integrate();
            let target = 0.5 * ((target.re + target.im) + sv1 * (target.re - target.im));
            let error = (target - value) / target;
            println!(
                "a = {:.3e}, eta = {:.3e}, n_min = {}: target = {:.3e}, value = {:.3e}, error = {:.2}%",
                a, eta, n_min, target, value, 100.0 * error,
            );
            assert!(error.abs() < 0.2 || target < 1.0e-12);
        }
    }
}