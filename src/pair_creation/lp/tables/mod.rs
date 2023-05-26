//! Tables for the total rate and CDF as a function of harmonic order

use crate::pwmci;
use super::TotalRate;

pub(super) mod mid_range;

/// Tabulated total rate for 0.1 < a < 20
mod total;

/// Tabulated CDF as a function of harmonic order for 0.1 < a < 20
mod cdf;

/// Utilities for extracting data from pol-dependent tables
mod pol_dep;
use pol_dep::PolDep;

pub fn contains(a: f64, eta: f64) -> bool {
    a.ln() >= total::LN_MIN_A && a < 25.0 && eta.ln() >= total::LN_MIN_ETA && eta < 2.0
}

#[allow(unused_parens)]
pub fn interpolate(a: f64, eta: f64) -> [f64; 2] {
    use total::*;

    let ia = ((a.ln() - LN_MIN_A) / LN_A_STEP) as usize;
    let ie = ((eta.ln() - LN_MIN_ETA) / LN_ETA_STEP) as usize;

    // bounds must be checked before calling
    assert!(ia < N_COLS - 1);
    assert!(ie < N_ROWS - 1);

    if a * eta > 2.0 {
        // linear interpolation of: log y against log x, best for power law
        let da = (a.ln() - LN_MIN_A) / LN_A_STEP - (ia as f64);
        let de = (eta.ln() - LN_MIN_ETA) / LN_ETA_STEP - (ie as f64);
        let f = (
            (1.0 - da) * (1.0 - de) * PolDep::from(TABLE[ie][ia])
            + da * (1.0 - de) * PolDep::from(TABLE[ie][ia+1])
            + (1.0 - da) * de * PolDep::from(TABLE[ie+1][ia])
            + da * de * PolDep::from(TABLE[ie+1][ia+1])
        );
        f.exp().into_inner()
    } else {
        // linear interpolation of: log y against 1/x, best for f(x) ~ exp(-1/x)
        let a_min = (LN_MIN_A + (ia as f64) * LN_A_STEP).exp();
        let a_max = (LN_MIN_A + ((ia+1) as f64) * LN_A_STEP).exp();
        let eta_min = (LN_MIN_ETA + (ie as f64) * LN_ETA_STEP).exp();
        let eta_max = (LN_MIN_ETA + ((ie+1) as f64) * LN_ETA_STEP).exp();
        let da = ((a_min * a_max / a) - a_min) / (a_max - a_min);
        let de = ((eta_min * eta_max / eta) - eta_min) / (eta_max - eta_min);
        let f: PolDep = (
            PolDep::from(TABLE[ie][ia]) * da * de
            + PolDep::from(TABLE[ie][ia+1]) * (1.0 - da) * de
            + PolDep::from(TABLE[ie+1][ia]) * da * (1.0 - de)
            + PolDep::from(TABLE[ie+1][ia+1]) * (1.0 - da) * (1.0 -de)
        );
        f.exp().into_inner()
    }
}

pub fn invert(a: f64, eta: f64, sv1: f64, frac: f64) -> i32 {
    use cdf::*;

    let ia = (a.ln() - MIN[0]) / STEP[0];
    let da = ia.fract();
    let ia = ia as usize;
    let ie = (eta.ln() - MIN[1]) / STEP[1];
    let de = ie.fract();
    let ie = ie as usize;

    let index = [
        N_COLS * ie + ia,
        N_COLS * ie + ia + 1,
        N_COLS * (ie + 1) + ia,
        N_COLS * (ie + 1) + (ia + 1),
    ];

    let weight = [
        (1.0 - da) * (1.0 - de),
        da * (1.0 - de),
        (1.0 - da) * de,
        da * de,
    ];

    let u: f64 = index.iter()
        .zip(weight.iter())
        .map(|(i, w)| {
            let src = &TABLE[*i];
            let n_min = src[0][0];
            let f_max = PolDep::new(src[15][1], src[15][2]).interp(sv1);

            let mut table = [[0.0; 2]; 15];
            for (elem, old) in table.iter_mut().zip(src.iter()) {
                let u = (old[0] - n_min + 1.0).ln();
                let f = PolDep::new(old[1], old[2]).interp(sv1);
                elem[0] = u;
                elem[1] = (-1.0 * (1.0 - f / f_max).ln()).ln();
            }

            // println!("[{}]: n_min = {}, table = {:?}", i, n_min, table);

            let frac = (-1.0 * (1.0 - frac).ln()).ln();
            let u = pwmci::Interpolant::new(&table)
                .extrapolate(true)
                .invert(frac)
                .unwrap();

            u * w
        })
        .sum();

    // what it should be at this point
    let (n_min, _n_max) = TotalRate::new(a, eta).sum_limits();
    let n = (n_min as f64) + u.exp_m1();
    (n as i32) + 1
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::super::DoubleDiffPartialRate;
    use super::*;

    #[test]
    fn inversion() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let sv1 = 1.0;
        let pts = [
            (2.5, 0.2),
            (5.0, 0.2),
            (11.0, 0.2),
            (8.0, 0.05),
        ];

        for (a, eta) in &pts {
            let (n_min, n_max) = TotalRate::new(*a, *eta).sum_limits();
            let harmonics: Vec<_> = ((n_min-1)..=(n_min+16)).collect();

            let rates: Vec<_> = harmonics.iter()
                .map(|n| {
                    let pr = DoubleDiffPartialRate::new(*n, *a, *eta).integrate();
                    0.5 * ((pr.re + pr.im) + sv1 * (pr.re - pr.im))
                })
                .collect();

            let total: f64 = {
                let [a, b] = interpolate(*a, *eta);
                0.5 * ((a + b) + sv1 * (a - b))
            };

            println!("a = {}, eta = {}, n = {}..{}:", a, eta, n_min, n_max);

            let mut counts = [0.0; 16];

            for _ in 0..1_000_000 {
                let frac: f64 = rng.gen();
                let n = invert(*a, *eta, 0.0, frac);
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
                // assert!(error < 0.5);
            };
        }
    }
}