//! Tables for the total rate and CDF as a function of harmonic order

use crate::pwmci;

pub(super) mod mid_range;

/// Tabulated total rate for 0.05 < a < 10
mod total;

/// Tabulate CDF as a function of harmonic order for 0.05 < a < 10
#[allow(unused)]
mod cdf;

pub fn contains(a: f64, eta: f64) -> bool {
    a.ln() >= total::LN_MIN_A && a < 20.0 && eta.ln() >= total::LN_MIN_ETA && eta < 2.0
}

#[allow(unused_parens)]
pub fn interpolate(a: f64, eta: f64) -> f64 {
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
            (1.0 - da) * (1.0 - de) * TABLE[ie][ia]
            + da * (1.0 - de) * TABLE[ie][ia+1]
            + (1.0 - da) * de * TABLE[ie+1][ia]
            + da * de * TABLE[ie+1][ia+1]
        );
        f.exp()
    } else {
        // linear interpolation of: 1 / log y against x, best for f(x) exp(-1/x)
        let a_min = (LN_MIN_A + (ia as f64) * LN_A_STEP).exp();
        let a_max = (LN_MIN_A + ((ia+1) as f64) * LN_A_STEP).exp();
        let eta_min = (LN_MIN_ETA + (ie as f64) * LN_ETA_STEP).exp();
        let eta_max = (LN_MIN_ETA + ((ie+1) as f64) * LN_ETA_STEP).exp();
        let da = (a - a_min) / (a_max - a_min);
        let de = (eta - eta_min) / (eta_max - eta_min);
        let f = (
            TABLE[ie][ia] * a_min * eta_min * (1.0 - da) * (1.0 - de)
            + TABLE[ie][ia+1] * a_max * eta_min * da * (1.0 - de)
            + TABLE[ie+1][ia] * a_min * eta_max * (1.0 - da) * de
            + TABLE[ie+1][ia+1] * a_max * eta_max * da * de
        ) / (a * eta);
        f.exp()
    }
}

pub fn invert(a: f64, eta: f64, frac: f64) -> i32 {
    use cdf::*;

    let ia = ((a.ln() - MIN[0]) / STEP[0]) as usize;
    let ie = ((eta.ln() - MIN[1]) / STEP[1]) as usize;
    let da = (a.ln() - MIN[0]) / STEP[0] - (ia as f64);
    let de = (eta.ln() - MIN[1]) / STEP[1] - (ie as f64);

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

    let n_alt: f64 = index.iter()
        .zip(weight.iter())
        .map(|(i, w)| {
            let table = &TABLE[*i];
            let n = if frac <= table[0][1] {
                table[0][0] - 0.1
            } else {
                pwmci::invert(frac, table).unwrap().0
            };
            //println!("\tgot n = {}, d = {}", n, n - table[0][0]);
            //(n - table[0][0]) * w
            n * w
        })
        .sum();

    let n = n_alt.ceil() as i32;
    let (n_min, n_max) = super::sum_limits(a, eta);
    if n < n_min {
        eprintln!("pair_creation::generate failed to sample a harmonic order (a = {:.3e}, eta = {:.3e}, {} <= n < {}), falling back to {}.", a, eta, n_min, n_max, n_min);
        n_min
    } else {
        n
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::*;

    #[test]
    fn inversion() {
        use crate::pair_creation::lp;
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let pts = [
            (2.5, 0.1),
        ];

        for (a, eta) in &pts {
            let (n_min, n_max) = lp::sum_limits(*a, *eta);
            let harmonics: Vec<_> = (n_min..=n_max).collect();
            let rates: Vec<_> = harmonics.iter().map(|n| lp::partial_rate(*n, *a, *eta)).collect();
            let total: f64 = rates.iter().sum();

            println!("a = {}, eta = {}, n = {}..{}:", a, eta, n_min, n_max);

            let mut counts = [0.0; 16];
            for _ in 0..1_000_000 {
                let n = invert(*a, *eta, rng.gen());
                for i in 0..harmonics.len() {
                    if n == harmonics[i] as i32 && i < 16 {
                        counts[i] += 1.0 / 1_000_000.0;
                    }
                }
            }

            for ((n, rate), count) in harmonics.iter().zip(rates.iter()).zip(counts.iter()) {
                let target = rate / total;
                // if target < 1.0e-6 {
                //     continue;
                // }
                let error = (target - count) / target;
                println!("\t{:>4} {:>9.3e} {:>9.3e} [{:>6.2}%]", n, target, count, 100.0 * error);
            };
        }
    }
}