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

    let (delta, shift) = index.iter()
        .zip(weight.iter())
        .map(|(i, w)| {
            let src = &TABLE[*i];
            let mut table = [[0.0; 2]; 17];

            // Where extrapolated CDF crosses y = 0.
            let n_zero = (src[0][0] * src[1][1] - src[1][0] * src[0][1]) / (src[1][1] - src[0][1]);
            // let n_zero = src[0][0] - 1.0;
            let shift = n_zero - src[0][0];

            for i in 0..16 {
                table[i+1][0] = (src[i][0] - n_zero) / (src[15][0] - n_zero);
                table[i+1][1] = src[i][1];
            }

            let delta = pwmci::Interpolant::new(&table).invert(frac).unwrap();

            (delta * w, shift * w)
        })
        .reduce(|a, b| (a.0 + b.0, a.1 + b.1))
        .unwrap();

    let (n_min, n_max) = super::sum_limits(a, eta);
    let n_zero = (n_min as f64) + shift;
    let n = (n_zero + ((n_max as f64) - n_zero) * delta).ceil() as i32;
    if n < n_min {
        // most probable harmonic, ish
        let fallback = n_min + a.powf(1.1).round() as i32;
        eprintln!(
            "pair_creation::lp::sample unable to obtain a harmonic order in [{}, {}] at a = {:.3e}, eta = {:.3e}, falling back to {}...",
            n_min, n_max, a, eta, fallback
        );
        fallback
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
            (2.5, 0.2),
            (5.0, 0.2),
            (11.0, 0.2),
            (8.0, 0.05),
        ];

        for (a, eta) in &pts {
            let (n_min, n_max) = lp::sum_limits(*a, *eta);
            let harmonics: Vec<_> = ((n_min-1)..=(n_min+16)).collect();

            let rates: Vec<_> = harmonics.iter().map(|n| lp::partial_rate(*n, *a, *eta)).collect();
            let total: f64 = interpolate(*a, *eta);

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
                if target == 0.0 && *count == 0.0 {
                    continue;
                }
                let error = (target - count) / target;
                println!("\t{:>4} {:>9.3e} {:>9.3e} [{:>6.2}%]", n, target, count, 100.0 * error);
                assert!(error < 0.5);
            };
        }
    }
}