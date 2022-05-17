//! For 0.05 < a < 1.0 and 0.1 < eta < 2

use crate::pwmci;

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

fn interpolate_in_eta_prime(eta_prime: f64, ia: usize) -> f64 {
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
        TABLE[ie+2][ia]
    );

    let mid = (
        2.0 / (1.0 + ((ie+1) as f64) / ETA_PRIME_DENSITY),
        TABLE[ie+1][ia]
    );

    let upper = (
        2.0 / (1.0 + (ie as f64) / ETA_PRIME_DENSITY),
        TABLE[ie][ia]
    );

    // println!("lower = {:?}", lower);
    // println!("mid = {:?}", mid);
    // println!("upper = {:?}", upper);

    // fit a power law y = alpha (e - e_0)^m + b
    let m = ((upper.1 - lower.1) / (mid.1 - lower.1)).ln() / ((upper.0 - lower.0) / (mid.0 - lower.0)).ln();
    let alpha = (mid.1 - lower.1) / (mid.0 - lower.0).powf(m);
    //println!("alpha = {}, m = {}, returning {}", alpha, m, alpha * (eta_prime - lower.0).powf(m) + lower.1);

    alpha * (eta_prime - lower.0).powf(m) + lower.1
}

#[allow(unused)]
fn interpolate_in_eta_prime_envelope(eta_prime: f64, ia: usize) -> f64 {
    use total::*;

    // eta' = 2 / (1 + i/r)
    let ie = (ETA_PRIME_DENSITY * (LN_MAX_ETA_PRIME.exp() / eta_prime - 1.0)) as usize;

    // fit g(x) exp(-C/x) between harmonics
    // remember increasing ie reduces eta
    let (upper, lower) = {
        let index = ie / (ETA_PRIME_DENSITY as usize);
        let index = (ETA_PRIME_DENSITY as usize) * index; // index of lower bdy
        (
            (2.0 / (1.0 + (index as f64) / ETA_PRIME_DENSITY), TABLE[index][ia]),
            (2.0 / (1.0 + ((index + 4) as f64) / ETA_PRIME_DENSITY), TABLE[index+4][ia]),
        )
    };

    // need something like g(x) exp(-C/x) + delta(x), where delta = 0 at bdys
    let envelope = {
        let weight = (eta_prime - lower.0) / (upper.0 - lower.0);
        (lower.1 * lower.0 * (1.0 - weight) + upper.1 * upper.0 * weight) / eta_prime
    };

    envelope
}

pub fn interpolate(a: f64, eta: f64) -> f64 {
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

    f.exp()
}

pub fn invert(a: f64, eta: f64, frac: f64) -> i32 {
    use cdf::*;
    let ia = ((a.ln() - LN_MIN_A) / LN_A_STEP) as usize;
    let da = (a.ln() - LN_MIN_A) / LN_A_STEP - (ia as f64);

    let eta_prime = eta / (1.0 + 0.5 * a * a);
    // eta' = 2 / (1 + i/r)
    let ie = (total::ETA_PRIME_DENSITY * (LN_MAX_ETA_PRIME.exp() / eta_prime - 1.0)) as usize;

    // fix ie if it's next to a harmonic boundary
    let ie = if ie % (total::ETA_PRIME_DENSITY as usize) == (total::ETA_PRIME_DENSITY as usize) - 1 {
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

    let n_alt: f64 = index.iter()
        .zip(weight.iter())
        .map(|(i, w)| {
            let table = &TABLE[*i];
            let n = if frac <= table[0][1] {
                table[0][0] - 0.1
            } else {
                pwmci::invert(frac, table).unwrap().0
            };
            n * w
        })
        .sum();

    n_alt.ceil() as i32
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
            (0.075, 0.6),
            (0.075, 0.13),
            (0.75, 0.6),
            (0.75, 0.13),
            (0.25, 0.34), // across bdy
            (0.25, 0.35),
        ];

        for (a, eta) in &pts {
            let (n_min, n_max) = lp::sum_limits(*a, *eta);
            let harmonics: Vec<_> = (n_min..=n_max).collect();
            let rates: Vec<_> = harmonics.iter().map(|n| lp::partial_rate(*n, *a, *eta)).collect();
            let total: f64 = rates.iter().sum();

            println!("a = {}, eta = {}:", a, eta);

            let mut counts = [0.0; 16];
            for _ in 0..1_000_000 {
                let n = invert(*a, *eta, rng.gen());
                for i in 0..harmonics.len() {
                    if n == harmonics[i] as i32 {
                        counts[i] += 1.0 / 1_000_000.0;
                    }
                }
            }

            for ((n, rate), count) in harmonics.iter().zip(rates.iter()).zip(counts.iter()) {
                let target = rate / total;
                if target < 1.0e-6 {
                    continue;
                }
                let error = (target - count) / target;
                println!("\t{:>4} {:>9.3e} {:>9.3e} [{:>6.2}%]", n, target, count, 100.0 * error);
            };
        }
    }
}