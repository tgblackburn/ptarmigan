//! For 0.05 < a < 1.0 and 0.1 < eta < 2

mod total;

pub fn contains(a: f64, eta: f64) -> bool {
    let ln_a = a.ln();
    let ln_eta_prime = (eta / (1.0 + 0.5 * a * a)).ln();
    ln_a > total::LN_MIN_A
        && a < 1.0
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

pub fn interpolate(a: f64, eta: f64) -> f64 {
    use total::*;
    let ia = ((a.ln() - LN_MIN_A) / LN_A_STEP) as usize;
    let da = (a.ln() - LN_MIN_A) / LN_A_STEP - (ia as f64);
    let eta_prime = eta / (1.0 + 0.5 * a * a);
    //println!("da = {}", da);
    let f = (1.0 - da) * interpolate_in_eta_prime(eta_prime, ia) + da * interpolate_in_eta_prime(eta_prime, ia+1);
    f.exp()
}
