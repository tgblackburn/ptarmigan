//! Nonlinear Compton scattering in a linearly polarized background

/// Calculates the auxiliary function
/// `Γ_0(m, a, b) = Σ_{r=-∞}^∞ J_{m+2r}(a) J_r(b)`
fn aux_gamma_0(m: i32, alpha: f64, beta: f64) -> f64 {
    // Sum runs from r = -rmax to rmax, backwards
    let rmax = (2 * m).max(10);
    let m_even = (m.rem_euclid(2) == 0) as usize;

    // These hold the unscaled values of J_r(alpha) and J_r(beta)
    let mut j_alpha = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut j_beta = [1.0, 0.0, 0.0, 0.0];

    // And the scaling factors
    let mut norm_alpha = 0.0;
    let mut norm_beta = 0.0;
    let mut result = 0.0;

    for r in (0..=rmax).rev().step_by(2) {
        j_beta[3] = j_beta[1];
        j_beta[2] = j_beta[0];
        j_beta[1] = ((2 * (r + 2)) as f64) * j_beta[2] / beta - j_beta[3];
        j_beta[0] = ((2 * (r + 1)) as f64) * j_beta[1] / beta - j_beta[2];
        norm_beta += j_beta[0];
        // Now j_beta holds J_r(beta), J_{r+1}(beta), etc. 

        j_alpha[5] = j_alpha[1];
        j_alpha[4] = j_alpha[0];
        j_alpha[3] = ((2 * (m + 2 * r + 4)) as f64) * j_alpha[4] / alpha - j_alpha[5];
        j_alpha[2] = ((2 * (m + 2 * r + 3)) as f64) * j_alpha[3] / alpha - j_alpha[4];
        j_alpha[1] = ((2 * (m + 2 * r + 2)) as f64) * j_alpha[2] / alpha - j_alpha[3];
        j_alpha[0] = ((2 * (m + 2 * r + 1)) as f64) * j_alpha[1] / alpha - j_alpha[2];
        norm_alpha += j_alpha[m_even] + j_alpha[m_even + 2];
        // Now j_alpha holds J_{m+2r}(alpha), J_{m+2r+1}(alpha), etc.

        println!("beta index = {}, alpha index = {} {}", r, m + 2 * r, m + 2 * r + 2);

        result += j_alpha[0] * j_beta[0] + j_alpha[2] * j_beta[1];
    }

    let partial = result;

    // After first loop, first entries are J_m(alpha) and J_0(beta)
    let norm_beta = 2.0 * norm_beta - j_beta[0];
    let test = j_beta[0] / norm_beta;

    // Need to loop down to m + 2r = 0 to get norm_alpha
    for r in (-(m+3)/2..=0).rev().step_by(2).skip(1) {
        j_beta[3] = j_beta[1];
        j_beta[2] = j_beta[0];
        j_beta[1] = ((2 * (r + 2)) as f64) * j_beta[2] / beta - j_beta[3];
        j_beta[0] = ((2 * (r + 1)) as f64) * j_beta[1] / beta - j_beta[2];

        j_alpha[5] = j_alpha[1];
        j_alpha[4] = j_alpha[0];
        j_alpha[3] = ((2 * (m + 2 * r + 4)) as f64) * j_alpha[4] / alpha - j_alpha[5];
        j_alpha[2] = ((2 * (m + 2 * r + 3)) as f64) * j_alpha[3] / alpha - j_alpha[4];
        j_alpha[1] = ((2 * (m + 2 * r + 2)) as f64) * j_alpha[2] / alpha - j_alpha[3];
        j_alpha[0] = ((2 * (m + 2 * r + 1)) as f64) * j_alpha[1] / alpha - j_alpha[2];
        norm_alpha += j_alpha[m_even] + j_alpha[m_even + 2];
        // Now j_alpha holds J_{m+2r}(alpha), J_{m+2r+1}(alpha), etc.

        println!("2nd: beta index = {}, alpha index = {} {}", r, m + 2 * r, m + 2 * r + 2);

        result += j_alpha[0] * j_beta[0] + j_alpha[2] * j_beta[1];
    }

    let index = m + 2 * (-(m+3)/2);
    println!(" got to {}", index);
    let (norm_alpha, test_alpha) = match index {
        -3 => {norm_alpha -= j_alpha[1]; (2.0 * norm_alpha - j_alpha[3], j_alpha[3])},
        -2 => {norm_alpha -= j_alpha[0]; (2.0 * norm_alpha - j_alpha[2], j_alpha[2])},
        -1 => (2.0 * norm_alpha - j_alpha[1], j_alpha[1]),
        0 => (2.0 * norm_alpha - j_alpha[0], j_alpha[0]),
        _ => panic!(),
    };
    let test_alpha = test_alpha / norm_alpha;

    println!("J(0, alpha) = {:.12e}, J(0, beta) = {:.12e}, partial = {:.6e}", test_alpha, test, partial / (norm_alpha * norm_beta));

    result / (norm_alpha * norm_beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aux_gammas() {
        let value = aux_gamma_0(2, 0.1, 0.2);
        let target = 0.1;
        println!("aux_gamma_0(2, 0.1, 0.2) = {:.3e}, target = {:.3e}", value, target);
    }
}