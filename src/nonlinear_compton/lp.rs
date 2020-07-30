//! Nonlinear Compton scattering in a linearly polarized background

/// Calculates the auxiliary functions `Γ_0(m, a, b)`, `Γ_1(m, a, b)`
/// and `Γ_2(m, a, b)`, used in the linear Compton rate.
#[allow(unused)]
fn triple_aux_gamma_alloc(m: i32, alpha: f64, beta: f64) -> [f64; 3] {
    // Functions defined by:
    // Γ_0(m, a, b) = Σ_{r=-∞}^∞ J_{m+2r}(a) J_r(b),
    // Γ_1(m, a, b) = (1/2) Σ_{r=-∞}^∞ [J_{m+2r-1}(a) + J_{m+2r+1}(a)] J_r(b)
    // Γ_2(m, a, b) = (1/4) Σ_{r=-∞}^∞ [J_{m+2r-2}(a) + 2 J_{m+2r}(a) + J_{m+2r+2}(a)] J_r(b)

    // Sum runs from r = -rmax to rmax, backwards
    let rmax = (2 * m).max(10);

    let mut j_beta: Vec<f64> = Vec::with_capacity(rmax as usize);
    let mut norm_beta = 0.0;
    j_beta.push(0.0f64);
    j_beta.push(1.0f64);
    for r in (0..=rmax).rev().step_by(2) {
        let len = j_beta.len();
        let mut tmp = [0.0, 0.0, j_beta[len-1], j_beta[len-2]];
        tmp[1] = ((2 * (r + 2)) as f64) * tmp[2] / beta - tmp[3];
        tmp[0] = ((2 * (r + 1)) as f64) * tmp[1] / beta - tmp[2];
        norm_beta += tmp[0];
        j_beta.push(tmp[1]);
        j_beta.push(tmp[0]);
    }

    let norm_beta = 2.0 * norm_beta - j_beta.last().unwrap();
    j_beta.iter_mut().for_each(|v| *v = *v / norm_beta);
    j_beta.reverse();
    let j_beta = j_beta;

    let m_even = (m.rem_euclid(2) == 0) as usize;
    let mut j_alpha: Vec<f64> = Vec::with_capacity((m + 2 * rmax) as usize);
    let mut norm_alpha = 0.0;
    j_alpha.push(0.0f64);
    j_alpha.push(1.0f64);
    for r in (0..=(m + 2 * rmax)).rev().step_by(2) {
        let len = j_alpha.len();
        let mut tmp = [0.0, 0.0, j_alpha[len-1], j_alpha[len-2]];
        tmp[1] = ((2 * (r + 2)) as f64) * tmp[2] / alpha - tmp[3];
        tmp[0] = ((2 * (r + 1)) as f64) * tmp[1] / alpha - tmp[2];
        norm_alpha += tmp[1-m_even];
        j_alpha.push(tmp[1]);
        j_alpha.push(tmp[0]);
    }

    if m_even == 0 {
        let len = j_alpha.len();
        let tmp = 2.0 * j_alpha[len-1] / alpha - j_alpha[len-2];
        norm_alpha += tmp;
        j_alpha.push(tmp);
    }

    let norm_alpha = 2.0 * norm_alpha - j_alpha.last().unwrap();
    j_alpha.iter_mut().for_each(|v| *v = *v / norm_alpha);
    j_alpha.reverse();
    let j_alpha = j_alpha;

    let mut result = 0.0;
    for r in 1..rmax {
        let a = j_alpha[(m + 2 * r) as usize];
        let bindex = m - 2 * r;
        let (sign, b) = if bindex >= 0 {
            ((-1.0f64).powi(r), j_alpha[bindex as usize])
        } else {
            ((-1.0f64).powi(r - bindex), j_alpha[-bindex as usize])
        };
        let c = j_beta[r as usize];
        result += (a + sign * b) * c;
    }
    // add comtribution from r = 0
    let result = result + j_alpha[m as usize] * j_beta[0];

    //println!("J(0, alpha) = {:.12e}, J(0, beta) = {:.12e}", j_alpha.first().unwrap(), j_beta.first().unwrap());
    [result, 0.0, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aux_gammas() {
        let value = triple_aux_gamma_alloc(2, 0.1, 0.2);
        let target = -0.0980095;
        println!("aux_gamma_0(2, 0.1, 0.2) = {:.6e}, target = {:.6e}", value[0], target);
    }
}