//! Nonlinear pair creation, gamma -> e- + e+, in a background field

use std::f64::consts;
use crate::constants::*;
//use super::pwmci;

mod tables;

/// Auxiliary function used in calculation of LCFA pair creation
/// rate, defined
/// T(chi) = 1/(6 sqrt3 pi chi) int_1^\infty (8u + 1)/\[u^(3/2) sqrt(u-1)\] K_{2/3}\[8u/(3chi)\] du
fn auxiliary_t(chi: f64) -> f64 {
    use tables::*;
    if chi <= 0.01 {
        // if chi < 5e-3, T(chi) < 1e-117, so ignore
        // 3.0 * 3.0f64.sqrt() / (8.0 * consts::SQRT_2) * (-4.0 / (3.0 * chi)).exp()
        0.0
    } else if chi < 1.0 {
        // use exp(-f/chi) fit
        let i = ((chi.ln() - LN_T_CHI_TABLE[0][0]) / DELTA_LN_CHI) as usize;
        let dx = (chi - LN_T_CHI_TABLE[i][0].exp()) / (LN_T_CHI_TABLE[i+1][0].exp() - LN_T_CHI_TABLE[i][0].exp());
        let tmp = (1.0 - dx) / LN_T_CHI_TABLE[i][1] + dx / LN_T_CHI_TABLE[i+1][1];
        (1.0 / tmp).exp()
    } else if chi < 100.0 {
        // use power-law fit
        let i = ((chi.ln() - LN_T_CHI_TABLE[0][0]) / DELTA_LN_CHI) as usize;
        let dx = (chi.ln() - LN_T_CHI_TABLE[i][0]) / DELTA_LN_CHI;
        ((1.0 - dx) * LN_T_CHI_TABLE[i][1] + dx * LN_T_CHI_TABLE[i+1][1]).exp()
    } else {
        // use asymptotic expression, which is accurate to better than 0.3%
        // for chi > 100:
        //   T(x) = [C - C_1 x^(-2/3)] x^(-1/3)
        // where C = 5 Gamma(5/6) (2/3)^(1/3) / [14 Gamma(7/6)] and C_1 = 2/3
        (0.37961230854357103776 - 2.0 * chi.powf(-2.0/3.0) / 3.0) * chi.powf(-1.0/3.0)
    }
}

/// Returns the nonlinear Breit-Wheeler rate, per unit time (in seconds)
pub fn rate(chi: f64, gamma: f64) -> f64 {
    ALPHA_FINE * chi * auxiliary_t(chi) / (COMPTON_TIME * gamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcfa_rate() {
        let max_error = 1.0e-2;

        let pts = [
            (0.042, 6.077538994929929904e-29),
            (0.105, 2.1082097875655204834e-12),
            (0.42,  0.00037796132366581330636),
            (1.05,  0.015977478443872017101),
            (4.2,   0.08917816786414408900),
            (12.0,  0.10884579479913803705),
            (42.0,  0.09266735324318656466),
        ];

        for (chi, target) in &pts {
            let result = auxiliary_t(*chi);
            let error = (result - target).abs() / target;
            //println!("chi = {:.3e}, t(chi) = {:.6e}, error = {:.3e}", chi, result, error);
            assert!(error < max_error);
        }
    }
}