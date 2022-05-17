//! Tables for the total rate and CDF as a function of harmonic order

pub(super) mod mid_range;

/// Tabulated total rate for 0.05 < a < 10
mod total;

/// Tabulate CDF as a function of harmonic order for 0.05 < a < 10
#[allow(unused)]
mod cdf;

pub fn contains(a: f64, eta: f64) -> bool {
    a.ln() >= total::LN_MIN_A && a < 10.0 && eta.ln() >= total::LN_MIN_ETA && eta < 2.0
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