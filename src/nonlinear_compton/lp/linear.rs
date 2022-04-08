//! Rates and spectra for NLC in the linear regime, a << 1

/// Returns the total rate of nonlinear Compton scattering,
/// calculated under the assumption that a << 1.
#[inline]
pub fn rate(a: f64, eta: f64) -> f64 {
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    //let eta4 = eta3 * eta;
    let first = if eta > 1.0e-3 {
        (a * a / 8.0) * (
            0.5
            + 4.0 / eta
            - 0.5 / ((1.0 + 2.0 * eta) * (1.0 + 2.0 * eta))
            + (1.0 - 2.0 / eta - 2.0 / eta2) * (1.0 + 2.0 * eta).ln()
        )
    } else {
        (a * a / 8.0) * (
            8.0 * eta / 3.0 - 16.0 * eta2 / 3.0 + 208.0 * eta3 / 15.0
        )
    };
    // let first_corr = -(a.powi(4) / 64.0) * (
    //     6.0
    //     - 10.0 / eta
    //     - 18.0 / eta2
    //     - 12.0 / eta3
    //     + 13.0 / (1.0 + 2.0 * eta)
    //     - 1.0 / ((1.0 + 2.0 * eta) * (1.0 + 2.0 * eta))
    //     - (
    //         5.0 / eta - 12.0 / eta2 - 15.0 / eta3 - 6.0 / eta4
    //     ) * (1.0 + 2.0 * eta).ln()
    // );
    // let second = (a.powi(4) / 64.0) * (
    //     6.0
    //     - 2.0 / eta
    //     - 18.0 / eta2
    //     - 6.0 / eta3
    //     + 1.0 / (1.0 + 4.0 * eta)
    //     - 1.0 / ((1.0 + 4.0 * eta) * (1.0 + 4.0 * eta))
    //     - 0.5 * (
    //         5.0 / eta - 15.0 / eta2 - 15.0 / eta3 - 3.0 / eta4
    //     ) * (1.0 + 4.0 * eta).ln()
    // );
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_approx_accuracy() {
        let eta = 0.1;
        for a in [0.005, 0.01, 0.02, 0.03, 0.04] {
            let nmax = (5.0 * (1.0 + 2.0 * a * a)) as i32;
            let target = (1..=nmax).map(|n| crate::nonlinear_compton::lp::partial_rate(n, a, eta).0).sum::<f64>();
            let value = rate(a, eta);
            let error = (target - value).abs() / target;
            println!(
                "a = {:>9.3e}, eta = {:>9.3e}: target = {:>12.6e}, value = {:>12.6e}, diff = {:.3e}",
                a, eta, target, value, error,
            );
            assert!(error < 1.0e-3);
        }
    }
}