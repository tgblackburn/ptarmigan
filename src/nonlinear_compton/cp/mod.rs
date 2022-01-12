//! Rates and spectra for circularly polarized backgrounds
use crate::special_functions::*;
use super::{GAUSS_32_NODES, GAUSS_32_WEIGHTS};

// Lookup tables
mod total;

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Compton rates. Equivalent to calling
/// ```
/// let nmax = (10.0 * (1.0 + a.powi(3))) as i32;
/// let rate = (1..=nmax).map(|n| integrated_spectrum(n, a, eta)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
pub fn sum_integrated_spectra(a: f64, eta: f64) -> f64 {
    let f = if a < total::LOW_A_LIMIT && eta < total::LOW_ETA_LIMIT {
        // linear Thomson
        2.0 * a  * a * eta / 3.0
    } else if a < total::LOW_A_LIMIT {
        // linear Compton rate for arbitrary eta
        a * a * (2.0 + 8.0 * eta + 9.0 * eta * eta + eta * eta * eta) / (2.0 * eta * (1.0 + 2.0 * eta).powi(2))
            - a * a * (2.0 + 2.0 * eta - eta * eta) * (1.0 + 2.0 * eta).ln() / (4.0 * eta * eta)
    } else if eta < total::LOW_ETA_LIMIT {
        eta *  total::LOW_ETA_RATE_TABLE.at(a).unwrap_or_else(|| {
            panic!("NLC rate lookup out of bounds (low eta table): a = {:.3e}, eta = {:.3e}", a, eta);
        })
    } else {
        total::RATE_TABLE.at(a, eta).unwrap_or_else(|| {
            panic!("NLC rate lookup out of bounds: a = {:.3e}, eta = {:.3e}", a, eta);
        })
    };
    f
}

/// Integrates the important part of the nonlinear Compton
/// differential rate, f, which gives either
///   `dP/(dv dϕ) = ⍺ f(n, a, η, v) / η`
/// or
///   `dP/(dv dt) = ⍺ m f(n, a, η, v) / γ`
/// over the domain `0 < v < 1`
pub fn integrated_spectrum(n: i32, a: f64, eta: f64) -> f64 {
    let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
    // approx harmonic index when sigma / mu < 0.25
    let n_switch = (32.3 * (1.0 + 0.476 * a.powf(1.56))) as i32;
    let integral: f64 = if sn < 2.0 || n < n_switch {
        let vmid = (1.0 + sn) / (2.0 + sn);
        let lower: f64 = GAUSS_32_NODES.iter()
            .map(|x| 0.5 * vmid * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(v, w)| 0.5 * vmid * w * spectrum(n, a, eta, v))
            .sum();
        // integrate from v = vmid to v = 1
        let upper: f64 = GAUSS_32_NODES.iter()
            .map(|x| vmid + 0.5 * (1.0 - vmid) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(v, w)| 0.5 * (1.0 - vmid) * w * spectrum(n, a, eta, v))
            .sum();
        lower + upper
    } else {
        // If peak of spectrum v >= 3/4 and peak is sufficiently narrow,
        // switch to integrating over u = v / (1 - v) instead.
        // Partition the range into 0 < u < 1 + sn, and
        // 1 + sn < u < 3(1 + sn) and integrate each separately,
        // to ensure we capture the peak.
        let lower: f64 = GAUSS_32_NODES.iter()
            .map(|x| 0.5 * (1.0 + sn) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(u, w)|
                0.5 * (1.0 + sn) * w * spectrum(n, a, eta, u / (1.0 + u)) / (1.0 + u).powi(2)
            )
            .sum();
        let upper: f64 = GAUSS_32_NODES.iter()
            .map(|x| (1.0 + sn) + 0.5 * 2.0 * (1.0 + sn) * (x + 1.0))
            .zip(GAUSS_32_WEIGHTS.iter())
            .map(|(u, w)|
                 0.5 * 2.0 * (1.0 + sn) * w * spectrum(n, a, eta, u / (1.0 + u)) / (1.0 + u).powi(2)
            )
            .sum();
        lower + upper
    };
    integral
}

/// Equivalent to `integrated_spectrum(n, a, eta, v) / eta` for eta -> 0.
#[allow(unused)]
fn integrated_spectrum_low_eta(n: i32, a: f64) -> f64 {
    // integrate from v = 0 to v = 1/2
    let lower: f64 = GAUSS_32_NODES.iter()
        .map(|x| 0.25 * (x + 1.0))
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(v, w)| 0.25 * w * spectrum_low_eta(n, a, v))
        .sum();

    // integrate from v = 1/2 to v = 1
    let upper: f64 = GAUSS_32_NODES.iter()
        .map(|x| 0.25 * (x + 3.0))
        .zip(GAUSS_32_WEIGHTS.iter())
        .map(|(v, w)| 0.25 * w * spectrum_low_eta(n, a, v))
        .sum();

    lower + upper
}

/// Evaluates the important part of the nonlinear Compton
/// differential rate, f,
/// either
///   `dP/(dv dϕ) = ⍺ f(n, a, η, v) / η`
/// or
///   `dP/(dv dt) = ⍺ m f(n, a, η, v) / γ`
/// where `0 < v < 1`
pub fn spectrum(n: i32, a: f64, eta: f64, v: f64) -> f64 {
    if v < 0.0 || v >= 1.0 {
        return 0.0;
    }

    let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
    let smax = sn / (1.0 + sn);
    let vsmax = v * smax;
    let z = (
        ((4 * n * n) as f64)
        * (a * a / (1.0 + a * a))
        * (vsmax / (sn * (1.0 - vsmax)))
        * (1.0 - vsmax / (sn * (1.0 - vsmax)))
    ).sqrt();
    let (j_nm1, j_n, j_np1) = z.j_pm(n);

    -smax * (
        j_n.powi(2)
        + 0.5 * a * a * (1.0 + 0.5 * vsmax.powi(2) / (1.0 - vsmax))
        * (2.0 * j_n.powi(2) - j_np1.powi(2) - j_nm1.powi(2))
    )
}

/// Equivalent to `spectrum(n, a, eta, v) / eta` for eta -> 0.
#[allow(unused)]
fn spectrum_low_eta(n: i32, a: f64, v: f64) -> f64 {
    if v < 0.0 || v >= 1.0 {
        return 0.0;
    }

    let z = (4.0 * a * a * ((n * n) as f64) * v * (1.0-v) / (1.0 + a * a)).sqrt();
    let (j_nm1, j_n, j_np1) = z.j_pm(n);
    (n as f64) * (a * a * j_nm1.powi(2) - 2.0 * (1.0 + a * a) * j_n.powi(2) + a * a * j_np1.powi(2)) / (1.0 + a * a)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use super::*;

    #[test]
    #[ignore]
    fn create_rate_table() {
        use std::f64::consts;

        const N_COLS: usize = 55;
        const N_ROWS: usize = 70;
        let mut table = [[0.0; N_COLS]; N_ROWS];
        for i in 0..N_ROWS {
            // eta = eta_min * 10^(i/20)
            let eta = total::LOW_ETA_LIMIT * 10.0f64.powf((i as f64) / 20.0);
            for j in 0..N_COLS {
                // a = a_min * 10^(j/20)
                let a = total::LOW_A_LIMIT * 10.0f64.powf((j as f64) / 20.0);
                let nmax = (10.0 * (1.0 + a.powi(3))) as i32;
                table[i][j] = (1..=nmax).map(|n| integrated_spectrum(n, a, eta)).sum();
                println!("NLC: eta = {:.3e}, a = {:.3e}, ln(rate) = {:.6e}", eta, a, table[i][j].ln());
            }
        }

        let mut file = File::create("output/nlc_rate_table.txt").unwrap();
        writeln!(file, "pub const RATE_TABLE: Table2D = Table2D {{").unwrap();
        writeln!(file, "\tlog_scaled: true,").unwrap();
        writeln!(file, "\tmin: [{:.12e}, {:.12e}],", total::LOW_A_LIMIT.ln(), total::LOW_ETA_LIMIT.ln()).unwrap();
        writeln!(file, "\tstep: [{:.12e}, {:.12e}],", consts::LN_10 / 20.0, consts::LN_10 / 20.0).unwrap();
        writeln!(file, "\tdata: [").unwrap();

        for row in table.iter() {
            write!(file, "\t\t[{:>18.12e}", row.first().unwrap().ln()).unwrap();
            for val in row.iter().skip(1) {
                write!(file, ", {:>18.12e}", val.ln()).unwrap();
            }
            writeln!(file, "],").unwrap();
        }

        writeln!(file, "\t],").unwrap();
        writeln!(file, "}};").unwrap();

        let mut table = [0.0; N_COLS];
        for j in 0..N_COLS {
            let a = total::LOW_A_LIMIT * 10.0f64.powf((j as f64) / 20.0);
            let nmax = (10.0 * (1.0 + a.powi(3))) as i32;
            table[j] = (1..=nmax).map(|n| integrated_spectrum_low_eta(n, a)).sum();
            println!("NLC: eta -> 0, a = {:.3e}, ln(rate) = {:.6e}", a, table[j].ln());
        }

        let mut file = File::create("output/nlc_low_eta_rate_table.txt").unwrap();
        writeln!(file, "pub const LOW_ETA_RATE_TABLE: Table1D = Table1D {{").unwrap();
        writeln!(file, "\tlog_scaled: true,").unwrap();
        writeln!(file, "\tmin: {:.12e},", total::LOW_A_LIMIT.ln()).unwrap();
        writeln!(file, "\tstep: {:.12e},", consts::LN_10 / 20.0).unwrap();
        writeln!(file, "\tdata: [").unwrap();

        write!(file, "\t\t{:>18.12e}", table.first().unwrap().ln()).unwrap();
        for val in table.iter().skip(1) {
            write!(file, ", {:>18.12e}", val.ln()).unwrap();
        }

        writeln!(file, "\n\t],").unwrap();
        writeln!(file, "}};").unwrap();
    }

    #[test]
    fn partial_rate() {
        let max_error = 1.0e-6;

        // n, a, eta, target
        let pts = [
            (2,    0.5, 0.15,   2.748486539e-3),
            (10,   1.0, 0.2,    1.984654425e-4),
            (80,   2.0, 0.2,    3.751480198e-6),
            (160,  2.0, 0.2,    6.842944878e-9),
            (50,   3.0, 0.1,    5.090018978e-4),
            (200,  3.0, 0.1,    3.504645316e-6),
            (200,  4.0, 0.1,    5.564288841e-5),
            (500,  4.0, 0.1,    9.722534139e-7),
            (100,  5.0, 0.1,    6.745093014e-4),
            (500,  5.0, 0.1,    1.258283729e-5),
            (1000, 5.0, 0.1,    4.137051481e-7),
            (40,   7.0, 0.1,    3.198368332e-3),
            (160,  7.0, 0.1,    6.698029091e-4),
            (640,  7.0, 0.1,    5.063579159e-5),
            (2560, 7.0, 0.1,    2.322138448e-7),
            (100,  9.5, 0.1,    1.656736051e-3),
            (1000, 9.5, 0.1,    6.425026440e-5),
            (8000, 9.5, 0.1,    2.056455838e-8),
            (100,  9.5, 0.01,   1.981917068e-4),
            (1000, 9.5, 0.01,   1.624055198e-5),
            (100,  9.5, 0.0012, 2.424143218e-5),
            (1000, 9.5, 0.0012, 2.319090168e-6),
            (5000, 9.5, 0.0012, 3.366698038e-8),
        ];

        for (n, a, eta, target) in &pts {
            let result = integrated_spectrum(*n, *a, *eta);
            let error = (target - result).abs() / target;
            println!("n = {}, a = {:.2e}, eta = {:.2e} => rate = (alpha/eta) {:.6e}, err = {:.3e}", n, a, eta, result, error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn total_rate() {
        let max_error = 1.0e-3;

        // nmax = 10 (1 + a^3), a, eta
        let pts = [
            (10,    0.5,  0.2),
            (20,    1.0,  0.2),
            (280,   3.0,  0.12),
            (650,   4.0,  0.12),
            (1260,  5.0,  0.12),
            (20,    1.0,  0.75),
            (280,   3.0,  0.75),
            (3440,  7.0,  0.1),
            (6151,  8.5,  0.6),
            (10010, 10.0, 0.0012),
            (10010, 10.0, 0.04),
            (10010, 10.0, 0.08),
            (10010, 10.0, 0.16),
        ];

        for (nmax, a, eta) in &pts {
            let rates: Vec<f64> = (1..=*nmax).map(|n| integrated_spectrum(n, *a, *eta)).collect();
            let total: f64 = rates.iter().sum();
            let target = sum_integrated_spectra(*a, *eta);
            let error = ((total - target) / target).abs();
            println!("a = {:.2e}, eta = {:.2e} => sum_{{n=1}}^{{{}}} rate_n = (alpha/eta) {:.6e}, err = {:.3e}", a, eta, nmax, total, error);
            assert!(error < max_error);
        }
    }

    #[test]
    fn total_rate_low_eta() {
        let max_error = 1.0e-3;

        let pts = [
            (280, 3.0, 0.0005),
            (10010, 10.0, 0.0005),
        ];

        for (nmax, a, eta) in &pts {
            let rates: Vec<f64> = (1..=*nmax).map(|n| integrated_spectrum_low_eta(n, *a)).collect();
            let total: f64 = eta * rates.iter().sum::<f64>();
            let target = sum_integrated_spectra(*a, *eta);
            let error = ((total - target) / target).abs();
            println!("a = {:.2e}, eta = {:.2e} => sum_{{n=1}}^{{{}}} rate_n = (alpha/eta) {:.6e}, err = {:.3e}", a, eta, nmax, total, error);
            assert!(error < max_error);
        }
    }
}
