//! Nonlinear Compton scattering

use std::f64::consts;
use rand::prelude::*;
use crate::constants::*;
use crate::geometry::*;
use crate::special_functions::*;

mod total;

/// The total probability that a photon is emitted
/// by an electron with normalized quasimomentum `q`
/// in a plane EM wave with (local) wavector `k`
/// over an interval `dt`.
///
/// Both `k` and `q` are expected to be normalized
/// to the electron mass.
pub fn probability(k: FourVector, q: FourVector, dt: f64) -> Option<f64> {
    let a_sqd = q * q - 1.0;
    let a = if a_sqd >= 0.0 {
        a_sqd.sqrt()
    } else {
        0.0
    };
    let eta = k * q;
    let dphi = dt * eta / (COMPTON_TIME * q[0]);

    let f = sum_integrated_spectra(a, eta);

    Some(ALPHA_FINE * f * dphi / eta)
}

/// Evaluates the important part of the nonlinear Compton
/// differential rate, f,
/// either
///   `dP/(dv dϕ) = ⍺ f(n, a, η, v) / η`
/// or
///   `dP/(dv dt) = ⍺ m f(n, a, η, v) / γ`
/// where `0 < v < 1`
fn spectrum(n: i32, a: f64, eta: f64, v: f64) -> f64 {
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

/// Integrates the important part of the nonlinear Compton
/// differential rate, f, which gives either
///   `dP/(dv dϕ) = ⍺ f(n, a, η, v) / η`
/// or
///   `dP/(dv dt) = ⍺ m f(n, a, η, v) / γ`
/// over the domain `0 < v < 1`
fn integrated_spectrum(n: i32, a: f64, eta: f64) -> f64 {
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

/// Returns the sum, over harmonic index, of the partial nonlinear
/// Compton rates. Equivalent to calling
/// ```
/// let nmax = (10.0 * (1.0 + a.powi(3))) as i32;
/// let rate = (1..=nmax).map(|n| integrated_spectrum(n, a, eta)).sum::<f64>();
/// ```
/// but implemented as a table lookup.
fn sum_integrated_spectra(a: f64, eta: f64) -> f64 {
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

/// Assuming that emission takes place, pseudorandomly
/// generate the momentum of the photon emitted
/// by an electron with normalized quasimomentum `q`
/// in a plane EM wave with (local) wavector `k`.
///
/// Returns the harmonic index of the photon and the
/// normalized momentum.
pub fn generate<R: Rng>(k: FourVector, q: FourVector, rng: &mut R, fixed_n: Option<i32>) -> (i32, FourVector) {
    let a = (q * q - 1.0).sqrt();
    let eta = k * q;

    // From the partial rates, pick a harmonic number
    // or use one specified
    // let n = fixed_n.unwrap_or_else(|| {
    //     let nmax = (10.0 * (1.0 + a * a * a)) as i32;
    //     let mut cumsum: f64 = 0.0;
    //     let mut rates: Vec<f64> = Vec::with_capacity(nmax as usize);
    //     for k in 1..=nmax {
    //         cumsum += integrated_spectrum(k, a, eta);
    //         rates.push(cumsum);
    //     }
    //     let total = rates.last().unwrap();
    //     let target = total * rng.gen::<f64>();
    //     let (index, _) = rates.iter().enumerate().find(|(_i, &cs)| cs > target).unwrap();
    //     (index + 1) as i32
    // });
    let n = fixed_n.unwrap_or_else(|| {
        let nmax = (10.0 * (1.0 + a * a * a)) as i32;
        let target = sum_integrated_spectra(a, eta) * rng.gen::<f64>();
        let mut cumsum: f64 = 0.0;
        let mut index: i32 = 1;
        for k in 1..=nmax {
            cumsum += integrated_spectrum(k, a, eta);
            if cumsum > target {
                index = k;
                break;
            }
        };
        index
    });

    assert!(n >= 1);

    // Approximate maximum value of the probability density:
    let max: f64 = GAUSS_32_NODES.iter()
        .map(|x| spectrum(n, a, eta, 0.5 * (x + 1.0)))
        .fold(0.0f64 / 0.0f64, |a: f64, b: f64| a.max(b));
    let max = 1.5 * max;

    // Rejection sampling
    let v = loop {
        let v = rng.gen::<f64>();
        let u = rng.gen::<f64>();
        let f = spectrum(n, a, eta, v);
        if u <= f / max {
            break v;
        }
    };

    // Four-momentum transfer s = k.l / k.q
    let s = {
        let sn = 2.0 * (n as f64) * eta / (1.0 + a * a);
        let smax = sn / (1.0 + sn);
        v * smax
    };

    let ell = n as f64;

    // Scattering momentum (/m) and angles in zero momentum frame
    let p_zmf = ell * eta / (1.0 + a * a + 2.0 * ell * eta).sqrt();
    let cos_theta_zmf = 1.0 - s * (1.0 + a * a + 2.0 * ell * eta) / (ell * eta);
    let cphi_zmf = 2.0 * consts::PI * rng.gen::<f64>();

    assert!(cos_theta_zmf <= 1.0);
    assert!(cos_theta_zmf >= -1.0);

    // Four-velocity of ZMF (normalized)
    let u_zmf = (q + ell * k) / (q + ell * k).norm_sqr().sqrt();

    // Unit vectors pointed antiparallel to electron momentum in ZMF
    // and perpendicular to it
    //println!("ZMF: q = [{}], ell k = [{}], |q| = {}", q.boost_by(u_zmf), (ell * k).boost_by(u_zmf), p_zmf);
    let along = -ThreeVector::from(q.boost_by(u_zmf)).normalize();
    let perp = along.orthogonal().rotate_around(along, cphi_zmf);

    // Construct photon momentum and transform back to lab frame
    let k_prime = p_zmf * (cos_theta_zmf * along + (1.0 - cos_theta_zmf.powi(2)).sqrt() * perp);
    //println!("ZMF: |k'| = {}", k_prime.norm_sqr().sqrt());
    let k_prime = FourVector::lightlike(k_prime[0], k_prime[1], k_prime[2]);
    //println!("ZMF: k.k' = {}", k.boost_by(u_zmf) * k_prime);
    let k_prime = k_prime.boost_by(u_zmf.reverse());
    //println!("lab: k.k' = {}", k * k_prime);

    (n, k_prime)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::*;

    #[test]
    #[ignore]
    fn partial_spectrum() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let n = 250;
        let a = 1.0;
        let k = (1.55e-6 / 0.511) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        let u = 10.0 * 1000.0 / 0.511;
        let u = FourVector::new(0.0, 0.0, 0.0, -u).unitize();
        let q = u + a * a * k / (2.0 * k * u);

        let rt = std::time::Instant::now();
        let vs: Vec<(f64,f64,f64)> = (0..10000)
            .map(|_n| {
                let (_, k_prime) = generate(k, q, &mut rng, Some(n));
                (k * k_prime / (k * q), k_prime[1], k_prime[2])
            })
            .collect();
        let rt = rt.elapsed();

        println!("a = {:.3e}, eta = {:.3e}, {} samples takes {:?}", (q * q - 1.0).sqrt(), k * q, vs.len(), rt);
        let mut file = File::create("output/partial_spectrum.dat").unwrap();
        for v in vs {
            writeln!(file, "{:.6e} {:.6e} {:.6e}", v.0, v.1, v.2).unwrap();
        }
    }

    #[test]
    #[ignore]
    fn spectrum() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let a = 2.0;
        let k = (1.55e-6 / 0.511) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        let u = 10.0 * 1000.0 / 0.511;
        let u = FourVector::new(0.0, 0.0, 0.0, -u).unitize();
        let q = u + a * a * k / (2.0 * k * u);

        let rt = std::time::Instant::now();
        let vs: Vec<(f64,f64,f64,i32)> = (0..100_000)
            .map(|_i| {
                let (n, k_prime) = generate(k, q, &mut rng, None);
                (k * k_prime / (k * q), k_prime[1], k_prime[2], n)
            })
            .collect();
        let rt = rt.elapsed();

        println!("a = {:.3e}, eta = {:.3e}, {} samples takes {:?}", (q * q - 1.0).sqrt(), k * q, vs.len(), rt);
        let mut file = File::create("output/spectrum.dat").unwrap();
        for v in vs {
            writeln!(file, "{:.6e} {:.6e} {:.6e} {}", v.0, v.1, v.2, v.3).unwrap();
        }
    }

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

    #[test]
    fn ds_crosscheck() {
        // Set up laser
        // kappa = hbar k / (m c^2)
        let kappa = (1.55e-6 / 0.511) * FourVector::new(1.0, 0.0, 0.0, 1.0);

        // and electron beam initial momentum
        let theta = 0.1 * consts::PI;
        let u = 17.5e3 / 0.511;
        let u = FourVector::new(0.0, u * theta.sin() , 0.0, -u * theta.cos()).unitize();

        // from Daniel
        let a0s = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0];
        let rates = [
            8.413499945078843e-07, 
            8.39719977159875e-05, 
            0.0003337455877314787, 
            0.0020039528587275273, 
            0.007101726591791834, 
            0.02110454417089373, 
            0.03615707160015743, 
            0.06534971657183357,
            0.10592260692789592,
            0.1296736562098593,
        ];

        for (a, target) in a0s.iter().zip(rates.iter()) {
            // electron quasi momentum q/mc
            let q = u + a * a * kappa / (2.0 * kappa * u);
            let rate = probability(kappa, q, 1.0e-15).unwrap_or(0.0);
            let target = target * (1.0 + theta.cos());
            let error = (target - rate).abs() / target;
            println!("a0 = {:.3e}, target = {:.3e}, got = {:.3e}, err = {:.3e}", a, target, rate, error);
        }
    }
}

static GAUSS_32_NODES: [f64; 32] = [
    -9.972638618494816e-1,
    -9.856115115452683e-1,
    -9.647622555875064e-1,
    -9.349060759377397e-1,
    -8.963211557660521e-1,
    -8.493676137325700e-1,
    -7.944837959679424e-1,
    -7.321821187402897e-1,
    -6.630442669302152e-1,
    -5.877157572407623e-1,
    -5.068999089322294e-1,
    -4.213512761306353e-1,
    -3.318686022821276e-1,
    -2.392873622521371e-1,
    -1.444719615827965e-1,
    -4.830766568773832e-2,
    4.830766568773832e-2,
    1.444719615827965e-1,
    2.392873622521371e-1,
    3.318686022821276e-1,
    4.213512761306353e-1,
    5.068999089322294e-1,
    5.877157572407623e-1,
    6.630442669302152e-1,
    7.321821187402897e-1,
    7.944837959679424e-1,
    8.493676137325700e-1,
    8.963211557660521e-1,
    9.349060759377397e-1,
    9.647622555875064e-1,
    9.856115115452683e-1,
    9.972638618494816e-1,
];

static GAUSS_32_WEIGHTS: [f64; 32] = [
    7.018610000000000e-3,
    1.627439500000000e-2,
    2.539206500000000e-2,
    3.427386300000000e-2,
    4.283589800000000e-2,
    5.099805900000000e-2,
    5.868409350000000e-2,
    6.582222280000000e-2,
    7.234579411000000e-2,
    7.819389578700000e-2,
    8.331192422690000e-2,
    8.765209300440000e-2,
    9.117387869576400e-2,
    9.384439908080460e-2,
    9.563872007927486e-2,
    9.654008851472780e-2,
    9.654008851472780e-2,
    9.563872007927486e-2,
    9.384439908080460e-2,
    9.117387869576400e-2,
    8.765209300440000e-2,
    8.331192422690000e-2,
    7.819389578700000e-2,
    7.234579411000000e-2,
    6.582222280000000e-2,
    5.868409350000000e-2,
    5.099805900000000e-2,
    4.283589800000000e-2,
    3.427386300000000e-2,
    2.539206500000000e-2,
    1.627439500000000e-2,
    7.018610000000000e-3,
];
