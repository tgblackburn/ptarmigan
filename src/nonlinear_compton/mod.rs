//! Nonlinear Compton scattering

use std::f64::consts;
use rand::prelude::*;
use crate::constants::*;
use crate::geometry::*;
use crate::field::Polarization;

mod cp;
mod lp;

/// The total probability that a photon is emitted
/// by an electron with normalized quasimomentum `q`
/// in a plane EM wave with (local) wavector `k`
/// and polarization `pol` over an interval `dt`.
///
/// Both `k` and `q` are expected to be normalized
/// to the electron mass.
pub fn probability(k: FourVector, q: FourVector, dt: f64, pol: Polarization) -> Option<f64> {
    let a_sqd = q * q - 1.0;
    let a = if a_sqd >= 0.0 {
        a_sqd.sqrt()
    } else {
        0.0
    };
    let eta = k * q;
    let dphi = dt * eta / (COMPTON_TIME * q[0]);

    let f = match pol {
        Polarization::Circular => cp::sum_integrated_spectra(a, eta),
        Polarization::Linear => unimplemented!(),
    };

    Some(ALPHA_FINE * f * dphi / eta)
}

/// Assuming that emission takes place, pseudorandomly
/// generate the momentum of the photon emitted
/// by an electron with normalized quasimomentum `q`
/// in a plane EM wave with (local) wavector `k`
/// and polarization `pol`.
///
/// Returns the harmonic index of the photon and the
/// normalized momentum.
pub fn generate<R: Rng>(k: FourVector, q: FourVector, pol: Polarization, rng: &mut R, fixed_n: Option<i32>) -> (i32, FourVector) {
    let a = (q * q - 1.0).sqrt();
    let eta = k * q;

    let (sum_integrated_spectra, integrated_spectrum, spectrum) = match pol {
        Polarization::Circular => (cp::sum_integrated_spectra, cp::integrated_spectrum, cp::spectrum),
        Polarization::Linear => unimplemented!(),
    };

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
                let (_, k_prime) = generate(k, q, Polarization::Circular, &mut rng, Some(n));
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
                let (n, k_prime) = generate(k, q, Polarization::Circular, &mut rng, None);
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
}

static GAUSS_16_NODES: [f64; 16] = [
    -9.894009349916499e-1,
    -9.445750230732326e-1,
    -8.656312023878317e-1,
    -7.554044083550030e-1,
    -6.178762444026437e-1,
    -4.580167776572274e-1,
    -2.816035507792589e-1,
    -9.501250983763744e-2,
    9.501250983763744e-2,
    2.816035507792589e-1,
    4.580167776572274e-1,
    6.178762444026437e-1,
    7.554044083550030e-1,
    8.656312023878317e-1,
    9.445750230732326e-1,
    9.894009349916499e-1,
];

static GAUSS_16_WEIGHTS: [f64; 16] = [
    2.715245941175400e-2,
    6.225352393864800e-2,
    9.515851168249300e-2,
    1.246289712555340e-1,
    1.495959888165770e-1,
    1.691565193950025e-1,
    1.826034150449236e-1,
    1.894506104550685e-1,
    1.894506104550685e-1,
    1.826034150449236e-1,
    1.691565193950025e-1,
    1.495959888165770e-1,
    1.246289712555340e-1,
    9.515851168249300e-2,
    6.225352393864800e-2,
    2.715245941175400e-2,
];

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
