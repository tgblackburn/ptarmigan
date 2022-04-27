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
        Polarization::Circular => cp::rate(a, eta).unwrap(),
        Polarization::Linear => lp::rate(a * consts::SQRT_2, eta).unwrap(),
    };

    Some(ALPHA_FINE * f * dphi / eta)
}

/// Assuming that emission takes place, pseudorandomly
/// generate the momentum of the photon emitted
/// by an electron with normalized quasimomentum `q`
/// in a plane EM wave with (local) wavector `k`
/// and polarization `pol`.
///
/// Returns the harmonic index of the photon,
/// the normalized momentum,
/// and the polarization (if applicable).
pub fn generate<R: Rng>(k: FourVector, q: FourVector, pol: Polarization, rng: &mut R) -> (i32, FourVector, StokesVector) {
    let a = (q * q - 1.0).sqrt(); // rms value!
    let eta = k * q;

    let (n, s, cphi_zmf, sv) = match pol {
        Polarization::Circular => cp::sample(a, eta, rng, None),
        Polarization::Linear => lp::sample(a * consts::SQRT_2, eta, rng, None),
    };

    let ell = n as f64;

    // Scattering momentum (/m) and angles in zero momentum frame
    let p_zmf = ell * eta / (1.0 + a * a + 2.0 * ell * eta).sqrt();
    let cos_theta_zmf = 1.0 - s * (1.0 + a * a + 2.0 * ell * eta) / (ell * eta);

    assert!(cos_theta_zmf <= 1.0);
    assert!(cos_theta_zmf >= -1.0);

    // Four-velocity of ZMF (normalized)
    let u_zmf = (q + ell * k) / (q + ell * k).norm_sqr().sqrt();

    // Unit vectors pointed antiparallel to electron momentum in ZMF
    // and perpendicular to it
    //println!("ZMF: q = [{}], ell k = [{}], |q| = {}", q.boost_by(u_zmf), (ell * k).boost_by(u_zmf), p_zmf);
    let along = -ThreeVector::from(q.boost_by(u_zmf)).normalize();
    let epsilon = ThreeVector::from(FourVector::new(0.0, 1.0, 0.0, 0.0).boost_by(u_zmf)).normalize();
    let epsilon = {
        let k = -ThreeVector::from(q).normalize();
        k.cross(epsilon.cross(k)).normalize()
    };
    //println!("ZMF: along.e_1 = {:.3e}, along = [{}], e_1 = [{}]", along * epsilon, along, epsilon);
    let perp = epsilon.rotate_around(along, cphi_zmf);

    // Construct photon momentum and transform back to lab frame
    let k_prime = p_zmf * (cos_theta_zmf * along + (1.0 - cos_theta_zmf.powi(2)).sqrt() * perp);
    //println!("ZMF: |k'| = {}", k_prime.norm_sqr().sqrt());
    let k_prime = FourVector::lightlike(k_prime[0], k_prime[1], k_prime[2]);
    //println!("ZMF: k.k' = {}", k.boost_by(u_zmf) * k_prime);
    let k_prime = k_prime.boost_by(u_zmf.reverse());
    //println!("lab: k.k' = {}", k * k_prime);

    // Stokes vector is defined w.r.t. to the orthonormal basis
    // e_1 = x - k_x (k - omega z) / (omega * (omega - k_z))
    // e_2 = y - k_y (k - omega z) / (omega * (omega - k_z))
    // which are perpendicular to k.

    // The global basis requires one vector to be in the x-z plane,
    // so e_1 gets rotated by
    let theta = {
        let sin_theta = -k_prime[1] * k_prime[2] / (k_prime[0] * (k_prime[0] - k_prime[3]));
        sin_theta.asin()
    };

    // and the Stokes parameters by
    let sv = sv.rotate_by(theta);

    (n, k_prime, sv)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use rand::prelude::*;
    use rayon::prelude::*;
    use super::*;

    #[test]
    #[ignore]
    fn spectrum() {
        let a = 2.0;
        let eta = 0.1;
        let k = (1.55e-6 / 0.511) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        let u = 0.511 * eta / (2.0 * 1.55e-6);
        let u = FourVector::new(0.0, 0.0, 0.0, -u).unitize();
        let q = u + 0.5 * a * a * k / (2.0 * k * u);

        let num: usize = std::env::var("RAYON_NUM_THREADS")
            .map(|s| s.parse().unwrap_or(1))
            .unwrap_or(1);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num)
            .build()
            .unwrap();

        println!("Running on {:?}", pool);

        let rt = std::time::Instant::now();

        let vs: Vec<_> = pool.install(|| {
            (0..1000).into_par_iter().map(|_i| {
                let mut rng = thread_rng();
                let (n, k_prime, sv) = generate(k, q, Polarization::Linear, &mut rng);
                let weight = sv.project_onto(ThreeVector::from(k_prime).normalize(), [0.0, 1.0, 0.0].into());
                (k * k_prime / (k * q), k_prime[1], k_prime[2], n, sv[1], weight)
            })
            .collect()
        });

        let rt = rt.elapsed();

        println!("a = {:.3e}, eta = {:.3e}, {} samples takes {:?}", (q * q - 1.0).sqrt(), k * q, vs.len(), rt);
        let filename = format!("output/nlc_spectrum_{}_{}.dat", a, eta);
        let mut file = File::create(&filename).unwrap();
        for v in vs {
            writeln!(file, "{:.6e} {:.6e} {:.6e} {} {:.6e} {:.6e}", v.0, v.1, v.2, v.3, v.4, v.5).unwrap();
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
