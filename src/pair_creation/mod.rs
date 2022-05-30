//! Nonlinear Breit-Wheeler pair creation

use std::f64::consts;
use rand::prelude::*;
use crate::constants::*;
use crate::geometry::*;
use crate::field::Polarization;

mod cp;
mod lp;

/// The total probability that an electron-positron pair
/// is created by a photon with momentum `ell` and
/// polarization `sv`,
/// in a plane EM wave with (local) wavector `k`,
/// root-mean-square amplitude `a`, 
/// and polarization `pol` over an interval `dt`.
///
/// Both `ell` and `k` are expected to be normalized
/// to the electron mass.
pub fn probability(ell: FourVector, _sv: StokesVector, k: FourVector, a: f64, dt: f64, pol: Polarization) -> Option<f64> {
    let eta = k * ell;

    let f = match pol {
        Polarization::Circular => cp::rate(a, eta).unwrap(),
        Polarization::Linear => lp::rate(a * consts::SQRT_2, eta).unwrap(),
    };

    Some(ALPHA_FINE * f * dt / (COMPTON_TIME * ell[0]))
}

/// Assuming that pair creation takes place, pseudorandomly
/// generate the momentum of the positron generated
/// by a photon with normalized momentum `ell` and polarization `sv`
/// in a plane EM wave with root-mean-square amplitude `a`,
/// (local) wavector `k` and polarization `pol`.
pub fn generate<R: Rng>(ell: FourVector, _sv: StokesVector, k: FourVector, a: f64, pol: Polarization, rng: &mut R) -> (i32, FourVector) {
    let eta: f64 = k * ell;

    let (n, s, cphi_zmf) = match pol {
        Polarization::Circular => cp::sample(a, eta, rng),
        Polarization::Linear => lp::sample(a * consts::SQRT_2, eta, rng),
    };

    // Scattering momentum (/m) and angles in zero momentum frame
    // if ell_perp = 0, (q_perp/m)^2 = 2 n eta s (1-s) - (1+a^2)
    let j = n as f64;
    let e_zmf = (0.5 * j * eta).sqrt();
    let p_zmf = (0.5 * j * eta - 1.0 - a * a).sqrt();
    let cos_theta_zmf = (1.0 - 2.0 * s) * e_zmf / p_zmf;

    assert!(cos_theta_zmf <= 1.0);
    assert!(cos_theta_zmf >= -1.0);

    // Four-velocity of ZMF (normalized)
    let u_zmf: FourVector = (ell + j * k) / (ell + j * k).norm_sqr().sqrt();

    // Unit vectors pointed parallel to gamma-ray momentum in ZMF
    // and perpendicular to it
    let along = -ThreeVector::from((j*k).boost_by(u_zmf)).normalize();

    let epsilon = ThreeVector::from(FourVector::new(0.0, 1.0, 0.0, 0.0).boost_by(u_zmf)).normalize();
    let epsilon = {
        let k = -ThreeVector::from(k).normalize();
        k.cross(epsilon.cross(k)).normalize()
    };
    let perp = epsilon.rotate_around(along, cphi_zmf);

    // Construct positron momentum and transform back to lab frame
    let q: ThreeVector = p_zmf * (cos_theta_zmf * along + (1.0 - cos_theta_zmf.powi(2)).sqrt() * perp);
    let q = FourVector::lightlike(q[0], q[1], q[2]).with_sqr(1.0 + a * a);
    let q = q.boost_by(u_zmf.reverse());

    (n, q)
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
    fn pair_spectrum() {
        let a = 0.2;
        let eta = 2.2;
        let k = (1.55e-6 / 0.511) * FourVector::new(1.0, 0.0, 0.0, 1.0);
        let ell = (0.511 * eta / (2.0 * 1.55e-6)) * FourVector::new(1.0, 0.0, 0.0, -1.0);
        let pol = StokesVector::unpolarized();
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let rt = std::time::Instant::now();
        let pts: Vec<(i32, f64, f64)> = (0..1_000_000)
            .map(|_| generate(ell, pol, k, a, Polarization::Circular, &mut rng))
            .map(|(n, q)| (n, (k * q) / (k * ell), q[1].hypot(q[2])))
            .collect();
        let rt = rt.elapsed();

        println!("a = {:.3e}, eta = {:.3e}, {} samples takes {:?}", a, k * ell, pts.len(), rt);
        let mut file = File::create("output/positron_spectrum.dat").unwrap();
        for (n, s, q_perp) in pts {
            writeln!(file, "{} {:.6e} {:.6e}", n, s, q_perp).unwrap();
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