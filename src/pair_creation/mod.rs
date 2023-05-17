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
pub fn probability(ell: FourVector, sv: StokesVector, k: FourVector, a: f64, dt: f64, pol: Polarization) -> (f64, StokesVector) {
    let eta = k * ell;
    let sv = sv.in_lma_basis(ell);
    let dphi = eta * dt / (COMPTON_TIME * ell[0]);

    let (prob, sv) = match pol {
        Polarization::Circular => {
            let rate = cp::TotalRate::new(a, eta);
            rate.probability(sv, dphi)
        },
        Polarization::Linear => {
            let rate = lp::TotalRate::new(a * consts::SQRT_2, eta);
            rate.probability(sv, dphi)
        },
    };

    let sv = sv.from_lma_basis(ell);

    (prob, sv)
}

/// Assuming that pair creation takes place, pseudorandomly
/// generate the momentum of the positron generated
/// by a photon with normalized momentum `ell` and polarization `sv`
/// in a plane EM wave with root-mean-square amplitude `a`,
/// (local) wavector `k` and polarization `pol`.
pub fn generate<R: Rng>(ell: FourVector, sv: StokesVector, k: FourVector, a: f64, pol: Polarization, rng: &mut R) -> (i32, FourVector) {
    let eta: f64 = k * ell;
    let sv = sv.in_lma_basis(ell);

    let (n, s, cphi_zmf) = match pol {
        Polarization::Circular => {
            let rate = cp::TotalRate::new(a, eta);
            rate.sample(sv[1], sv[2], sv[3], rng)
        },
        Polarization::Linear => {
            let rate = lp::TotalRate::new(a * consts::SQRT_2, eta);
            rate.sample(sv[1], sv[2], rng)
        },
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
        let k = along;
        k.cross(epsilon.cross(k)).normalize()
    };
    let perp = epsilon.rotate_around(along, cphi_zmf);

    // Construct positron momentum and transform back to lab frame
    let q: ThreeVector = p_zmf * (cos_theta_zmf * along + (1.0 - cos_theta_zmf.powi(2)).sqrt() * perp);
    let q = FourVector::lightlike(q[0], q[1], q[2]).with_sqr(1.0 + a * a);
    let q = q.boost_by(u_zmf.reverse());

    // Verify construction of positron momentum
    // println!("s: sampled = {:.6e}, reconstructed = {:.6e}", s, 1.0 - (k * q) / (k * ell));
    let s_new = 1.0 - (k * q) / (k * ell); // flipped sign
    let error = (s - s_new).abs() / s;
    if error >= 1.0e-3 {
        eprintln!(
            "pair_creation::generate failed sanity check by {:.3}% during construction of positron momentum \
            at eta = {:.3e}, a = {:.3e}, n = {}: sampled s = {:.3e}, reconstructed = {:.3e}, halting...",
            100.0 * error, eta, a, n, s, s_new,
        );
    }
    assert!(error < 1.0e-3);

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
