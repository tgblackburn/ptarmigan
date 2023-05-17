//! Nonlinear Compton scattering

use std::f64::consts;
use rand::prelude::*;
use crate::constants::*;
use crate::geometry::*;
use crate::field::{Polarization, RadiationMode};

mod cp;
mod lp;

/// The total probability that a photon is emitted
/// by an electron with normalized quasimomentum `q`
/// in a plane EM wave with (local) wavector `k`
/// and polarization `pol` over an interval `dt`.
///
/// Both `k` and `q` are expected to be normalized
/// to the electron mass.
pub fn probability(k: FourVector, q: FourVector, dt: f64, pol: Polarization, mode: RadiationMode) -> Option<f64> {
    use {Polarization::*, RadiationMode::*};

    let a_sqd = q * q - 1.0;
    let a = if a_sqd >= 0.0 {
        a_sqd.sqrt()
    } else {
        0.0
    };
    let eta = k * q;
    let dphi = dt * eta / (COMPTON_TIME * q[0]);

    let f = match (pol, mode) {
        (Circular, Quantum) => cp::rate(a, eta).unwrap(),
        (Circular, Classical) => cp::classical::rate(a, eta).unwrap(),
        (Linear, Quantum) => lp::rate(a * consts::SQRT_2, eta).unwrap(),
        (Linear, Classical) => lp::classical::rate(a * consts::SQRT_2, eta).unwrap(),
    };

    // let f = match pol {
    //     Polarization::Circular => cp::rate(a, eta).unwrap(),
    //     Polarization::Linear => match mode {
    //         RadiationMode::Quantum => lp::rate(a * consts::SQRT_2, eta).unwrap(),
    //         RadiationMode::Classical => lp::classical::rate(a * consts::SQRT_2, eta).unwrap()
    //     },
    // };

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
pub fn generate<R: Rng>(k: FourVector, q: FourVector, pol: Polarization, mode: RadiationMode, rng: &mut R) -> (i32, FourVector, StokesVector) {
    use {Polarization::*, RadiationMode::*};

    let a = (q * q - 1.0).sqrt(); // rms value!
    let eta = k * q;

    let (n, s, cphi_zmf, sv) = match (pol, mode) {
        (Circular, Quantum) => cp::sample(a, eta, rng, None),
        (Circular, Classical) => cp::classical::sample(a, eta, rng, None),
        (Linear, Quantum) => lp::sample(a * consts::SQRT_2, eta, rng, None),
        (Linear, Classical) => lp::classical::sample(a * consts::SQRT_2, eta, rng, None),
    };

    let ell = n as f64;

    // Scattering momentum (/m) and angles in zero momentum frame
    let p_zmf = match mode {
        Classical => ell * eta / (1.0 + a * a).sqrt(),
        Quantum => ell * eta / (1.0 + a * a + 2.0 * ell * eta).sqrt(),
    };

    let cos_theta_zmf = match mode {
        Classical => 1.0 - s * (1.0 + a * a) / (ell * eta),
        Quantum => 1.0 - s * (1.0 + a * a + 2.0 * ell * eta) / (ell * eta),
    };

    assert!(cos_theta_zmf <= 1.0);
    assert!(cos_theta_zmf >= -1.0);

    // Four-velocity of ZMF (normalized)
    let u_zmf = match mode {
        Classical => q / q.norm_sqr().sqrt(),
        Quantum => (q + ell * k) / (q + ell * k).norm_sqr().sqrt(),
    };

    // Unit vectors pointed antiparallel to electron momentum in ZMF
    // and perpendicular to it
    // println!("ZMF: q = [{}], ell k = [{}], |q| = {}", q.boost_by(u_zmf), (ell * k).boost_by(u_zmf), p_zmf);
    let along = match mode {
        // need to avoid that q = (0, 0, o) because ZMF coincides with RF in classical mode
        Classical => ThreeVector::from((ell * k).boost_by(u_zmf)).normalize(),
        Quantum => -ThreeVector::from(q.boost_by(u_zmf)).normalize()
    };
    let epsilon = ThreeVector::from(FourVector::new(0.0, 1.0, 0.0, 0.0).boost_by(u_zmf)).normalize();
    let epsilon = {
        let k = along;
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

    let sv = sv.from_lma_basis(k_prime);

    // Verify construction of photon momentum
    let s_new = (k * k_prime) / (k * q);
    let error = (s - s_new).abs() / s;
    if error >= 1.0e-3 {
        eprintln!(
            "nonlinear_compton::generate failed sanity check by {:.3}% during construction of photon momentum \
            at eta = {:.3e}, a = {:.3e}, n = {}: sampled s = {:.3e}, reconstructed = {:.3e}, {}...",
            100.0 * error, eta, a, n, s, s_new, if error >= 1.0e-2 {"halting"} else {"continuing"},
        );
    }
    assert!(error < 1.0e-2);

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
        let theta = 30_f64.to_radians();
        let u = FourVector::new(0.0, u * theta.sin(), 0.0, -u * theta.cos()).unitize();
        let q = u + 0.5 * a * a * k / (2.0 * k * u);
        let mode = RadiationMode::Quantum;

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
                let (n, k_prime, sv) = generate(k, q, Polarization::Linear, mode, &mut rng);
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
