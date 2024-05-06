//! Default implementations for `push`, `radiate` and `pair_create`,
//! which assume that the field may be treated as locally constant.

use rand::prelude::*;
use crate::constants::*;
use crate::geometry::{FourVector, StokesVector, ThreeVector};
use crate::lcfa;
use super::{EquationOfMotion, RadiationMode, RadiationEvent, PairCreationEvent};

/// Returns the position and momentum of a particle with charge-to-mass ratio `rqm`,
/// which has been accelerated in an electric field `E` and magnetic field `B`
/// over a time interval `dt`.
/// Assumes that ui is defined at t = 0, and r, E, B are defined at t = dt/2.
/// If `with_rr` is true, the energy loss due to radiation emission is handled
/// as part of the particle push, following the classical LL prescription.
#[allow(non_snake_case)]
#[inline(always)]
pub(super) fn vay_push(r: FourVector, ui: FourVector, E: ThreeVector, B: ThreeVector, rqm: f64, dt: f64, eqn: EquationOfMotion) -> (FourVector, FourVector, f64, f64) {
    // velocity in SI units
    let u = ThreeVector::from(ui);
    let gamma = (1.0 + u * u).sqrt(); // enforce mass-shell condition
    let v = SPEED_OF_LIGHT * u / gamma;

    // u_i = u_{i-1/2} + (q dt/2 m c) (E + v_{i-1/2} x B)
    let alpha = rqm * dt / (2.0 * SPEED_OF_LIGHT);
    let u_half = u + alpha * (E + v.cross(B));

    // classical work done by external field, time centered
    let gamma_half = (1.0 + u_half * u_half).sqrt();
    let dwork = rqm * (E * u_half) * dt / (gamma_half * SPEED_OF_LIGHT);

    // (classical) radiated momentum
    let u_rad = if eqn.includes_rr() {
        let u_half_mag = u_half.norm_sqr().sqrt();
        let beta = u_half / u_half_mag;
        let E_rf_sqd = (E + SPEED_OF_LIGHT * beta.cross(B)).norm_sqr() - (E * beta).powi(2);
        let chi = if E_rf_sqd > 0.0 {
            gamma_half * E_rf_sqd.sqrt() / CRITICAL_FIELD
        } else {
            0.0
        };
        let power = 2.0 * ALPHA_FINE * chi * chi / (3.0 * COMPTON_TIME);
        let g_chi = match eqn {
            EquationOfMotion::ModifiedLandauLifshitz => lcfa::photon_emission::gaunt_factor(chi),
            _ => 1.0,
        };
        g_chi * power * dt * u_half / u_half_mag
    } else {
        [0.0; 3].into()
    };

    // u' =  u_{i-1/2} + (q dt/2 m c) E
    let u_prime = u_half + alpha * E;
    let gamma_prime_sqd = 1.0 + u_prime * u_prime;

    // update Lorentz factor
    let tau = alpha * SPEED_OF_LIGHT * B;
    let u_star = u_prime * tau;
    let sigma = gamma_prime_sqd - tau * tau;

    let gamma = (
        0.5 * sigma +
        (0.25 * sigma.powi(2) + tau * tau + u_star.powi(2)).sqrt()
    ).sqrt();

    // and momentum
    let t = tau / gamma;
    let s = 1.0 / (1.0 + t * t);

    let u_new = s * (u_prime + (u_prime * t) * t + u_prime.cross(t));
    let u_new = u_new - u_rad;
    let gamma = (1.0 + u_new * u_new).sqrt();

    let u_new = FourVector::new(gamma, u_new[0], u_new[1], u_new[2]);
    let r_new = r + 0.5 * SPEED_OF_LIGHT * u_new * dt / gamma;

    (r_new, u_new, dt, dwork)
}

/// Pseudorandomly emit a photon from an electron with normalized
/// momentum `u`, which is accelerated by an electric field `E` and
/// magnetic field `B`.
#[allow(non_snake_case)]
#[inline(always)]
pub(super) fn radiate<R: Rng>(u: FourVector, E: ThreeVector, B: ThreeVector, a: f64, dt: f64, rng: &mut R, mode: RadiationMode) -> Option<RadiationEvent> {
    let classical = mode == RadiationMode::Classical;
    let beta = ThreeVector::from(u) / u[0];
    let E_rf_sqd = (E + SPEED_OF_LIGHT * beta.cross(B)).norm_sqr() - (E * beta).powi(2);
    let chi = if E_rf_sqd > 0.0 {
        u[0] * E_rf_sqd.sqrt() / CRITICAL_FIELD
    } else {
        0.0
    };

    let prob = if classical {
        dt * lcfa::photon_emission::classical::rate(chi, u[0])
    } else {
        dt * lcfa::photon_emission::rate(chi, u[0])
    };

    if rng.gen::<f64>() < prob {
        let (omega_mc2, theta, cphi) = if classical {
            lcfa::photon_emission::classical::sample(chi, u[0], rng.gen(), rng.gen(), rng.gen())
        } else {
            lcfa::photon_emission::sample(chi, u[0], rng.gen(), rng.gen(), rng.gen())
        };

        if let Some(theta) = theta {
            let long: ThreeVector = beta.normalize();
            let w = -(E - (long * E) * long / E.norm_sqr().sqrt() + SPEED_OF_LIGHT * beta.cross(B)).normalize();
            let perp: ThreeVector = w.rotate_around(long, cphi);
            let k: ThreeVector = omega_mc2 * (theta.cos() * long + theta.sin() * perp);
            let k = FourVector::lightlike(k[0], k[1], k[2]);
            let pol = if classical {
                lcfa::photon_emission::classical::stokes_parameters(k, chi, u[0], beta, w)
            } else {
                lcfa::photon_emission::stokes_parameters(k, chi, u[0], beta, w)
            };

            Some(RadiationEvent {
                k,
                u_prime: u - k,
                pol,
                a_eff: a,
                absorption: 0.0
            })
        } else {
            None
        }
    } else {
        None
    }
}

/// Pseudorandomly create an electron-positron pair from a photon with
/// normalized momentum `u`, in an electric field `E` and
/// magnetic field `B`, returning the probability, the actual rate
/// increase used, the new Stokes parameters of the photon, as well as
/// the momenta of the electron and positron that are created and
/// the effective amplitude at the point of creation.
#[allow(non_snake_case)]
#[inline(always)]
pub(super) fn pair_create<R: Rng>(u: FourVector, sv: StokesVector, E: ThreeVector, B: ThreeVector, a: f64, dt: f64, rng: &mut R, rate_increase: f64) -> (f64, StokesVector, Option<PairCreationEvent>) {
    let n = ThreeVector::from(u).normalize();

    // transverse "acceleration"
    let a_perp = E - (E * n) * n + SPEED_OF_LIGHT * n.cross(B);
    let E_rf_sqd = a_perp.norm_sqr();

    let (chi, prob, sv_new) = if E_rf_sqd > 0.0 {
        let chi = u[0] * E_rf_sqd.sqrt() / CRITICAL_FIELD;
        let (prob, sv_new) = lcfa::pair_creation::probability(u, sv, chi, a_perp, dt);
        (chi, prob, sv_new)
    } else {
        (0.0, 0.0, sv)
    };

    let rate_increase = if prob * rate_increase > 0.1 {
        0.1 / prob // limit the rate increase
    } else {
        rate_increase
    };

    if rng.gen::<f64>() < prob * rate_increase {
        let (gamma_p, cos_theta, cphi, _, _) = lcfa::pair_creation::sample(u, sv, chi, a_perp, rng);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let u_p = gamma_p * (1.0 - 1.0 / (gamma_p * gamma_p)).sqrt();

        // axes
        let e_1 = a_perp.normalize();
        let e_2 = n.cross(e_1);
        let u_p = u_p * (cos_theta * n + sin_theta * cphi.cos() * e_1 + sin_theta * cphi.sin() * e_2);

        // conserving three-momentum
        let u_e = ThreeVector::from(u) - u_p;
        let u_p = FourVector::new(0.0, u_p[0], u_p[1], u_p[2]).unitize();
        let u_e = FourVector::new(0.0, u_e[0], u_e[1], u_e[2]).unitize();

        let event = PairCreationEvent {
            u_e,
            u_p,
            frac: 1.0 / rate_increase,
            a_eff: a,
            absorption: 0.0,
        };

        (prob, sv_new, Some(event))
    } else {
        (prob, sv_new, None)
    }
}