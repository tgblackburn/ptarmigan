//! Provides possible particle outputs

use crate::constants;
use crate::particle::Particle;
use super::ParticleOutput;

/// Returns the ParticleOutput and its unit.
pub fn identify(name: &str) -> Option<(ParticleOutput, &str)> {
    match name {
        "angle_x" => Some(
            (angle_x as ParticleOutput, "rad")
        ),
        "angle_y" => Some(
            (angle_y as ParticleOutput, "rad")
        ),
        "theta" => Some(
            (theta as ParticleOutput, "rad")
        ),
        "px" => Some(
            (px as ParticleOutput, "MeV/c")
        ),
        "py" => Some(
            (py as ParticleOutput, "MeV/c")
        ),
        "pz" => Some(
            (pz as ParticleOutput, "MeV/c")
        ),
        "p_perp" => Some(
            (p_perp as ParticleOutput, "MeV/c")
        ),
        "r_perp" => Some(
            (r_perp as ParticleOutput, "1")
        ),
        "p^-" | "p-" => Some(
            (p_minus as ParticleOutput, "MeV/c")
        ),
        "p^+" | "p+" => Some(
            (p_plus as ParticleOutput, "MeV/c")
        ),
        "gamma" => Some(
            (gamma as ParticleOutput, "1")
        ),
        "energy" => Some(
            (energy as ParticleOutput, "MeV")
        ),
        "unit" | "number" => Some(
            (unit as ParticleOutput, "1")
        ),
        "x" => Some(
            (x as ParticleOutput, "m")
        ),
        "y" => Some(
            (y as ParticleOutput, "m")
        ),
        "z" => Some(
            (z as ParticleOutput, "m")
        ),
        "birth_a" => Some(
            (payload as ParticleOutput, "1")
        ),
        "n_inter" | "n_gamma" | "n_pos" => Some(
            (interaction_count as ParticleOutput, "1")
        ),
        _ => None,
    }
}

pub fn angle_x(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1].atan2(-p[3])
}

pub fn angle_y(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[2].atan2(-p[3])
}

pub fn theta(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1].hypot(p[2]).atan2(-p[3])
}

pub fn px(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1]
}

pub fn py(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[2]
}

pub fn pz(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[3]
}

pub fn p_perp(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1].hypot(p[2])
}

pub fn p_minus(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[0] - p[3]
}

pub fn p_plus(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[0] + p[3]
}

pub fn r_perp(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1].hypot(p[2]) / (p[0] - p[3])
}

pub fn gamma(pt: &Particle) -> f64 {
    let p = pt.normalized_momentum();
    p[0]
}

pub fn energy(pt: &Particle) -> f64 {
    let p = pt.normalized_momentum();
    p[0] * constants::ELECTRON_MASS_MEV
}

pub fn unit(_pt: &Particle) -> f64 {
    1.0
}

pub fn x(pt: &Particle) -> f64 {
    let r = pt.position();
    r[1]
}

pub fn y(pt: &Particle) -> f64 {
    let r = pt.position();
    r[2]
}

pub fn z(pt: &Particle) -> f64 {
    let r = pt.position();
    r[3]
}

pub fn payload(pt: &Particle) -> f64 {
    pt.payload()
}

pub fn interaction_count(pt: &Particle) -> f64 {
    pt.interaction_count()
}

pub fn weighted_by_energy(pt: &Particle) -> f64 {
    let p = pt.normalized_momentum();
    let energy = p[0] * constants::ELECTRON_MASS_MEV;
    pt.weight() * energy
}

pub fn weighted_by_number(pt: &Particle) -> f64 {
    pt.weight()
}