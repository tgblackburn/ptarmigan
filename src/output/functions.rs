//! Provides possible particle outputs

use crate::constants;
use crate::particle::Particle;
use super::ParticleOutput;

/// Returns the ParticleOutput and its unit.
pub fn identify(name: &str) -> Option<(ParticleOutput, &str)> {
    match name {
        "angle_x" => Some(
            (Box::new(angle_x) as ParticleOutput, "rad")
        ),
        "angle_y" => Some(
            (Box::new(angle_y) as ParticleOutput, "rad")
        ),
        "theta" => Some(
            (Box::new(theta) as ParticleOutput, "rad")
        ),
        "px" => Some(
            (Box::new(px) as ParticleOutput, "MeV/c")
        ),
        "py" => Some(
            (Box::new(py) as ParticleOutput, "MeV/c")
        ),
        "pz" => Some(
            (Box::new(pz) as ParticleOutput, "MeV/c")
        ),
        "p_perp" => Some(
            (Box::new(p_perp) as ParticleOutput, "MeV/c")
        ),
        "r_perp" => Some(
            (Box::new(r_perp) as ParticleOutput, "1")
        ),
        "p^-" | "p-" => Some(
            (Box::new(p_minus) as ParticleOutput, "MeV/c")
        ),
        "p^+" | "p+" => Some(
            (Box::new(p_plus) as ParticleOutput, "MeV/c")
        ),
        "gamma" => Some(
            (Box::new(gamma) as ParticleOutput, "1")
        ),
        "energy" => Some(
            (Box::new(energy) as ParticleOutput, "MeV")
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
