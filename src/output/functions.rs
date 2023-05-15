//! Provides possible particle outputs

use crate::constants;
use crate::particle::Particle;
use super::ParticleOutput;
use super::ParticleOutputType;
use super::ParticleOutputType::*;

/// Returns the ParticleOutput and its unit.
pub fn identify(name: &str) -> Option<(ParticleOutput, ParticleOutputType)> {
    match name {
        "angle_x" => Some(
            (angle_x as ParticleOutput, Angle)
        ),
        "angle_y" => Some(
            (angle_y as ParticleOutput, Angle)
        ),
        "theta" | "pi_minus_angle" => Some(
            (theta as ParticleOutput, Angle)
        ),
        "angle" | "polar_angle" => Some(
            (polar_angle as ParticleOutput, Angle)
        ),
        "px" | "p_x" => Some(
            (px as ParticleOutput, Momentum)
        ),
        "py" | "p_y" => Some(
            (py as ParticleOutput, Momentum)
        ),
        "pz" | "p_z" => Some(
            (pz as ParticleOutput, Momentum)
        ),
        "p_perp" => Some(
            (p_perp as ParticleOutput, Momentum)
        ),
        "r_perp" => Some(
            (r_perp as ParticleOutput, Dimensionless)
        ),
        "r_x" | "rx" => Some(
            (r_x as ParticleOutput, Dimensionless)
        ),
        "r_y" | "ry" => Some(
            (r_y as ParticleOutput, Dimensionless)
        ),
        "p^-" | "p-" | "p_minus" => Some(
            (p_minus as ParticleOutput, Momentum)
        ),
        "p^+" | "p+" | "p_plus" => Some(
            (p_plus as ParticleOutput, Momentum)
        ),
        "gamma" => Some(
            (gamma as ParticleOutput, Dimensionless)
        ),
        "energy" => Some(
            (energy as ParticleOutput, Energy)
        ),
        "unit" => Some(
            (unit as ParticleOutput, Dimensionless)
        ),
        "x" => Some(
            (x as ParticleOutput, Length)
        ),
        "y" => Some(
            (y as ParticleOutput, Length)
        ),
        "z" => Some(
            (z as ParticleOutput, Length)
        ),
        "birth_a" => Some(
            (payload as ParticleOutput, Dimensionless)
        ),
        "n_inter" | "n_gamma" | "n_pos" => Some(
            (interaction_count as ParticleOutput, Dimensionless)
        ),
        "weight" | "number" => Some(
            (weighted_by_number as ParticleOutput, Dimensionless)
        ),
        "S_1" | "S1" => Some(
            (stokes_1 as ParticleOutput, Dimensionless)
        ),
        "S_2" | "S2" => Some(
            (stokes_2 as ParticleOutput, Dimensionless)
        ),
        "S_3" | "S3" => Some(
            (stokes_3 as ParticleOutput, Dimensionless)
        ),
        "pol_x" | "pol_sigma" => Some(
            (pol_x as ParticleOutput, Dimensionless)
        ),
        "pol_y" | "pol_pi" => Some(
            (pol_y as ParticleOutput, Dimensionless)
        ),
        "helicity" => Some(
            (weighted_by_helicity as ParticleOutput, Dimensionless)
        ),
        _ => None,
    }
}

pub fn angle_x(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1].atan2(p[3])
}

pub fn angle_y(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[2].atan2(p[3])
}

pub fn theta(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1].hypot(p[2]).atan2(-p[3])
}

pub fn polar_angle(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1].hypot(p[2]).atan2(p[3])
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

pub fn r_x(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[1] / (p[0] - p[3])
}

pub fn r_y(pt: &Particle) -> f64 {
    let p = pt.momentum();
    p[2] / (p[0] - p[3])
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

pub fn stokes_1(pt: &Particle) -> f64 {
    pt.polarization()[1]
}

pub fn stokes_2(pt: &Particle) -> f64 {
    pt.polarization()[2]
}

pub fn stokes_3(pt: &Particle) -> f64 {
    pt.polarization()[3]
}

pub fn pol_x(pt: &Particle) -> f64 {
    pt.polarization_along_x()
}

pub fn pol_y(pt: &Particle) -> f64 {
    pt.polarization_along_y()
}

pub fn weighted_by_energy(pt: &Particle) -> f64 {
    let p = pt.normalized_momentum();
    let energy = p[0] * constants::ELECTRON_MASS_MEV;
    pt.weight() * energy
}

pub fn weighted_by_number(pt: &Particle) -> f64 {
    pt.weight()
}

pub fn weighted_by_pol_x(pt: &Particle) -> f64 {
    pt.polarization_along_x() * pt.weight()
}

pub fn weighted_by_pol_y(pt: &Particle) -> f64 {
    pt.polarization_along_y() * pt.weight()
}

pub fn weighted_by_helicity(pt: &Particle) -> f64 {
    pt.polarization()[3] * pt.weight()
}