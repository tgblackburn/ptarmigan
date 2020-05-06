use std::error::Error;
use std::f64::consts;

use mpi::traits::*;
use mpi::Threading;
use rand::prelude::*;
use rand_distr::{Exp1, StandardNormal};
use rand_xoshiro::*;

mod constants;
mod field;
mod geometry;
mod particle;
mod nonlinear_compton;
mod hgram;
mod special_functions;

use constants::*;
use field::*;
use geometry::*;
use hgram::*;
use particle::*;

fn collide<F: Field, R: Rng>(field: &F, incident: Particle, rng: &mut R) -> Shower {
    let mut primary = incident;
    let mut secondaries: Vec<Particle> = Vec::new();
    let dt = field.max_timestep().unwrap_or(1.0);

    while field.contains(primary.position()) {
        let (r, u) = field.push(
            primary.position(), 
            primary.normalized_momentum(),
            primary.charge_to_mass_ratio(),
            dt
        );
        
        if let Some(k) = field.radiate(r, u, dt, rng) {
            let photon = Particle::create(Species::Photon, r)
                .with_normalized_momentum(k);
            secondaries.push(photon);
        }

        primary.with_position(r);
        primary.with_normalized_momentum(u);
    }

    Shower {
        primary,
        secondaries,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (universe, _) = mpi::initialize_with_threading(Threading::Funneled).unwrap();
    let world = universe.world();
    let id = world.rank();
    let numtasks = world.size();

    let a0 = 100.0;
    let wavelength = 0.8e-6;
    let waist = 4.0e-6;
    let duration = 30.0e-15;
    let pol = Polarization::Linear;
    let num: i32 = 100_000;
    let gamma = 1000.0;
    let radius = 1.0e-6;

    let mut rng = Xoshiro256StarStar::seed_from_u64(id as u64);

    let primaries: Vec<Particle> = (0..num).into_iter()
        .map(|_i| {
            let z = 2.0 * SPEED_OF_LIGHT * duration;
            let x = radius * rng.sample::<f64,_>(StandardNormal);
            let y = radius * rng.sample::<f64,_>(StandardNormal);
            let r = FourVector::new(-z, x, y, z);
            let u = -(gamma * gamma - 1.0f64).sqrt();
            let u = FourVector::new(0.0, 0.0, 0.0, u).unitize();
            Particle::create(Species::Electron, r)
                .with_normalized_momentum(u)
                .with_optical_depth(rng.sample(Exp1))
        })
        .collect();

    let laser = FocusedLaser::new(a0, wavelength, waist, duration, pol);

    let (electrons, photons) = primaries.iter()
        .map(|pt: &Particle| -> Shower {
            collide(&laser, *pt, &mut rng)
        })
        .fold(
            (Vec::<Particle>::new(), Vec::<Particle>::new()),
            |(mut p, mut s), mut shower| {
                p.push(shower.primary);
                s.append(&mut shower.secondaries);
                (p, s)
        });

    println!("e.len = {}, ph.len = {}", electrons.len(), photons.len());

    let angle_x = |pt: &Particle| {let p = pt.momentum(); p[1].atan2(-p[3])};
    let angle_y = |pt: &Particle| {let p = pt.momentum(); p[2].atan2(-p[3])};
    let unit_weight = |_pt: &Particle| 1.0;

    let angle = Histogram::generate_2d(
        &world,
        &electrons,
        &angle_x,
        &angle_y,
        &unit_weight,
        ["angle_x", "angle_y"],
        ["rad", "rad"],
        [BinSpec::Automatic; 2],
        HeightSpec::Density
    );

    angle.unwrap().write_fits("!output/angle.fits")?;

    //let sci = 70.69 * (consts::PI / (16.0 * consts::LN_2)).sqrt();
    //let expected = a0.powi(2) * sci * r0[1] * wavelength * (-2.0 * r0[1].powi(2) / waist.powi(2)).exp() / (2.0 * consts::PI * u0[3].powi(2) * waist.powi(2));
    Ok(())
}
