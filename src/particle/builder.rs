use std::f64::consts;
use rand::prelude::*;
use rand_distr::{Exp1, StandardNormal};
use crate::geometry::{ThreeVector, FourVector};
use super::{Species, Particle};

#[derive(Copy,Clone)]
pub struct BeamBuilder {
    species: Species,
    num: usize,
    weight: f64,
    normal_espec: Option<bool>,
    gamma: f64,
    sigma: f64,
    gamma_min: f64,
    gamma_max: f64,
    normally_distributed: bool,
    sigma_x: f64,
    sigma_y: f64,
    sigma_z: f64,
    energy_chirp: f64,
    r_max: f64,
    angle: f64,
    rms_div: f64,
    initial_z: f64,
    offset: ThreeVector,
}

impl BeamBuilder {
    pub fn new(species: Species, num: usize, initial_z: f64) -> Self {
        BeamBuilder {
            species,
            num,
            weight: 1.0,
            normal_espec: None,
            gamma: 0.0,
            sigma: 0.0,
            gamma_min: 0.0,
            gamma_max: 0.0,
            normally_distributed: true,
            sigma_x: 0.0,
            sigma_y: 0.0,
            sigma_z: 0.0,
            energy_chirp: 0.0,
            r_max: 0.0,
            angle: 0.0,
            rms_div: 0.0,
            initial_z,
            offset: ThreeVector::new(0.0, 0.0, 0.0),
        }
    }

    pub fn with_weight(&self, weight: f64) -> Self {
        BeamBuilder {
            weight,
            ..*self
        }
    }

    pub fn with_normal_energy_spectrum(&self, gamma: f64, sigma: f64) -> Self {
        BeamBuilder {
            normal_espec: Some(true),
            gamma,
            sigma,
            ..*self
        }
    }

    pub fn with_bremsstrahlung_spectrum(&self, gamma_min: f64, gamma_max: f64) -> Self {
        BeamBuilder {
            normal_espec: Some(false),
            gamma_min,
            gamma_max,
            ..*self
        }
    }

    pub fn with_divergence(&self, rms_div: f64) -> Self {
        BeamBuilder {
            rms_div,
            ..*self
        }
    }

    pub fn with_collision_angle(&self, angle: f64) -> Self {
        BeamBuilder {
            angle,
            ..*self
        }
    }

    pub fn with_normally_distributed_xy(&self, sigma_x: f64, sigma_y: f64) -> Self {
        BeamBuilder {
            normally_distributed: true,
            sigma_x,
            sigma_y,
            ..*self
        }
    }

    pub fn with_uniformly_distributed_xy(&self, r_max: f64) -> Self {
        BeamBuilder {
            normally_distributed: false,
            r_max,
            ..*self
        }
    }

    pub fn with_length(&self, sigma_z: f64) -> Self {
        BeamBuilder {
            sigma_z,
            ..*self
        }
    }

    pub fn with_offset(&self, offset: ThreeVector) -> Self {
        let offset = ThreeVector::new(-offset[0], offset[1], -offset[2]);
        BeamBuilder {
            offset,
            ..*self
        }
    }

    pub fn with_energy_chirp(&self, rho: f64) -> Self {
        BeamBuilder {
            energy_chirp: rho,
            ..*self
        }
    }

    pub fn build<R: Rng>(&self, rng: &mut R) -> Vec<Particle> {
        let normal_espec = self.normal_espec.expect("primary energy spectrum not specified");
        (0..self.num).into_iter()
            .map(|i| {
                // Sample gamma from relevant distribution
                let (gamma, dz) = if normal_espec {
                    loop {
                        // for correlated gamma and z
                        let rho = -self.energy_chirp;
                        let n0 = rng.sample::<f64,_>(StandardNormal);
                        let n1 = rng.sample::<f64,_>(StandardNormal);
                        let n2 = rho * n0 + (1.0 - rho * rho).sqrt() * n1;

                        let dz = self.sigma_z * n0;
                        let gamma = self.gamma + self.sigma * n2;
                        if gamma > 1.0 {
                            break (gamma, dz);
                        }
                    }
                } else { // brem spec
                    let x_min = self.gamma_min / self.gamma_max;
                    let y_max = 4.0 / (3.0 * x_min) - 4.0 / 3.0 + x_min;
                    let x = loop {
                        let x = x_min + (1.0 - x_min) * rng.gen::<f64>();
                        let u = rng.gen::<f64>();
                        let y = 4.0 / (3.0 * x) - 4.0 / 3.0 + x;
                        if u <= y / y_max {
                            break x;
                        }
                    };

                    let dz = self.sigma_z * rng.sample::<f64,_>(StandardNormal);
                    (x * self.gamma_max, dz)
                };

                let u = match self.species {
                    Species::Electron | Species::Positron => -(gamma * gamma - 1.0).sqrt(),
                    Species::Photon => -gamma,
                };

                let theta_x = self.angle + self.rms_div * rng.sample::<f64,_>(StandardNormal);
                let theta_y = self.rms_div * rng.sample::<f64,_>(StandardNormal);

                let u = match self.species {
                    Species::Electron | Species::Positron => FourVector::new(0.0, u * theta_x.sin() * theta_y.cos(), u * theta_y.sin(), u * theta_x.cos() * theta_y.cos()).unitize(),
                    Species::Photon => FourVector::lightlike(u * theta_x.sin() * theta_y.cos(), u * theta_y.sin(), u * theta_x.cos() * theta_y.cos()),
                };

                let t = -self.initial_z;
                let z = self.initial_z + dz;
                let (x, y) = if self.normally_distributed {
                    (
                        self.sigma_x * rng.sample::<f64,_>(StandardNormal),
                        self.sigma_y * rng.sample::<f64,_>(StandardNormal)
                    )
                } else { // uniformly distributed
                    let r = self.r_max * rng.gen::<f64>().sqrt();
                    let theta = 2.0 * consts::PI * rng.gen::<f64>();
                    (r * theta.cos(), r * theta.sin())
                };

                let r = ThreeVector::new(x, y, z);
                let r = r + self.offset;
                let r = r.rotate_around_y(self.angle);
                let r = FourVector::new(t, r[0], r[1], r[2]);

                Particle::create(self.species, r)
                    .with_normalized_momentum(u)
                    .with_optical_depth(rng.sample(Exp1))
                    .with_weight(self.weight)
                    .with_id(i as u64)
                    .with_parent_id(i as u64)
            })
        .collect()
    }
}