use std::f64::consts;
use rand::prelude::*;
use rand_distr::StandardNormal;
use crate::geometry::{ThreeVector, FourVector, StokesVector};
use super::{Species, Particle};
use super::dstr::{GammaDistribution, SpatialDistribution};

#[derive(Clone)]
pub struct BeamBuilder {
    species: Species,
    num: usize,
    pub weight: f64,
    gamma_dstr: GammaDistribution,
    r_dstr: SpatialDistribution,
    z_dstr: SpatialDistribution,
    angle: f64,
    collision_plane_angle: f64,
    pub rms_div: f64,
    initial_z: f64,
    offset: ThreeVector,
    pub pol: StokesVector,
}

impl BeamBuilder {
    pub fn new(species: Species, num: usize, gamma_dstr: GammaDistribution) -> Self {
        BeamBuilder {
            species,
            num,
            weight: 1.0,
            // gamma_dstr: GammaDistribution::Normal { mu: 1.0, sigma: 0.0, rho: 0.0 },
            gamma_dstr,
            r_dstr: SpatialDistribution::Normal { sigma: 0.0, max: None, dim: 2 },
            z_dstr: SpatialDistribution::Normal { sigma: 0.0, max: None, dim: 1 },
            angle: 0.0,
            collision_plane_angle: 0.0,
            rms_div: 0.0,
            initial_z: 0.0,
            offset: ThreeVector::new(0.0, 0.0, 0.0),
            pol: StokesVector::unpolarized(),
        }
    }

    pub fn with_initial_z(self, initial_z: f64) -> Self {
        BeamBuilder {
            initial_z,
            ..self
        }
    }

    pub fn with_weight(self, weight: f64) -> Self {
        BeamBuilder {
            weight,
            ..self
        }
    }

    pub fn with_divergence(self, rms_div: f64) -> Self {
        BeamBuilder {
            rms_div,
            ..self
        }
    }

    pub fn with_collision_angle(self, angle: f64) -> Self {
        BeamBuilder {
            angle,
            ..self
        }
    }

    pub fn with_collision_plane_at(self, angle: f64) -> Self {
        BeamBuilder {
            collision_plane_angle: angle,
            ..self
        }
    }

    pub fn with_transverse_dstr(self, dstr: SpatialDistribution) -> Self {
        BeamBuilder {
            r_dstr: dstr,
            ..self
        }
    }

    pub fn with_longitudinal_dstr(self, dstr: SpatialDistribution) -> Self {
        BeamBuilder {
            z_dstr: dstr,
            ..self
        }
    }

    pub fn with_offset(self, offset: ThreeVector) -> Self {
        BeamBuilder {
            offset,
            ..self
        }
    }

    pub fn with_energy_chirp(self, energy_chirp: f64) -> Self {
        // note sign change!
        let gamma_dstr = self.gamma_dstr.with_correlation_coeff(-energy_chirp);

        BeamBuilder {
            gamma_dstr,
            ..self
        }
    }

    pub fn with_polarization(self, sv: StokesVector) -> Self {
        BeamBuilder {
            pol: sv,
            ..self
        }
    }

    #[cfg(feature = "hdf5-output")]
    pub fn transverse_dstr_is_normal(&self) -> bool {
        matches!(self.r_dstr, SpatialDistribution::Normal { .. })
    }

    #[cfg(feature = "hdf5-output")]
    pub fn has_brem_spec(&self) -> bool {
        match self.gamma_dstr {
            GammaDistribution::Brem { .. } => true,
            _ => false
        }
    }

    #[cfg(feature = "hdf5-output")]
    pub fn radius(&self) -> (f64, f64) {
        match self.r_dstr {
            SpatialDistribution::Normal { sigma, max, dim: _ } => (sigma, max.unwrap_or(std::f64::INFINITY)),
            SpatialDistribution::Disk { max, dim: _ } => (max, max),
        }
    }

    /// Root mean square displacement of a particle from the beam centroid, along the beam propagation axis
    pub fn sigma_z(&self) -> f64 {
        self.z_dstr.std_dev()
    }

    pub fn gamma(&self) -> f64 {
        self.gamma_dstr.gamma()
    }

    #[cfg(feature = "hdf5-output")]
    pub fn sigma(&self) -> f64 {
        self.gamma_dstr.std_dev()
    }

    #[cfg(feature = "hdf5-output")]
    pub fn gamma_min(&self) -> f64 {
        self.gamma_dstr.min_gamma()
    }

    pub fn build<R: Rng>(&self, rng: &mut R) -> Vec<Particle> {
        // let normal_espec = self.normal_espec.expect("primary energy spectrum not specified");
        (0..self.num).into_iter()
            .map(|i| {
                // Sample gamma from relevant distribution
                let (gamma, dz) = self.gamma_dstr.sample(&self.z_dstr, rng);

                let u = match self.species {
                    Species::Electron | Species::Positron => -(gamma * gamma - 1.0).sqrt(),
                    Species::Photon => -gamma,
                };

                let theta_x = self.angle + self.rms_div * rng.sample::<f64,_>(StandardNormal);
                let theta_y = self.rms_div * rng.sample::<f64,_>(StandardNormal);

                let u = ThreeVector::new(u * theta_x.sin() * theta_y.cos(), u * theta_y.sin(), u * theta_x.cos() * theta_y.cos());
                let u = u.rotate_around_z(self.collision_plane_angle);
                let u = match self.species {
                    Species::Electron | Species::Positron => FourVector::new(0.0, u[0], u[1], u[2]).unitize(),
                    Species::Photon => FourVector::lightlike(u[0], u[1], u[2]),
                };

                let r = self.r_dstr.sample(rng.gen());
                let theta = 2.0 * consts::PI * rng.gen::<f64>();

                let r = ThreeVector::new(r * theta.cos(), r * theta.sin(), dz) + self.offset;
                let r = r.rotate_around_y(self.angle);
                let r = r.rotate_around_z(self.collision_plane_angle);
                let r = r.with_time(0.0);

                // Displace particles to starting point:
                // Initialise particle at r^- = t - z = -2 z0, using r(t) = r(0) + beta * t
                let r = {
                    let beta = u / u[0];
                    r - (2.0 * self.initial_z + r[0] - r[3]) * beta / (beta[0] - beta[3])
                };

                Particle::create(self.species, r)
                    .with_normalized_momentum(u)
                    .with_polarization(self.pol)
                    .with_weight(self.weight)
                    .with_id(i as u64)
                    .with_parent_id(i as u64)
            })
        .collect()
    }
}