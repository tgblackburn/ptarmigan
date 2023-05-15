//! Defines a polarization state

use num_complex::Complex;
use super::ThreeVector;
use super::FourVector;

#[cfg(feature = "hdf5-output")]
use hdf5_writer::{Hdf5Type, Datatype};

/// A set of Stokes parameters
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct StokesVector {
    i: f64,
    q: f64,
    u: f64,
    v: f64,
}

#[cfg(feature = "hdf5-output")]
impl Hdf5Type for StokesVector {
    fn new() -> Datatype {
        Datatype::array::<f64>(4)
    }
}

impl StokesVector {
    /// Creates a new Stokes vector with the specified components.
    pub fn new(i: f64, q: f64, u: f64, v: f64) -> Self {
        Self{i, q, u, v}
    }

    pub fn unpolarized() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Degree of polarization
    pub fn dop(&self) -> f64 {
        self.q.hypot(self.u).hypot(self.v) / self.i
    }

    /// Returns the Stokes vector if the polarization basis is rotated
    /// around the direction of propagation by an angle `theta`
    pub fn rotate_by(&self, theta: f64) -> Self {
        Self {
            i: self.i,
            q: (2.0 * theta).cos() * self.q + (2.0 * theta).sin() * self.u,
            u: -(2.0 * theta).sin() * self.q + (2.0 * theta).cos() * self.u,
            v: self.v,
        }
    }

    /// Returns the Stokes parameters if the basis is redefined in terms of
    /// the principal axis `a`.
    ///
    /// `a` must be perpendicular to particle propagation direction `k`!
    pub fn in_basis(&self, a: ThreeVector, k: ThreeVector) -> (StokesVector, f64, f64) {
        // In the standard basis, e_1 is guaranteed to lie in the x-z
        // plane and to be perpendicular to the propagation direction.
        // So we need to rotate the basis by the angle
        let k_mag = k[0].hypot(k[2]);
        let a_mag = a.norm_sqr().sqrt();
        let cos_theta = if k_mag == 0.0 {
            // photon pointed along y => e_1 = (1,0,0)
            a[0] / a_mag
        } else {
            // e_1 = (k3, 0, -k1) / |k|
            (k[2] * a[0] - k[0] * a[2]) / (k_mag * a_mag)

        };

        // rotating from e_1 to a, RH or LH?
        // calculate k.(e_1 x a), if positive, RH
        let det = if k_mag == 0.0 {
            -k[1] * a[2] + k[2] * a[1]
        } else {
            a[1] * (k[0].powi(2) + k[2].powi(2)) - k[1] * (a[0] * k[0] + a[2] * k[2])
        };

        // if positive, rotate basis by +theta and Stokes parameters by -2 theta
        let sign = -det.signum();

        let cos_theta = cos_theta.clamp(-1.0, 1.0);
        let sin_theta = sign * (1.0 - cos_theta * cos_theta).sqrt();
        let cos_2theta = cos_theta * cos_theta - sin_theta * sin_theta;
        let sin_2theta = 2.0 * cos_theta * sin_theta;

        let sv = Self {
            i: self.i,
            q: cos_2theta * self.q - sin_2theta * self.u,
            u: sin_2theta * self.q + cos_2theta * self.u,
            v: self.v,
        };

        (sv, cos_2theta, sin_2theta)
    }

    /// Returns the sine and cosine of the angle that rotates the LMA basis
    /// vector e_1 into the global basis vector.
    fn lma_basis_rotation(k: FourVector) -> (f64, f64) {
        let k_minus = k[0] - k[3];
        let cos_theta = (k[3] * k_minus - k[1].powi(2)) / (k_minus * k[1].hypot(k[3]));

        // det = k · (e_LMA × e_g), ignoring positive factors as we only need the sign
        let det = k[1] * k[2]; // k[0] * k[1] * k[2] / (k_minus * k[1].hypot(k[3]))
        let sign = det.signum();

        let cos_theta = cos_theta.clamp(-1.0, 1.0);
        let sin_theta = sign * (1.0 - cos_theta * cos_theta).sqrt();

        (sin_theta, cos_theta)
    }

    /// Returns the Stokes parameters in the standard LMA basis.
    /// See [`from_lma_basis`](Self::from_lma_basis).
    pub fn in_lma_basis(&self, k: FourVector) -> StokesVector {
        // RH angle that rotates LMA basis to global basis...
        let (sin_theta, cos_theta) = Self::lma_basis_rotation(k);
        // Rotate Stokes parameters by 2 theta...
        let cos_2theta = cos_theta * cos_theta - sin_theta * sin_theta;
        let sin_2theta = 2.0 * cos_theta * sin_theta;

        Self {
            i: self.i,
            q: cos_2theta * self.q - sin_2theta * self.u,
            u: sin_2theta * self.q + cos_2theta * self.u,
            v: self.v,
        }
    }

    /// Returns the Stokes parameters in the global basis, where e_1 is
    /// in the x-z plane:
    /// ```text
    ///   e_1_global = (k'_z, 0, -k'_x) / norm
    /// ```
    /// In the LMA, the Stokes parameters are defined w.r.t. to the orthonormal basis
    /// ```text
    ///   e_1 = x - k'_x (k' - omega' z) / (omega' * (omega' - k'_z))
    /// ```
    pub fn from_lma_basis(&self, k: FourVector) -> StokesVector {
        // RH angle that rotates LMA basis to global basis...
        let (sin_theta, cos_theta) = Self::lma_basis_rotation(k);
        // Rotate Stokes parameters by -2 theta...
        let sin_theta = -sin_theta;
        let cos_2theta = cos_theta * cos_theta - sin_theta * sin_theta;
        let sin_2theta = 2.0 * cos_theta * sin_theta;

        Self {
            i: self.i,
            q: cos_2theta * self.q - sin_2theta * self.u,
            u: sin_2theta * self.q + cos_2theta * self.u,
            v: self.v,
        }
    }

    /// Projects the polarization of a particle, travking along `dir`,
    /// onto the given `axis`
    pub fn project_onto(&self, dir: ThreeVector, axis: ThreeVector) -> f64 {
        // Degree of polarization
        let frac = self.dop();

        // Fix normalisation! (sv[0] should be 1)
        let sv = *self / (frac * self.i);

        // Convert Stokes vector to Jones vector
        let ex = if 1.0 + sv[1] < 1.0e-12 {
            // 1 + q ~= 0, use q^2 + u^2 + v^2 = 1 and u, v << 1
            let x_sqd = sv[2] * sv[2] + sv[3] * sv[3];
            let delta = 0.5 * x_sqd + 0.125 * x_sqd * x_sqd;
            (0.5 * delta).sqrt()
        } else {
            (0.5 * (1.0 + sv[1])).sqrt()
        };

        let ey = if ex == 0.0 {
            Complex::new(1.0, 0.0)
        } else {
            Complex::new(0.5 * sv[2] / ex, -0.5 * sv[3] / ex)
        };

        let ex = Complex::new(ex, 0.0);
        //println!("\tJones vector = {}, {}", ex, ey);

        // Construct axes of the ellipse
        let (x, y) = {
            let n = dir;
            let mag = n[0].hypot(n[2]);
            let x: ThreeVector = if mag == 0.0 {
                // so photon pointed along y
                [1.0, 0.0, 0.0].into()
            } else {
                [n[2] / mag, 0.0, -n[0] / mag].into()
            };
            let y = x.cross(n);
            (x, y)
        };
        //println!("\tx = {}, y = {}, x.y = {}", x, y, x * y);

        // Project intensities
        let axis = axis.normalize();
        //println!("|ex|^2 = {:.2e}, ex.axis = {:.2e}, |ey|^2 = {:.2e}, ey.axis = {:.2e}", ex.norm_sqr(), x * axis, ey.norm_sqr(), y * axis);

        // Polarized and unpolarized contributions
        let pol_contr = (ex * (x * axis) + ey * (y * axis)).norm_sqr();
        let unpol_contr = 0.5 * ((x * axis).powi(2) + (y * axis).powi(2));

        if frac > 0.0 {
            frac * pol_contr + (1.0 - frac) * unpol_contr
        } else {
            unpol_contr
        }
    }
}

impl std::ops::Index<i32> for StokesVector {
    type Output = f64;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &self.i,
            1 => &self.q,
            2 => &self.u,
            3 => &self.v,
            _ => panic!("index out of bounds: a Stokes vector has 4 components but the index is {}", index)
        }
    }
}

impl std::ops::Add for StokesVector {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        StokesVector {
            i: self.i + other.i,
            q: self.q + other.q,
            u: self.u + other.u,
            v: self.v + other.v
        }
    }
}

impl std::ops::Div<f64> for StokesVector {
    type Output = Self;
    fn div(self, other: f64) -> Self {
        Self {
            i: self.i / other,
            q: self.q / other,
            u: self.u / other,
            v: self.v / other,
        }
    }
}

impl std::convert::From<[f64; 4]> for StokesVector {
    fn from(item: [f64; 4]) -> Self {
        Self::new(item[0], item[1], item[2], item[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_is_normalized() {
        let err = 2.8e-8_f64;
        let sv: StokesVector = [1.0, -(1.0 - err * err).sqrt(), err, 0.0].into();
        let dir: ThreeVector = {
            let x = 0.000126978;
            let y = -0.000220558;
            let z = -(1.0_f64 - x * x - y * y).sqrt();
            [x, y, z].into()
        };
        let pol_x = sv.project_onto(dir, [1.0, 0.0, 0.0].into());
        let pol_y = sv.project_onto(dir, [0.0, 1.0, 0.0].into());
        println!("weight = {} + {} = {}, dir = {}", pol_x, pol_y, pol_x + pol_y, dir);
        assert!(pol_x + pol_y <= 1.0);
    }
}