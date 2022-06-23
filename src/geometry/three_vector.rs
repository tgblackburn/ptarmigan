//! Defines a spatial 3-vector: (x, y, z)

#[cfg(feature = "hdf5-output")]
use hdf5_writer::{Hdf5Type, Datatype};

use super::FourVector;

/// A three-vector
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThreeVector {
    x: f64,
    y: f64,
    z: f64,
}

#[cfg(feature = "hdf5-output")]
impl Hdf5Type for ThreeVector {
    fn new() -> Datatype {
        Datatype::array::<f64>(3)
    }
}

impl ThreeVector {
    /// Creates a new three-vector with the specified components.
    pub fn new(x: f64, y: f64, z: f64) -> ThreeVector {
        ThreeVector{x: x, y: y, z: z}
    }

    /// Creates a new three-vector with the specified components.
    pub fn new_from_slice(a: &[f64; 3]) -> ThreeVector {
        ThreeVector{x: a[0], y: a[1], z: a[2]}
    }

    /// Returns the cross product of two three-vectors.
    pub fn cross(self, other: ThreeVector) -> ThreeVector {
        ThreeVector {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns the squared magnitude of the three-vector.
    pub fn norm_sqr(self) -> f64 {
        self * self
    }

    /// Returns a new four-vector which has the same direction,
    /// but unit magnitude.
    ///
    /// # Panics
    /// If `self` does not have positive definite norm.
    pub fn normalize(self) -> Self {
        let mag = self.norm_sqr().sqrt();
        assert!(mag > 0.0);
        self / mag
    }

    /// Returns a unit vector that is orthogonal to `self`.
    /// The choice is arbitrary, but fixed for a given input.
    /// ```
    /// let a = ThreeVector::new(0.0, 1.0, 2.0);
    /// let b = a.orthogonal();
    /// assert!(a * b < 1.0e-10);
    /// ```
    pub fn orthogonal(self) -> Self {
        let perp = if self.x.abs() > self.z.abs() {
            ThreeVector::new(-self.y, self.x, 0.0)
        } else {
            ThreeVector::new(0.0, -self.z, self.y)
        };
        perp.normalize()
    }

    /// Rotates `self` around the given `axis` by an angle `theta`,
    /// with positive angles corresponding to a right-handed rotation,
    /// and returns the result. The axis must be correctly normalized.
    /// `rotate_around_{x,y,z}` are provided for convenience.
    pub fn rotate_around(self, axis: ThreeVector, theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        let out = ThreeVector::new(
            (c + axis.x * axis.x * (1.0-c)) * self.x
	        + (axis.x * axis.y * (1.0-c) - axis.z * s) * self.y
            + (axis.x * axis.z * (1.0-c) + axis.y * s) * self.z,
            (axis.y * axis.x * (1.0-c) + axis.z * s) * self.x
	        + (c + axis.y * axis.y * (1.0-c)) * self.y
            + (axis.y * axis.z * (1.0-c) - axis.x * s) * self.z,
            (axis.z * axis.x * (1.0-c) - axis.y * s) * self.x
	        + (axis.z * axis.y * (1.0-c) + axis.x * s) * self.y
	        + (c + axis.z * axis.z * (1.0-c)) * self.z
        );
        out
    }

    /// Rotates `self` around the x-axis by angle `theta` and returns
    /// the result.
    pub fn rotate_around_x(self, theta: f64) -> Self {
        self.rotate_around(ThreeVector::new(1.0, 0.0, 0.0), theta)
    }

    /// Rotates `self` around the y-axis by angle `theta` and returns
    /// the result.
    pub fn rotate_around_y(self, theta: f64) -> Self {
        self.rotate_around(ThreeVector::new(0.0, 1.0, 0.0), theta)
    }

    /// Rotates `self` around the z-axis by angle `theta` and returns
    /// the result.
    pub fn rotate_around_z(self, theta: f64) -> Self {
        self.rotate_around(ThreeVector::new(0.0, 0.0, 1.0), theta)
    }
}

impl std::ops::Index<i32> for ThreeVector {
    type Output = f64;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds: a three vector has 3 components but the index is {}", index)
        }
    }
}

impl std::convert::From<FourVector> for ThreeVector {
    fn from(fv: FourVector) -> Self {
        ThreeVector {
            x: fv[1],
            y: fv[2],
            z: fv[3],
        }
    }
}

impl std::fmt::Display for ThreeVector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

impl std::ops::Add for ThreeVector {
    type Output = ThreeVector;
    fn add(self, other: ThreeVector) -> ThreeVector {
        ThreeVector {x: self.x + other.x, y: self.y + other.y, z: self.z + other.z}
    }
}

impl std::ops::Sub for ThreeVector {
    type Output = ThreeVector;
    fn sub(self, other: ThreeVector) -> ThreeVector {
        ThreeVector {x: self.x - other.x, y: self.y - other.y, z: self.z - other.z}
    }
}

impl std::ops::Mul for ThreeVector {
    type Output = f64;
    fn mul(self, other: ThreeVector) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl std::ops::Mul<f64> for ThreeVector {
    type Output = ThreeVector;
    fn mul(self, other: f64) -> ThreeVector {
        ThreeVector{x: self.x * other, y: self.y * other, z: self.z * other}
    }
}

impl std::ops::Mul<ThreeVector> for f64 {
    type Output = ThreeVector;
    fn mul(self, other: ThreeVector) -> ThreeVector {
        ThreeVector{x: self * other.x, y: self * other.y, z: self * other.z}
    }
}

impl std::ops::Neg for ThreeVector {
    type Output = ThreeVector;
    fn neg(self) -> ThreeVector {
        -1.0 * self
    }
}

impl std::ops::Div<f64> for ThreeVector {
    type Output = ThreeVector;
    fn div(self, other: f64) -> ThreeVector {
        ThreeVector{x: self.x / other, y: self.y / other, z: self.z / other}
    }
}

impl std::convert::From<[f64; 3]> for ThreeVector {
    fn from(item: [f64; 3]) -> Self {
        ThreeVector::new(item[0], item[1], item[2])
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts;
    use super::*; // import from outer scope

    #[test]
    fn orthogonality() {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let a = ThreeVector::new(rng.gen(), rng.gen(), rng.gen());
        let b = a.orthogonal();
        println!("a = {:?}, b = {:?}, a.b = {}", a, b, a*b);
        assert!(a*b < 1.0e-10);
    }

    #[test]
    fn rotation() {
        let v = ThreeVector::new(1.0, 0.0, 0.0); // along x
        let v = v.rotate_around(ThreeVector::new(0.0, 0.0, 1.0), consts::FRAC_PI_2); // along y
        let v = v.rotate_around(ThreeVector::new(1.0, 0.0, 0.0), consts::FRAC_PI_2); // along z
        println!("v = {:?}", v);
        let target = ThreeVector::new(0.0, 0.0, 1.0);
        assert!((v - target).norm_sqr().sqrt() < 1.0e-10);
    }
}
