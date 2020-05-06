//! Vectors, matrices etc.

/// A four-vector
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourVector(f64, f64, f64, f64);

impl FourVector {
    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        FourVector {0: t, 1: x, 2: y, 3: z}
    }
        
    pub fn lightlike(x: f64, y: f64, z: f64) -> Self {
        FourVector {
            0: (x.powi(2) + y.powi(2) + z.powi(2)).sqrt(),
            1: x,
            2: y,
            3: z
        }
    }

    pub fn sqr(&self) -> f64 {
        self * self
    }

    /// Returns a new four vector `s` that has unit length, i.e. `s * s == 1`,
    /// but unchanged spatial components
    pub fn unitize(&self) -> Self {
        FourVector {
            0: (1.0 + self.1.powi(2) + self.2.powi(2) + self.3.powi(2)).sqrt(),
            1: self.1,
            2: self.2,
            3: self.3,
        }
    }

    /// Returns a new four vector `s` that has norm squared `b`, i.e. `s * s == b`,
    /// but unchanged spatial components
    pub fn with_sqr(&self, b: f64) -> Self {
        FourVector {
            0: (b + self.1.powi(2) + self.2.powi(2) + self.3.powi(2)).sqrt(),
            1: self.1,
            2: self.2,
            3: self.3,
        }
    }

    /// Returns the squared norm of the four-vector
    pub fn norm_sqr(self) -> f64 {
        self * self
    }

    /// Returns the equivalent four vector in a new inertial frame,
    /// which is travelling with four-velocity `u` with respect to
    /// the current frame.
    ///
    /// `u` is expected to be normalized (i.e. gamma v / c)
    pub fn boost_by(self, u: FourVector) -> Self {
        let gamma = u[0];
        let beta = (1.0 - 1.0 / (gamma * gamma)).sqrt();
        let n = ThreeVector::from(u).normalize();
        let a = self[0];
        let z = ThreeVector::from(self);
        FourVector {
            0: gamma * (a - beta * (n * z)),
            1: z[0] + (gamma - 1.0) * (n * z) * n[0] - gamma * beta * a * n[0],
            2: z[1] + (gamma - 1.0) * (n * z) * n[1] - gamma * beta * a * n[1],
            3: z[2] + (gamma - 1.0) * (n * z) * n[2] - gamma * beta * a * n[2]
        }
    }

    /// Reverses the spatial components of the four-vector
    pub fn reverse(self) -> Self {
        FourVector {0: self.0, 1: -self.1, 2: -self.2, 3: -self.3}
    }
}

// Index into four vector
impl std::ops::Index<i32> for FourVector {
    type Output = f64;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            3 => &self.3,
            _ => panic!("index out of bounds: a four vector has 4 components but the index is {}", index)
        }
    }
}

// Add two four vectors together
impl std::ops::Add for FourVector {
    type Output = FourVector;
    fn add(self, other: FourVector) -> FourVector {
        FourVector {
            0: self.0 + other.0,
            1: self.1 + other.1,
            2: self.2 + other.2,
            3: self.3 + other.3
        }
    }
}

// Subtract two four vectors
impl std::ops::Sub for FourVector {
    type Output = FourVector;
    fn sub(self, other: FourVector) -> FourVector {
        FourVector {
            0: self.0 - other.0,
            1: self.1 - other.1,
            2: self.2 - other.2,
            3: self.3 - other.3
        }
    }
}

// Multiply (i.e. dot) two four vectors together
impl std::ops::Mul for FourVector {
    type Output = f64;
    fn mul(self, other: FourVector) -> f64 {
        self.0 * other.0 - self.1 * other.1 - self.2 * other.2 - self.3 * other.3
    }
}

impl std::ops::Mul for &FourVector {
    type Output = f64;
    fn mul(self, other: &FourVector) -> f64 {
        self.0 * other.0 - self.1 * other.1 - self.2 * other.2 - self.3 * other.3
    }
}

// Multiply a four vector by a scalar
impl std::ops::Mul<f64> for FourVector {
    type Output = FourVector;
    fn mul (self, other: f64) -> FourVector {
        FourVector {
            0: self.0 * other,
            1: self.1 * other,
            2: self.2 * other,
            3: self.3 * other
        }
    }
}

// and multiply a scalar by a four vector
impl std::ops::Mul<FourVector> for f64 {
    type Output = FourVector;
    fn mul(self, other: FourVector) -> FourVector {
        FourVector {
            0: self * other.0,
            1: self * other.1,
            2: self * other.2,
            3: self * other.3
        }
    }
}

impl std::ops::Neg for FourVector {
    type Output = FourVector;
    fn neg(self) -> FourVector {
        -1.0 * self
    }
}

// Divide four vector by scalar. Other way round doesn't exist.
impl std::ops::Div<f64> for FourVector {
    type Output = FourVector;
    fn div(self, other: f64) -> FourVector {
        FourVector {
            0: self.0 / other,
            1: self.1 / other,
            2: self.2 / other,
            3: self.3 / other
        }
    }
}

impl std::fmt::Display for FourVector {
    fn fmt(&self, f : &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {} {}", self.0, self.1, self.2, self.3)
    }
}

/// A three-vector
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThreeVector {
    x: f64,
    y: f64,
    z: f64,
}

impl ThreeVector {
    pub fn new(x: f64, y: f64, z: f64) -> ThreeVector {
        ThreeVector{x: x, y: y, z: z}
    }

    pub fn new_from_slice(a: &[f64; 3]) -> ThreeVector {
        ThreeVector{x: a[0], y: a[1], z: a[2]}
    }

    pub fn cross(self, other: ThreeVector) -> ThreeVector {
        ThreeVector {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn norm_sqr(self) -> f64 {
        self * self
    }

    pub fn normalize(self) -> Self {
        let mag = self.norm_sqr().sqrt();
        assert!(mag > 0.0);
        self / mag
    }

    pub fn orthogonal(self) -> Self {
        let perp = if self.x.abs() > self.z.abs() {
            ThreeVector::new(-self.y, self.x, 0.0)
        } else {
            ThreeVector::new(0.0, -self.z, self.y)
        };
        perp.normalize()
    }

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
            x: fv.1,
            y: fv.2,
            z: fv.3,
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

#[cfg(test)]
mod tests {
    use std::f64::consts;
    use super::*; // import from outer scope

    #[test]
    fn boost_fv() {
        let p = FourVector::new(1.0, 0.0, 0.0, 0.0);
        let u = FourVector::new(0.0, 0.0, 50.0, 0.0).unitize();
        let p_prime = p.boost_by(u);
        let err = (p.norm_sqr() - p_prime.norm_sqr()).abs();
        println!("u = [{}], p_prime = [{}], p_prime^2 = [{}], err = {:e}", u, p_prime, p_prime.norm_sqr(), err);
        assert!(err < 1.0e-9);
    }
    
    #[test]
    fn add_fv() {
        let a = FourVector::new(5.0, 3.0, 4.0, 0.0);
        let b = FourVector::new(15.0, 14.0, 5.0, 2.0);
        assert_eq!(a + b, FourVector::new(20.0, 17.0, 9.0, 2.0));
    }

    #[test]
    fn norm() {
        let a = FourVector::new(5.0, 3.0, 4.0, 0.0);
        let b = FourVector::new(15.0, 14.0, 5.0, 2.0);
        assert!(a.sqr().abs() < 1.0e-10);
        assert!(b.sqr().abs() < 1.0e-10);
    }

    #[test]
    fn lightlike_is_null() {
        let a = FourVector::lightlike(1.0, -17.0, 2.6);
        assert!(a.sqr().abs() < 1.0e-10);
    }

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