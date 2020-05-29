//! Airy functions
//! 
//! Currently implemented:
//! 
//! * Airy function of the first kind, Ai(z), for real,
//!   positive argument.
//! 
//! Algorithms adapted from:
//! 
//! * A. Gil, J. Segura and N. M. Tenne,
//!   "Computing Complex Airy Functions by Numerical Quadrature",
//!   Numerical Algorithms 30, 11--23 (2002)

use super::Series;

pub trait Airy {
    /// Returns the value of the Airy function for real, positive argument.
    /// If `x` is negative, or the result would underflow, the
    /// return value is None.
    fn ai(&self) -> Option<Self> where Self: Sized;

    /// Returns the derivative of the Airy function for real, positive argument.
    /// If `x` is negative, or the result would underflow, the
    /// return value is None.
    fn ai_prime(&self) -> Option<Self> where Self: Sized;
}

impl Airy for f64 {
    fn ai(&self) -> Option<Self> {
        let x = *self;
        if x < 0.0 {
            None
        } else if x < 1.0 {
            // Use Taylor series expansion
            Some(SMALL_X_EXPANSION.evaluate_up_to(x, 14))
        } else if x < 2.0 {
            // Numerically integrate the integal representation
            // the Airy function using 40-point Gauss-Laguerre
            // quadrature.
            //
            // That representation is
            //   Ai(x) = a(x) \int_0^\infty f(t) w(t) dt
            // where the integrand
            //   f(t) = (2 + t/s)^(-1/6),
            // the weight function
            //   w(t) = t^(-1/6) exp(-t),
            // the scale factor
            //   a(x) = s^(-1/6) exp(-s) / (sqrt(pi) (48)^(1/6) Gamma(5/6))
            // and
            //   s = 2 x^(3/2) / 3.
            let s = 2.0 * x.powf(1.5) / 3.0;
            let a = 0.262183997088323 * s.powf(-1.0/6.0) * (-s).exp();
            let integral: f64 = GAUSS_LAGUERRE_40_NODES.iter()
                .zip(GAUSS_LAGUERRE_40_WEIGHTS.iter())
                .map(|(x, w)| w * (2.0 + x / s).powf(-1.0/6.0))
                .sum();
            Some(a * integral)
        } else if x < 10.0 {
            // As above, but using fewer nodes
            let s = 2.0 * x.powf(1.5) / 3.0;
            let a = 0.262183997088323 * s.powf(-1.0/6.0) * (-s).exp();
            let integral: f64 = GAUSS_LAGUERRE_16_NODES.iter()
                .zip(GAUSS_LAGUERRE_16_WEIGHTS.iter())
                .map(|(x, w)| w * (2.0 + x / s).powf(-1.0/6.0))
                .sum();
            Some(a * integral)
        } else if x < 50.0 {
            // As above, but using fewer nodes
            let s = 2.0 * x.powf(1.5) / 3.0;
            let a = 0.262183997088323 * s.powf(-1.0/6.0) * (-s).exp();
            let integral: f64 = GAUSS_LAGUERRE_4_NODES.iter()
                .zip(GAUSS_LAGUERRE_4_WEIGHTS.iter())
                .map(|(x, w)| w * (2.0 + x / s).powf(-1.0/6.0))
                .sum();
            Some(a * integral)
        } else {
            // Ai(50) < 4.5e-104
            None
        }
    }

    fn ai_prime(&self) -> Option<Self> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MAX_REL_ERR: f64 = 1.0e-12;

    #[test]
    fn airy_0() {
        let val = Airy::ai(&0.0).unwrap();
        let target = 0.3550280538878172;
        println!("Ai(0) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn airy_2() {
        let val = Airy::ai(&2.0).unwrap();
        let target = 0.03492413042327438;
        println!("Ai(2) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn airy_17() {
        let val = Airy::ai(&17.0).unwrap();
        let target = 7.05019729838861e-22;
        println!("Ai(17) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn airy_20() {
        let val = Airy::ai(&20.0).unwrap();
        let target = 1.69167286867e-27;
        println!("Ai(20) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    #[should_panic]
    fn airy_200() {
        let _val = Airy::ai(&200.0).unwrap();
    }
}

static SMALL_X_EXPANSION: Series<i32> = Series {
    a: [
		3.550280538878172e-1,
		-2.588194037928068e-1,
		5.917134231463621e-2,
		-2.156828364940057e-2,
		1.972378077154540e-3,
		-5.135305630809659e-4,
		2.739413996047973e-5,
		-5.705895145344065e-6,
		2.075313633369676e-7,
		-3.657625093169273e-8,
		9.882445873188934e-10,
		-1.524010455487197e-10,
		3.229557474898344e-12,
		-4.456170922477184e-13,
		7.689422559281773e-15,
		-9.645391607093472e-16,
		1.393011333203220e-17,
		-1.607565267848912e-18,
		1.984346628494615e-20,
		-2.126409084456233e-21,
    ],
    n: [
        0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28
    ],
};
 
static GAUSS_LAGUERRE_40_NODES: [f64; 40] = [
    2.838914179945677e-2,
    1.709853788600349e-1,
    4.358716783417705e-1,
    8.235182579130309e-1,
    1.334525432542274e+0,
    1.969682932064351e+0,
    2.729981340028599e+0,
    3.616621619161009e+0,
    4.631026110526541e+0,
    5.774851718305477e+0,
    7.050005686302187e+0,
    8.458664375132378e+0,
    1.000329552427494e+1,
    1.168668459477224e+1,
    1.351196593446936e+1,
    1.548265969593771e+1,
    1.760271568080691e+1,
    1.987656560227855e+1,
    2.230918567739628e+1,
    2.490617202129742e+1,
    2.767383207394972e+1,
    3.061929632950841e+1,
    3.375065608502399e+1,
    3.707713497083912e+1,
    4.060930496943413e+1,
    4.435936195160668e+1,
    4.834148224345283e+1,
    5.257229170785049e+1,
    5.707149458398093e+1,
    6.186273503855476e+1,
    6.697480787736505e+1,
    7.244341162998353e+1,
    7.831377964843565e+1,
    8.464480548222756e+1,
    9.151587398018528e+1,
    9.903899485517280e+1,
    1.073824762956655e+2,
    1.168236917656583e+2,
    1.278937448431646e+2,
    1.419607885990635e+2,
];

static GAUSS_LAGUERRE_40_WEIGHTS: [f64; 40] = [
    1.437204088033139e-1,
    2.304075592418809e-1,
    2.422530455213276e-1,
    2.036366391034408e-1,
    1.437606306229214e-1,
    8.691288347060781e-2,
    4.541750018329159e-2,
    2.061180312060695e-2,
    8.142788212686070e-3,
    2.802660756633776e-3,
    8.403374416217193e-4,
    2.193037329077657e-4,
    4.974016590092524e-5,
    9.785080959209777e-6,
    1.665428246036952e-6,
    2.445027367996577e-7,
    3.085370342362143e-8,
    3.332960729372821e-9,
    3.067818923653773e-10,
    2.393313099090116e-11,
    1.572947076762871e-12,
    8.649360130178674e-14,
    3.948198167006651e-15,
    1.482711730481083e-16,
    4.533903748150563e-18,
    1.115479804520358e-19,
    2.177666605892262e-21,
    3.318788910579756e-23,
    3.872847904397466e-25,
    3.381185924262449e-27,
    2.146990618932626e-29,
    9.574538399305471e-32,
    2.868778345026473e-34,
    5.452034672917572e-37,
    6.082128006541067e-40,
    3.571351222207245e-43,
    9.375169717620775e-47,
    8.418177761921027e-51,
    1.554777624272071e-55,
    1.625726581852354e-61,
];

static GAUSS_LAGUERRE_16_NODES: [f64; 16] = [
    6.990398696320011e-2,
    4.216550531234919e-1,
    1.077886957549787e+0,
    2.045007240070608e+0,
    3.332589390629165e+0,
    4.954060392944802e+0,
    6.927564456099590e+0,
    9.277260547765162e+0,
    1.203531807856921e+1,
    1.524508602669737e+1,
    1.896636896602229e+1,
    2.328480784962387e+1,
    2.833015260757935e+1,
    3.431685610993765e+1,
    4.165487031615267e+1,
    5.139394535360512e+1,
];

static GAUSS_LAGUERRE_16_WEIGHTS: [f64; 16] = [
    2.922417179018809e-1,
    3.811335382664116e-1,
    2.723536895279418e-1,
    1.292443070163705e-1,
    4.241214760845679e-2,
    9.693442499237142e-3,
    1.531608889114675e-3,
    1.643808449854753e-4,
    1.165782218980020e-5,
    5.251146380292064e-7,
    1.420270694502774e-8,
    2.126306491706232e-10,
    1.557034573630566e-12,
    4.548074008671047e-15,
    3.600541715551579e-18,
    2.928371092535847e-22,
];

static GAUSS_LAGUERRE_4_NODES: [f64; 4] = [
    2.605163387076808e-1,
    1.609745468990262e+0,
    4.334508323683735e+0,
    9.128563201951656e+0,
];

static GAUSS_LAGUERRE_4_WEIGHTS: [f64; 4] = [
    7.261979421222567e-1,
    3.654875045103909e-1,
    3.661964191912544e-2,
    4.819413563529568e-4,
];