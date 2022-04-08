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

#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use std::f64::consts;
use super::Series;

pub trait Airy: Sized {
    /// Returns the value of the Airy function for real, positive argument.
    /// If `x` is negative, or the result would underflow, the
    /// return value is None.
    fn ai(&self) -> Option<Self>;

    /// Returns the derivative of the Airy function for real, positive argument.
    /// If `x` is negative, or the result would underflow, the
    /// return value is None.
    fn ai_prime(&self) -> Option<Self>;

    /// Returns the value of the modified Bessel function K_{1/3} for
    /// real, positive argument.
    fn bessel_K_1_3(&self) -> Option<Self>;

    /// Returns the value of the modified Bessel function K_{2/3} for
    /// real, positive argument.
    fn bessel_K_2_3(&self) -> Option<Self>;
}

impl Airy for f64 {
    fn ai(&self) -> Option<Self> {
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
        let x = *self;
        if x < 0.0 {
            None
        } else if x < 1.0 {
            // Use Taylor series expansion
            Some(AIRY_SMALL_X.evaluate_up_to(x, 20))
        } else {
            let s = 2.0 * x.powf(1.5) / 3.0;
            let a = 0.262183997088323 * s.powf(-1.0/6.0) * (-s).exp();
            let (nodes, weights): (&[f64], &[f64]) = if x < 2.0 {
                (&GL_m1_6_NODES_40, &GL_m1_6_WEIGHTS_40)
            } else if x < 10.0 {
                (&GL_m1_6_NODES_16, &GL_m1_6_WEIGHTS_16)
            } else {
                (&GL_m1_6_NODES_4, &GL_m1_6_WEIGHTS_4)
            };
            let integral: f64 = nodes.iter()
                .zip(weights.iter())
                .map(|(x, w)| w * (2.0 + x / s).powf(-1.0/6.0))
                .sum();
            Some(a * integral)
        }
    }

    fn ai_prime(&self) -> Option<Self> {
        // Numerically integrate the integal representation
        // the Airy prime function using Gauss-Laguerre quadrature.
        //
        // That representation is
        //   Ai'(x) = b(x) \int_0^\infty f(t) w(t) dt
        // where the integrand
        //   f(t) = (2 + t/s)^(1/6),
        // the weight function
        //   w(t) = t^(1/6) exp(-t),
        // the scale factor
        //   b(x) = -(3s)^(1/6) exp(-s) / (2^(4/3) sqrt(pi) Gamma(7/6))
        // and
        //   s = 2 x^(3/2) / 3.
        let x = *self;
        if x < 0.0 {
            None
        } else if x < 1.0 {
            // Use Taylor series expansion
            Some(AIRY_PRIME_SMALL_X.evaluate_up_to(x, 20))
        } else {
            let s = 2.0 * x.powf(1.5) / 3.0;
            let b = -0.2898380090915846 * s.powf(1.0/6.0) * (-s).exp();
            let (nodes, weights): (&[f64], &[f64]) = if x < 2.0 {
                (&GL_p1_6_NODES_32, &GL_p1_6_WEIGHTS_32)
            } else if x < 4.0 {
                (&GL_p1_6_NODES_16, &GL_p1_6_WEIGHTS_16)
            } else if x < 11.0 {
                (&GL_p1_6_NODES_8, &GL_p1_6_WEIGHTS_8)
            } else {
                (&GL_p1_6_NODES_4, &GL_p1_6_WEIGHTS_4)
            };
            let integral: f64 = nodes.iter()
                .zip(weights.iter())
                .map(|(x, w)| w * (2.0 + x / s).powf(1.0/6.0))
                .sum();
            Some(b * integral)
        }
    }

    fn bessel_K_1_3(&self) -> Option<Self> {
        // K(1/3, x) = pi (2 sqrt(3) / x)^(1/3) Ai[ (1.5 x)^(2/3) ]
        let x = self;
        (1.5 * x)
            .powf(2.0 / 3.0)
            .ai()
            .map(|y| consts::PI * (2.0 * 3.0f64.sqrt() / x).cbrt() * y)
    }

    fn bessel_K_2_3(&self) -> Option<Self> {
        // K(2/3, x) = -pi / 3^(1/6) (2 / x)^(2/3) Ai'[ (1.5 x)^(2/3) ]
        let x = self;
        (1.5 * x)
            .powf(2.0 / 3.0)
            .ai_prime()
            .map(|y| -consts::PI * 3.0f64.powf(-1.0/6.0) * (2.0 / x).powf(2.0 / 3.0) * y)
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
        let val = Airy::ai(&200.0).unwrap();
        let target = 9.1536243084526844166e-821;
        println!("Ai(200) = {:e}, calculated = {:e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn airy_prime_0_5() {
        let val = Airy::ai_prime(&0.5).unwrap();
        let target = -0.2249105326646839;
        println!("Ai'(0) = {:.15e}, calculated = {:.15e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn airy_prime_1_2() {
        let val = Airy::ai_prime(&1.2).unwrap();
        let target = -0.1327853785572262;
        println!("Ai'(1.2) = {:.15e}, calculated = {:.15e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn airy_prime_3() {
        let val = Airy::ai_prime(&3.0).unwrap();
        let target = -0.01191297670595132;
        println!("Ai'(3) = {:.15e}, calculated = {:.15e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn airy_prime_5() {
        let val = Airy::ai_prime(&5.0).unwrap();
        let target = -0.0002474138908684625;
        println!("Ai'(5) = {:.15e}, calculated = {:.15e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn airy_prime_100() {
        let val = Airy::ai_prime(&100.0).unwrap();
        let target = -2.635140361604410e-290;
        println!("Ai'(100) = {:.15e}, calculated = {:.15e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    #[should_panic]
    fn airy_prime_200() {
        let val = Airy::ai_prime(&200.0).unwrap();
        let target = -1.294632359221882e-819; // too small to represent
        println!("Ai'(200) = {:.15e}, calculated = {:.15e}", target, val);
        assert!( ((val - target)/target).abs() < MAX_REL_ERR );
    }

    #[test]
    fn bessel_k() {
        let pts = [
            (1.0, 0.43843063344153436171, 0.49447506210420826699),
            (2.0, 0.11654496129616524876, 0.12483892748812831057),
            (10.0, 0.000017874608271055334883, 0.000018161187569530204281),
            (20.0, 5.7568278247790870062e-10, 5.8038484271925806951e-10),
        ];

        for (x, k1_3, k2_3) in pts.iter() {
            let value = x.bessel_K_1_3().unwrap();
            let error = ((value - k1_3) / k1_3).abs();
            println!("K(1/3, {}) = {:.15e}, error = {:.3e}", x, value, error);
            assert!(error < MAX_REL_ERR);

            let value = x.bessel_K_2_3().unwrap();
            let error = ((value - k2_3) / k2_3).abs();
            println!("K(2/3, {}) = {:.15e}, error = {:.3e}", x, value, error);
            assert!(error < MAX_REL_ERR);
        }
    }
}

static AIRY_SMALL_X: Series<i32> = Series {
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
 
static GL_m1_6_NODES_40: [f64; 40] = [
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

static GL_m1_6_WEIGHTS_40: [f64; 40] = [
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

static GL_m1_6_NODES_16: [f64; 16] = [
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

static GL_m1_6_WEIGHTS_16: [f64; 16] = [
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

static GL_m1_6_NODES_4: [f64; 4] = [
    2.605163387076808e-1,
    1.609745468990262e+0,
    4.334508323683735e+0,
    9.128563201951656e+0,
];

static GL_m1_6_WEIGHTS_4: [f64; 4] = [
    7.261979421222567e-1,
    3.654875045103909e-1,
    3.661964191912544e-2,
    4.819413563529568e-4,
];

static AIRY_PRIME_SMALL_X: Series<i32> = Series {
    a: [
        -2.588194037928068e-1,
        1.775140269439086e-1,
        -8.627313459760227e-2,
        1.183426846292724e-2,
        -3.594713941566761e-3,
        2.465472596443175e-4,
        -5.705895145344065e-5,
        2.490376360043611e-6,
        -4.754912621120054e-7,
        1.482366880978340e-8,
        -2.438416728779515e-9,
        5.813203454817020e-11,
        -8.466724752706649e-12,
        1.614778737449172e-13,
        -2.121986153560564e-14,
        3.343227199687727e-16,
        -4.018913169622280e-17,
        5.357735896935460e-19,
        -5.953945436477452e-20,
        6.842574581015913e-22,
    ],
    n: [
        0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29
    ],
};

static GL_p1_6_NODES_32: [f64; 32] = [
    5.419216335178797e-2,
    2.563592519394501e-1,
    6.105337846619220e-1,
    1.117601250499268e+0,
    1.778772012187290e+0,
    2.595638278983496e+0,
    3.570199899661993e+0,
    4.704893160223047e+0,
    6.002626347905401e+0,
    7.466823827262402e+0,
    9.101480444265405e+0,
    1.091122854489481e+1,
    1.290142060175126e+1,
    1.507823142543966e+1,
    1.744878530615157e+1,
    2.002131536333526e+1,
    2.280536515887909e+1,
    2.581204670185127e+1,
    2.905437507794347e+1,
    3.254770931734353e+1,
    3.631034393434832e+1,
    4.036431973636326e+1,
    4.473656336085551e+1,
    4.946053708147523e+1,
    5.457871410534143e+1,
    6.014645784733515e+1,
    6.623844176939932e+1,
    7.296004468269724e+1,
    8.046956683977571e+1,
    8.902770553058033e+1,
    9.913306653084352e+1,
    1.120679739957869e+2,
];

static GL_p1_6_WEIGHTS_32: [f64; 32] = [
    7.367724323641389e-2,
    1.714949262919796e-1,
    2.153058701016087e-1,
    1.945396597062869e-1,
    1.372953180154878e-1,
    7.829966135819920e-2,
    3.668710726033288e-2,
    1.424309297649275e-2,
    4.600499408929096e-3,
    1.237919864444931e-3,
    2.773347849650942e-4,
    5.162030773770084e-5,
    7.955256663254091e-6,
    1.010381613093466e-6,
    1.051377364457356e-7,
    8.899151696883182e-9,
    6.074337175352181e-10,
    3.309246110261001e-11,
    1.421363931034547e-12,
    4.743105898350198e-14,
    1.208266778686291e-15,
    2.300146249450871e-17,
    3.187891977157999e-19,
    3.113810948822685e-21,
    2.056799463698863e-23,
    8.706362674815503e-26,
    2.196575768354253e-28,
    2.982555239475078e-31,
    1.870639634978154e-34,
    4.218964065274585e-38,
    2.128144435698866e-42,
    7.223360574564270e-48,
];

static GL_p1_6_NODES_16: [f64; 16] = [
    1.065086833791228e-1,
    5.045951155037972e-1,
    1.204887493317772e+0,
    2.214046599347790e+0,
    3.541916988814612e+0,
    5.202171768085574e+0,
    7.213182107425566e+0,
    9.599322961711244e+0,
    1.239297800449880e+1,
    1.563772034253402e+1,
    1.939360520131596e+1,
    2.374657817383913e+1,
    2.882679983703437e+1,
    3.484935526158435e+1,
    4.222537043072563e+1,
    5.200762769754892e+1,
];

static GL_p1_6_WEIGHTS_16: [f64; 16] = [
    1.538656081051387e-1,
    2.953754551696966e-1,
    2.638474205248174e-1,
    1.455113440029979e-1,
    5.332203916022353e-2,
    1.327529826855286e-2,
    2.246975117999703e-3,
    2.552794884543879e-4,
    1.899652442772008e-5,
    8.918653910312376e-7,
    2.501268133009190e-8,
    3.867337606990208e-10,
    2.915656187795174e-12,
    8.748503423240594e-15,
    7.104981102043080e-18,
    5.929862290898086e-22,
];

static GL_p1_6_NODES_8: [f64; 8] = [
    2.059975989105105e-1,
    9.813842157761038e-1,
    2.367409551874067e+0,
    4.419136826604687e+0,
    7.232706381634387e+0,
    1.097875723010299e+1,
    1.599457655964550e+1,
    2.315336496878509e+1,
];

static GL_p1_6_WEIGHTS_8: [f64; 8] = [
    3.011583649037287e-1,
    4.012843245437889e-1,
    1.846919290002817e-1,
    3.721629829536716e-2,
    3.257951078451206e-3,
    1.094133825516171e-4,
    1.051096399731320e-6,
    1.329470132838947e-9,
];

static GL_p1_6_NODES_4: [f64; 4] = [
    3.871921360861267e-1,
    1.882252071875780e+0,
    4.737752987864212e+0,
    9.659469470840548e+0,
];

static GL_p1_6_WEIGHTS_4: [f64; 4] = [
    5.272915276171070e-1,
    3.578901722679283e-1,
    4.192732430657323e-2,
    6.103094384307378e-4,
];
