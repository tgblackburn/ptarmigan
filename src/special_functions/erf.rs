//! Error and inverse error functions.

pub trait Erf {
    /// Returns the value of the error function for real argument,
    /// 2/√π ∫_0^z exp(-t^2) dt
    fn erf(&self) -> Self;

    /// Returns the complementary error function 1 - erf(x).
    fn erfc(&self) -> Self;

    /// Returns the value of the inverse error function for real argument
    /// between -1 and +1. Outside this range, the function returns NaN.
    fn inv_erf(&self) -> Self;
}

impl Erf for f64 {
    fn erf(&self) -> Self {
        1.0 - self.erfc()
    }

    fn erfc(&self) -> Self {
        // from Y. D. Dia, "Approximate Incomplete Integrals, Application to Complementary Error Function"
        let x = self.abs();
        let x2 = x * x;

        let a = 0.56418958354775629 / (x + 2.06955023132914151);
        let b = (x2 + 2.71078540045147805 * x +  5.80755613130301624) / (x2 + 3.47954057099518960 * x + 12.06166887286239555);
        let c = (x2 + 3.47469513777439592 * x + 12.07402036406381411) / (x2 + 3.72068443960225092 * x +  8.44319781003968454);
        let d = (x2 + 4.00561509202259545 * x +  9.30596659485887898) / (x2 + 3.90225704029924078 * x +  6.36161630953880464);
        let e = (x2 + 5.16722705817812584 * x +  9.12661617673673262) / (x2 + 4.03296893109262491 * x +  5.13578530585681539);
        let f = (x2 + 5.95908795446633271 * x +  9.19435612886969243) / (x2 + 4.11240942957450885 * x +  4.48640329523408675);

        let erfc_x = (a * b * c * d * e * f) * (-x2).exp();
        if *self > 0.0 { erfc_x } else { 2.0 - erfc_x }
    }

    fn inv_erf(&self) -> Self {
        // from W. T. Shaw, T. Luu and N. Brickmann, arXiv:0901.0638 [q-fin.CP]
        let x = self.abs();
        let v = -(1.0 - x).ln();

        let p = 0.000036313870818023761224 + 4.3304513840364031401e-8 * v;
        let p = 0.0034424140686962222423 + v * p;
        let p = 0.085838533424158257377 +  v * p;
        let p = 0.73176759583280610539 + v * p;
        let p = 2.3884158540184385711 + v * p;
        let p = 3.0333178251950406994 + v * p;
        let p = 1.2533141359896652729 + v * p;

        let q = 0.00030617264753008793976 + 1.3141263119543315917e-6 * v;
        let q = 0.014494272424798068406 + v * q;
        let q = 0.2168237095066675527 + v * q;
        let q = 1.2356513216582148689 + v * q;
        let q = 2.9373357991677046357 + v * q;
        let q = 2.9202373175993672857 + v * q;
        let q = 1.0 + v * q;

        let inv_erf_x = v * p / (2_f64.sqrt() * q);
        inv_erf_x.copysign(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_function() {
        let args = [
            -2.0,
            0.07,
            0.38,
            1.3,
            5.2,
        ];

        let targets = [
            -0.995322265018952734,
            0.0788577197708907434,
            0.4090094534196940449,
            0.9340079449407,
            0.9999999999998075093890003,
        ];

        for (x, target) in args.iter().zip(targets.iter()) {
            let value = x.erf();
            let error = (target - value) / target;
            println!("x = {:.3}, erf = {:.6e}, err = {:.3e}", x, value, error);
            assert!(error.abs() < 1.0e-12);
        }
    }

    #[test]
    fn inverse_error_function() {
        let args = [
            -0.78,
            0.3,
            0.56,
            0.88,
            0.996,
            0.99999,
        ];

        let targets = [
            -0.867286350993874741,
            0.2724627147267543556,
            0.546023058139055096,
            1.099390951949219265,
            2.0351676830660831,
            3.123413274340875,
        ];

        for (x, target) in args.iter().zip(targets.iter()) {
            let value = x.inv_erf();
            let error = (target - value) / target;
            println!("x = {:.3}, inv_erf = {:.6e}, err = {:.3e}", x, value, error);
            assert!(error.abs() < 1.0e-8);
        }
    }

    #[test]
    fn inverse_error_and_error_function() {
        let mut rms_err = 0.0;
        for i in -20..20 {
            let x = 0.01 + 0.14 * (i as f64);
            let y = x.erf().inv_erf();
            let err = (y - x) / x;
            rms_err += err * err;
        }
        let rms_err = (rms_err / 40.0).sqrt();
        println!("x ?= inv_erf(erf(x)) => rms err = {:.3e}", rms_err);
        assert!(rms_err < 1.0e-9);
    }
}