use num_complex::Complex64;
use rustfft::FftPlanner;

pub struct IndefiniteIntegral {
    a: f64,
    b: f64,
    bdy: Complex64,
    // alpha: [Complex64; 15],
    c: [Complex64; 14],
}

impl IndefiniteIntegral {
    pub fn new(f: &[Complex64], a: f64, b: f64, bdy: Complex64) -> Self {
        assert!(f.len() == 15);

        // fill with function values
        let mut buffer: Vec<Complex64> = Vec::with_capacity(28);
        for i in 0..15 {
            buffer.push(f[14 - i]);
        }
        for i in 15..28 {
            buffer.push(f[i - 14]);
        }
        assert!(buffer.len() == 28);

        // cosine transform
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(28);
        fft.process(&mut buffer);
        buffer.truncate(15);
        for alpha in buffer.iter_mut() {
            *alpha  = 2.0 * *alpha / 28_f64;
        }

        // let mut alpha: [Complex64; 15] = [Default::default(); 15];
        // alpha[0] = 0.5 * buffer[0];
        // alpha[14] = 0.5 * buffer[14];
        // for i in 1..14 {
        //     alpha[i] = buffer[i];
        // }

        let mut c: [Complex64; 14] = [Default::default(); 14];
        for i in 1..14 {
            c[i] = (buffer[i-1] - buffer[i+1]) / ((2 * i) as f64);
            c[0] += (-1_f64).powi(i as i32) * -c[i];
        }

        // for elem in c.iter() {
        //     println!("{:.3e} {:.3e}", elem.re, elem.im);
        // }

        Self {
            a,
            b,
            bdy,
            // alpha,
            c,
        }
    }

    // pub fn diff(&self, x: f64) -> Complex64 {
    //     let x = (2.0 * x - self.a - self.b) / (self.b - self.a); // rescale from [a,b] to [-1,1]
    //     let mut t_nm2 = 1.0;
    //     let mut t_nm1 = x;
    //     let mut f = self.alpha[0] * t_nm2 + self.alpha[1] * t_nm1;
    //     for n in 2..15 {
    //         let t_n = 2.0 * x * t_nm1 - t_nm2;
    //         f += self.alpha[n] * t_n;
    //         t_nm2 = t_nm1;
    //         t_nm1 = t_n;
    //     }
    //     f
    // }

    pub fn at(&self, x: f64) -> Complex64 {
        //   F(x) = sum_{k=0}^n c_k T_k(x)
        // where the Chebyshev polynomials satisfy:
        //   T_0(x) = 1
        //   T_1(x) = x
        //   T_n(x) = 2 x T_{n-1}(x) - T_{n-2}(x)
        let x = (2.0 * x - self.a - self.b) / (self.b - self.a); // rescale from [a,b] to [-1,1]
        let mut t_nm2 = 1.0;
        let mut t_nm1 = x;
        let mut f = self.c[0] * t_nm2 + self.c[1] * t_nm1;
        for n in 2..14 {
            let t_n = 2.0 * x * t_nm1 - t_nm2;
            f += self.c[n] * t_n;
            t_nm2 = t_nm1;
            t_nm1 = t_n;
        }
        0.5 * (self.b - self.a) * f + self.bdy
    }
}

#[cfg(test)]
mod tests {
    use crate::quadrature::CLENSHAW_CURTIS_15_NODES_WEIGHTS;
    use super::*;

    #[test]
    fn chebyshev() {
        let (a, b) = (0.0, 2.0);

        let f: Vec<Complex64> = CLENSHAW_CURTIS_15_NODES_WEIGHTS.iter()
            .map(|(x, _)| a + (b - a) * x)
            .map(|x| Complex64::new((-4.0 * x * x).exp(), x))
            .collect();

        let interpolant = IndefiniteIntegral::new(&f, a, b, Default::default());

        let target = 0.3734120664062135;
        let result = interpolant.at(0.5).re;
        let err = (target - result).abs() / target;
        println!("∫_0^{{1/2}} f(x) dx = {:.4}, err = {:.3} per mille", target, 1000.0 * err);
        assert!(err < 1.0e-5);

        let target = 0.4431134558947878;
        let result = interpolant.at(2.0).re;
        let err = (target - result).abs() / target;
        println!("∫_0^2 f(x) dx = {:.4}, err = {:.3} per mille", target, 1000.0 * err);
        assert!(err < 1.0e-5);
    }
}