//! Hilbert transforms

use num_complex::Complex64;
use rustfft::FftPlanner;

pub trait AnalyticalSignal {
    type Output;

    /// Computes the analytical signal f_c(t) = f_x(t) + i f_y(t) associated with
    /// the real function f_x(t), removing any DC component if necessary
    fn analytical_signal(&self) -> Self::Output;
}

impl AnalyticalSignal for [f64] {
    type Output = Vec<Complex64>;

    /// Computes the analytical signal associated with the real function f(t)
    fn analytical_signal(&self) -> Self::Output {
        let n = self.len();
        let mut buffer: Vec<Complex64> = Vec::with_capacity(n);

        for f in self {
            buffer.push(f.into());
        }

        buffer.analytical_signal()
    }
}

impl AnalyticalSignal for [Complex64] {
    type Output = Vec<Complex64>;

    fn analytical_signal(&self) -> Self::Output {
        let mut planner = FftPlanner::new();

        let n = self.len();
        let mut buffer = self.to_vec();

        // First, FFT forwards
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Kill any DC component
        buffer[0] *= 0.0;

        // Zero out negative frequency components
        for i in (n/2 + 1)..n {
            buffer[i] *= 0.0;
        }

        for i in 1..n/2 {
            buffer[i] *= 2.0;
        }

        // Go backwards
        let fft = planner.plan_fft_inverse(n);
        fft.process(&mut buffer);

        // Fix magnitudes
        for v in &mut buffer {
            *v = *v / (n as f64);
        }

        buffer
    }
}