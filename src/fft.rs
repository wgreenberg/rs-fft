use std::convert::From;
use crate::complex::{Complex, ZERO, exp_i};
use std::f64::consts::PI;

#[derive(Debug, PartialEq)]
pub struct FFTVec {
    pub data: Vec<Complex>,
}

impl FFTVec {
    pub fn split(mut self) -> (FFTVec, FFTVec) {
        let mut even = Vec::new();
        let mut odd = Vec::new();
        for (i, x_i) in self.data.drain(..).enumerate() {
            if i % 2 == 0 {
                even.push(x_i);
            } else {
                odd.push(x_i);
            }
        }
        (FFTVec { data: even }, FFTVec { data: odd })
    }
}

impl From<Vec<Complex>> for FFTVec {
    fn from(v: Vec<Complex>) -> FFTVec {
        FFTVec { data: pad_if_necessary(v) }
    }
}

fn pad_if_necessary(mut v: Vec<Complex>) -> Vec<Complex> {
    let len = v.len();
    let is_power_of_2 = (len & (len - 1)) == 0;
    if !is_power_of_2 {
        let next_pow_of_2 = (len as f64).log2().floor() as i32 + 1;
        let padded_size = (2.0 as f64).powi(next_pow_of_2) as usize;
        let padding = vec![ZERO; padded_size - len];
        v.extend(padding);
    }
    v
}

impl From<Vec<f64>> for FFTVec {
    fn from(mut v: Vec<f64>) -> FFTVec {
        let complex = v.drain(..).map(|x| Complex::new(x, 0.0)).collect();
        FFTVec { data: pad_if_necessary(complex) }
    }
}

pub fn fft<T>(vec: T) -> FFTVec where T: Into<FFTVec> {
    fft_general(vec.into(), false)
}

pub fn ifft<T>(vec: T) -> FFTVec where T: Into<FFTVec> {
    let fftvec = vec.into();
    let n = fftvec.data.len() as f64;
    let mut result = fft_general(fftvec, true);
    for c in result.data.iter_mut() {
        *c = Complex::new(c.re / n, c.im / n);
    }
    result
}

fn fft_general(vec: FFTVec, inverse: bool) -> FFTVec {
    let n = vec.data.len();
    if n == 1 {
        vec
    } else {
        let (even, odd) = vec.split();
        let even_fft = fft_general(even, inverse);
        let mut odd_fft = fft_general(odd, inverse);
        let mut result = even_fft;
        result.data.extend(odd_fft.data.drain(..));
        for k in 0 .. n/2 {
            let x_k = result.data[k];
            let w = if inverse {
                exp_i(2.0 * PI * (k as f64 / n as f64))
            } else {
                exp_i(-2.0 * PI * (k as f64 / n as f64))
            };
            result.data[k] = x_k + w * result.data[k + n/2];
            result.data[k + n/2] = x_k - w * result.data[k + n/2];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding() {
        let no_padding: FFTVec = vec![ZERO; 4].into();
        assert_eq!(no_padding, FFTVec {
            data: vec![ZERO; 4]
        });

        let needs_padding: FFTVec = vec![ZERO; 5].into();
        assert_eq!(needs_padding, FFTVec {
            data: vec![ZERO; 8]
        });
    }

    #[test]
    fn real_values() {
        let complexified: FFTVec = vec![0.0; 4].into();
        assert_eq!(complexified, FFTVec {
            data: vec![ZERO; 4]
        });
    }

    #[test]
    fn split() {
        let incr: Vec<f64> = (0..4).map(|x| x as f64).collect();
        let v: FFTVec = incr.into();
        let (even, odd) = v.split();
        assert_eq!(even, FFTVec { data: vec![
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
        ]});
        assert_eq!(odd, FFTVec { data: vec![
            Complex::new(1.0, 0.0),
            Complex::new(3.0, 0.0),
        ]});
    }

    #[test]
    fn fft_simple() {
        assert_eq!(fft(vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, -1.0),
            Complex::new(0.0, -1.0),
            Complex::new(-1.0, 2.0),
        ]), FFTVec { data: vec![
            Complex::new(2.0, 0.0),
            Complex::new(-2.0, -2.0),
            Complex::new(0.0, -2.0),
            Complex::new(4.0, 4.0),
        ]});

        assert_eq!(ifft(vec![
            Complex::new(2.0, 0.0),
            Complex::new(-2.0, -2.0),
            Complex::new(0.0, -2.0),
            Complex::new(4.0, 4.0),
        ]), FFTVec { data: vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, -1.0),
            Complex::new(0.0, -1.0),
            Complex::new(-1.0, 2.0),
        ]});
    }
}
