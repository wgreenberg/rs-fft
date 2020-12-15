use std::ops::{Add, Sub, Mul};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

impl Complex {
    pub fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    pub fn mag(self) -> f64 {
        (self.re.powi(2) + self.im.powi(2)).sqrt()
    }
}

pub fn exp_i(x: f64) -> Complex {
    Complex::new(x.cos(), x.sin())
}

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let (a, b) = (self.re, self.im);
        let (c, d) = (other.re, other.im);
        Self {
            re: a*c - b*d,
            im: a*d + b*c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication() {
        let a = Complex::new(1.0, 4.0);
        let b = Complex::new(5.0, 1.0);
        assert_eq!(a * b, Complex::new(1.0, 21.0));
    }

    #[test]
    fn euler() {
        let e_to_the_pi_i = exp_i(std::f64::consts::PI);
        assert!(e_to_the_pi_i.mag() - 1.0 < 0.000000001);
    }
}
