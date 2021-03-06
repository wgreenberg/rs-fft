use std::convert::Into;
use crate::complex::Complex;
use crate::fft::fft;

struct FrequencyComponent {
    f: f64,
    coeff: f64,
}

fn get_primary_frequencies<T>(mut samples: Vec<T>, sample_rate: u32, threshold: f64) -> Vec<FrequencyComponent> where T: Into<f64> {
    let n = samples.len();
    let bin_size = (sample_rate as f64) / (n as f64);
    let f64_samples: Vec<f64> = samples.drain(..).map(|s| s.into()).collect();
    let freq_domain: Vec<f64> = fft(f64_samples).data.drain(..)
        .map(|s| s.mag() / (n as f64))
        .take(n/2)
        .collect();
    freq_domain.windows(3).enumerate().filter_map(|(k_prev, window)| {
        let (prev, curr, next) = (window[0], window[1], window[2]);
        if curr > threshold && curr > prev && curr > next {
            Some(FrequencyComponent {
                f: (k_prev + 1) as f64 * bin_size,
                coeff: curr,
            })
        } else {
            None
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav() {
        use hound;
        use plotters::prelude::*;
        let mut reader = hound::WavReader::open("sine.wav").unwrap();
        let samples: Vec<i16> = reader.samples::<i16>().flat_map(|s| s).take(8192).collect();
        let sample_rate = reader.spec().sample_rate;
        let freqs = get_primary_frequencies(samples, sample_rate, 20.0);
        assert!(freqs.len() == 1);
        assert!(freqs[0].f - 440.0 < 10.0);
    }
}
