//! Monophonic pitch tracking (YIN, de Cheveigné & Kawahara 2002) and note segmentation, for
//! audio-to-MIDI conversion.

use rustfft::{FftPlanner, num_complex::Complex};

/// A detected note: start and end in samples, MIDI pitch, and a velocity estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedNote {
    pub start: usize,
    pub end: usize,
    pub pitch: u8,
    pub velocity: u8,
}

/// Fundamental frequency of one frame, or None when unvoiced. `threshold` ≈ 0.1–0.2.
pub fn yin(frame: &[f32], sample_rate: f32, min_hz: f32, max_hz: f32, threshold: f32) -> Option<f32> {
    if !(sample_rate.is_finite() && sample_rate > 0.0) || frame.len() < 64 {
        return None;
    }
    let max_tau = ((sample_rate / min_hz.max(20.0)) as usize).min(frame.len() / 2);
    let min_tau = ((sample_rate / max_hz.max(min_hz + 1.0)) as usize).max(2);
    if max_tau <= min_tau.saturating_add(2) {
        return None;
    }
    let w = frame.len() - max_tau;
    // Compute sum(frame[j] * frame[j + tau]) for the fixed YIN window with one
    // cross-correlation. The prefix-energy terms are range sums, so each lag is O(1).
    let fft_len = (frame.len() + w - 1).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(fft_len);
    let inverse = planner.plan_fft_inverse(fft_len);
    let mut prefix = vec![Complex::new(0.0, 0.0); fft_len];
    let mut signal = vec![Complex::new(0.0, 0.0); fft_len];
    for (slot, &sample) in prefix.iter_mut().zip(&frame[..w]) {
        slot.re = sample;
    }
    for (slot, &sample) in signal.iter_mut().zip(frame) {
        slot.re = sample;
    }
    forward.process(&mut prefix);
    forward.process(&mut signal);
    for (signal, prefix) in signal.iter_mut().zip(prefix) {
        *signal *= prefix.conj();
    }
    inverse.process(&mut signal);

    let mut energy = vec![0.0f32; frame.len() + 1];
    for (i, sample) in frame.iter().enumerate() {
        energy[i + 1] = energy[i] + sample * sample;
    }
    let mut d = vec![0.0f32; max_tau + 1];
    let scale = 1.0 / fft_len as f32;
    for (tau, dv) in d.iter_mut().enumerate().skip(1) {
        let left_energy = energy[w];
        let right_energy = energy[tau + w] - energy[tau];
        *dv = (left_energy + right_energy - 2.0 * signal[tau].re * scale).max(0.0);
    }
    // Cumulative mean normalised difference.
    let mut cmnd = vec![1.0f32; max_tau + 1];
    let mut running = 0.0f32;
    for tau in 1..=max_tau {
        let v = d.get(tau).copied().unwrap_or(0.0);
        running += v;
        if let Some(c) = cmnd.get_mut(tau) {
            *c = if running > 0.0 { v * tau as f32 / running } else { 1.0 };
        }
    }
    let mut tau = min_tau;
    while tau < max_tau {
        let c = cmnd.get(tau).copied().unwrap_or(1.0);
        if c < threshold {
            // Walk to the local minimum.
            while tau + 1 < max_tau && cmnd.get(tau + 1).copied().unwrap_or(1.0) < cmnd.get(tau).copied().unwrap_or(1.0) {
                tau += 1;
            }
            // Parabolic interpolation.
            let (a, b, c) =
                (cmnd.get(tau - 1).copied().unwrap_or(1.0), cmnd.get(tau).copied().unwrap_or(1.0), cmnd.get(tau + 1).copied().unwrap_or(1.0));
            let denom = a - 2.0 * b + c;
            let shift = if denom.abs() > 1e-9 { 0.5 * (a - c) / denom } else { 0.0 };
            let t = tau as f32 + shift.clamp(-1.0, 1.0);
            return (t > 0.0).then_some(sample_rate / t);
        }
        tau += 1;
    }
    None
}

/// Convert mono audio to notes: frames of `hop` samples, pitch per frame, then runs of the same
/// semitone at least `min_frames` long become notes.
pub fn audio_to_notes(audio: &[f32], sample_rate: f32) -> Vec<DetectedNote> {
    let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48_000.0 };
    let win = ((sr * 0.046) as usize).clamp(256, 8192);
    let hop = win / 4;
    let min_frames = 3;
    let mut frames: Vec<(usize, Option<u8>, f32)> = Vec::new();
    let mut pos = 0usize;
    while pos + win <= audio.len() && frames.len() < 2_000_000 {
        let Some(frame) = audio.get(pos..pos + win) else { break };
        let rms = (frame.iter().map(|x| x * x).sum::<f32>() / win as f32).sqrt();
        let pitch = if rms > 0.01 {
            yin(frame, sr, 50.0, 1600.0, 0.15).map(|f| (69.0 + 12.0 * (f / 440.0).log2()).round().clamp(0.0, 127.0) as u8)
        } else {
            None
        };
        frames.push((pos, pitch, rms));
        pos += hop;
    }
    let mut notes = Vec::new();
    let mut i = 0;
    while i < frames.len() {
        let Some((start, Some(p), _)) = frames.get(i).copied() else {
            i += 1;
            continue;
        };
        let mut j = i;
        let mut peak = 0.0f32;
        while let Some((_, Some(q), rms)) = frames.get(j).copied() {
            if q != p {
                break;
            }
            peak = peak.max(rms);
            j += 1;
        }
        if j - i >= min_frames {
            let end = frames.get(j).map_or(audio.len(), |f| f.0);
            let velocity = (40.0 + 87.0 * (peak / 0.5).min(1.0)) as u8;
            notes.push(DetectedNote { start, end, pitch: p, velocity: velocity.clamp(1, 127) });
        }
        i = j.max(i + 1);
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_difference(frame: &[f32], max_tau: usize) -> Vec<f32> {
        let w = frame.len() - max_tau;
        (0..=max_tau)
            .map(|tau| {
                (0..w)
                    .map(|j| {
                        let diff = frame[j] - frame[j + tau];
                        diff * diff
                    })
                    .sum()
            })
            .collect()
    }

    fn tone(f: f32, secs: f32, sr: f32) -> Vec<f32> {
        (0..(secs * sr) as usize).map(|i| (i as f32 * f * std::f32::consts::TAU / sr).sin() * 0.5).collect()
    }

    #[test]
    fn yin_finds_a440() {
        let t = tone(440.0, 0.1, 48_000.0);
        let f = yin(&t[..2048], 48_000.0, 50.0, 1600.0, 0.15).unwrap();
        assert!((f - 440.0).abs() < 2.0, "{f}");
    }

    #[test]
    fn fft_difference_matches_direct_yin_sum() {
        let mut state = 7u32;
        let frame: Vec<f32> = (0..2048)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as i32 as f32) / i32::MAX as f32
            })
            .collect();
        let max_tau = 960;
        let direct = direct_difference(&frame, max_tau);

        let w = frame.len() - max_tau;
        let fft_len = (frame.len() + w - 1).next_power_of_two();
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft_len);
        let inverse = planner.plan_fft_inverse(fft_len);
        let mut prefix = vec![Complex::new(0.0, 0.0); fft_len];
        let mut signal = vec![Complex::new(0.0, 0.0); fft_len];
        for (slot, &sample) in prefix.iter_mut().zip(&frame[..w]) {
            slot.re = sample;
        }
        for (slot, &sample) in signal.iter_mut().zip(&frame) {
            slot.re = sample;
        }
        forward.process(&mut prefix);
        forward.process(&mut signal);
        for (signal, prefix) in signal.iter_mut().zip(prefix) {
            *signal *= prefix.conj();
        }
        inverse.process(&mut signal);
        let mut energy = vec![0.0f32; frame.len() + 1];
        for (i, sample) in frame.iter().enumerate() {
            energy[i + 1] = energy[i] + sample * sample;
        }
        for tau in 0..=max_tau {
            let estimate = energy[w] + energy[tau + w] - energy[tau] - 2.0 * signal[tau].re / fft_len as f32;
            let scale = direct[tau].max(1.0);
            assert!((estimate - direct[tau]).abs() <= scale * 2e-4, "tau={tau}: {estimate} != {}", direct[tau]);
        }
    }

    #[test]
    fn melody_to_notes() {
        let sr = 48_000.0;
        let mut a = tone(261.63, 0.4, sr);
        a.extend(vec![0.0; 4800]);
        a.extend(tone(392.0, 0.4, sr));
        let n = audio_to_notes(&a, sr);
        let pitches: Vec<u8> = n.iter().map(|x| x.pitch).collect();
        assert_eq!(pitches, vec![60, 67], "{n:?}");
    }

    #[test]
    fn silence_and_garbage() {
        assert!(audio_to_notes(&vec![0.0; 48_000], 48_000.0).is_empty());
        assert!(audio_to_notes(&[], f32::NAN).is_empty());
        assert!(yin(&[0.0; 10], 48_000.0, 50.0, 1600.0, 0.15).is_none());
    }

    #[test]
    fn huge_finite_rate_is_unvoiced() {
        // #241: the shortest lag saturates to usize::MAX; that is "no pitch", not an overflow.
        assert!(yin(&[0.0; 64], 1.0e30, 50.0, 1600.0, 0.15).is_none());
        assert!(yin(&[0.5; 8192], f32::MAX, 50.0, 1600.0, 0.15).is_none());
    }

    #[test]
    fn audio_to_notes_survives_huge_finite_rate() {
        assert!(audio_to_notes(&[0.5; 8192], 1.0e30).is_empty());
    }
}
