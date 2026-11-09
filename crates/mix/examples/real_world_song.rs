//! A small real-song editing path for the #291 SoundCraft journey.
//!
//! Decode a pinned song, lay it out as sixteen edited regions, render several play/scrub
//! positions, run audio-to-MIDI on an eight-second selection, then bounce the full song to WAV.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use soundcraft_audio_io::{AudioBuffer, EncodeOptions, decode, encode};
use soundcraft_model::{ChannelFormat, Clip, Session, SourceAudio, SourceId, TrackKind};
use soundcraft_time::{Range, SampleRate};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next().ok_or("usage: real_world_song INPUT.ogg OUTPUT.wav")?);
    let output = PathBuf::from(args.next().ok_or("usage: real_world_song INPUT.ogg OUTPUT.wav")?);
    let started = Instant::now();

    let encoded = fs::read(&input)?;
    let (info, audio) = decode(&encoded, input.extension().and_then(|ext| ext.to_str()))?;
    let sample_rate = SampleRate::new(info.sample_rate)?;
    let frames = i64::try_from(audio.frames())?;
    let channels = audio.num_channels();
    if frames == 0 || channels == 0 {
        return Err("the selected song has no decoded audio".into());
    }

    let midi_window = (audio.sample_rate as usize * 8).min(audio.frames());
    let mono = audio.channels.first().ok_or("the selected song has no first channel")?;
    let notes = soundcraft_dsp::pitch_detect::audio_to_notes(&mono[..midi_window], audio.sample_rate as f32);

    let mut session = Session::new("Real song journey", sample_rate);
    let source = SourceId(1_000_000);
    session.pool.insert(source, Arc::new(SourceAudio::new(audio)));
    let track = session.add_track(TrackKind::Audio, ChannelFormat::Stereo, Some("Music"));
    let region_len = (frames / 16).max(1);
    let mut clips = Vec::with_capacity(16);
    for region in 0..16i64 {
        let start = region * region_len;
        if start >= frames {
            break;
        }
        let length = (frames - start).min(region_len);
        let clip_id = session.new_clip_id();
        clips.push(Clip::audio(clip_id, format!("Region {}", region + 1), source, start, start, length));
    }
    let track_model = session.track_mut(track).ok_or("could not create audio track")?;
    let playlist = track_model.playlist_mut().ok_or("audio track has no playlist")?;
    playlist.clips.extend(clips);

    // Simulate normal playback and several timeline scrubs, each asking the mix engine for one
    // device-sized block at the requested position.
    let mut listened_peak = 0.0f32;
    for seconds in [0.0, 35.0, 78.0, 126.0, 170.0] {
        let position = sample_rate.samples(seconds).clamp(0, frames.saturating_sub(1));
        let block = soundcraft_mix::render_range(&session, Range::new(position, (position + 1024).min(frames)), 512);
        for sample in block.iter().flatten() {
            listened_peak = listened_peak.max(sample.abs());
        }
    }

    // Bounce the complete timeline; the output waveform and note count form the identity check.
    let rendered = soundcraft_mix::render_range(&session, Range::new(0, frames), 512);
    let bounce = AudioBuffer { sample_rate: sample_rate.hz(), channels: rendered };
    let wav = encode(&bounce, &EncodeOptions::default())?;
    fs::write(&output, &wav)?;
    let elapsed_ns = started.elapsed().as_nanos();
    let mut hasher = Sha256::new();
    hasher.update(&wav);
    for note in &notes {
        hasher.update((note.start as u64).to_le_bytes());
        hasher.update((note.end as u64).to_le_bytes());
        hasher.update([note.pitch, note.velocity]);
    }
    println!("CRAFT_JOURNEY_NS={elapsed_ns}");
    println!("CRAFT_OUTPUT_SHA256={:x}", hasher.finalize());
    println!("song_seconds={:.3} regions=16 scrub_peak={listened_peak:.6} detected_notes={}", info.duration_secs(), notes.len());
    Ok(())
}
