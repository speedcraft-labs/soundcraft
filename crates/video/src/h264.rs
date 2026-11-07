//! H.264 decoding is provided by the shared deterministic `craft-h264` implementation.
//! The re-exports keep SoundCraft's existing `soundcraft_video::h264` API stable.

pub use craft_h264::{ColorInfo, DecodeStats, Decoder, Error, Picture, Result, default_threads, set_plane_allocator};
pub use craft_h264::{dpb, params, slice};
