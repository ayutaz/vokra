//! Stream handles and session/stream management (M0-02-T14).
//!
//! This is the "session/stream 管理の初期実装" deliverable of WP M0-02: a
//! [`Session`] can open multiple independent [`Stream`] handles, each with a
//! session-unique id, released independently on [`Drop`].
//!
//! # Direction (recorded for later WPs)
//!
//! Streaming state — RNN h/c, KV cache, iSTFT tail — will be *owned by the
//! stream handle* so users never manage tensor names themselves (FR-ST-05,
//! a **v0.1 MVP requirement**; M0 ships only the [`StreamState`] shell).
//!
//! # Deliberately NOT implemented in M0 (v0.1 MVP = M1-08 scope)
//!
//! Per the SRS version tags (and milestones.md §4.2 表注 1's principle of
//! not pulling M1 API into M0), the following are intentionally absent from
//! the public API and must not be added here ahead of schedule:
//!
//! - `Session::step_chunk` / `step_frame` (FR-ST-01),
//! - `stream.push(chunk)` → `stream.poll(events)` with a lock-free ring
//!   buffer (FR-ST-02),
//! - compile-time `Send` / `Sync` verification of `Session` / `Stream`
//!   (FR-API-03).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::error::Result;
use crate::session::{Session, SessionInner};

/// Placeholder for per-stream state (M0 shell).
///
/// FR-ST-05 (v0.1 MVP): RNN h/c, KV cache and the iSTFT tail buffer will
/// live here, hidden behind the stream handle. `#[non_exhaustive]` keeps
/// construction crate-internal.
#[derive(Debug)]
#[non_exhaustive]
pub struct StreamState {}

/// Handle to one streaming inference context of a [`Session`].
///
/// Streams are created with [`Session::open_stream`], carry a
/// session-unique [`id`](Stream::id), and release their slot on [`Drop`]
/// independently of other streams.
#[derive(Debug)]
pub struct Stream {
    session: Arc<SessionInner>,
    id: u64,
    state: StreamState,
}

impl Session {
    /// Opens a new independent stream on this session.
    ///
    /// M0 behaviour: allocates a session-unique stream id and tracks the
    /// stream in the session's active-stream count. Streaming I/O itself
    /// (push/poll, step APIs) is v0.1 MVP scope — see the module docs.
    ///
    /// ```no_run
    /// let session = vokra_core::Session::from_file("voice.gguf").build()?;
    /// let a = session.open_stream()?;
    /// let b = session.open_stream()?;
    /// assert_ne!(a.id(), b.id());
    /// assert_eq!(session.active_stream_count(), 2);
    /// # Ok::<(), vokra_core::VokraError>(())
    /// ```
    pub fn open_stream(&self) -> Result<Stream> {
        // Relaxed ordering: plain counters with no cross-variable ordering
        // requirements (id uniqueness comes from the atomic RMW itself).
        let id = self.inner.next_stream_id.fetch_add(1, Ordering::Relaxed);
        self.inner.active_streams.fetch_add(1, Ordering::Relaxed);
        Ok(Stream {
            session: Arc::clone(&self.inner),
            id,
            state: StreamState {},
        })
    }

    /// Number of currently open streams on this session.
    pub fn active_stream_count(&self) -> u64 {
        self.inner.active_streams.load(Ordering::Relaxed)
    }
}

impl Stream {
    /// Identifier of this stream, unique within its originating session.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Per-stream state (M0: placeholder shell; see [`StreamState`]).
    pub fn state(&self) -> &StreamState {
        &self.state
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.session.active_streams.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::tests::TempModelFile;

    fn session(tag: &str) -> (TempModelFile, Session) {
        let file = TempModelFile::new(tag);
        let session = Session::from_file(&file.0).build().expect("session builds");
        (file, session)
    }

    #[test]
    fn open_and_drop_lifecycle() {
        let (_file, session) = session("stream-lifecycle");
        assert_eq!(session.active_stream_count(), 0);
        {
            let stream = session.open_stream().expect("stream opens");
            let _ = stream.state();
            assert_eq!(session.active_stream_count(), 1);
        }
        assert_eq!(session.active_stream_count(), 0);
    }

    #[test]
    fn multiple_streams_have_unique_ids_and_release_independently() {
        let (_file, session) = session("stream-multi");
        let s0 = session.open_stream().expect("s0");
        let s1 = session.open_stream().expect("s1");
        let s2 = session.open_stream().expect("s2");
        assert_eq!(session.active_stream_count(), 3);
        assert_ne!(s0.id(), s1.id());
        assert_ne!(s1.id(), s2.id());
        assert_ne!(s0.id(), s2.id());

        // Independent release, out of creation order.
        drop(s1);
        assert_eq!(session.active_stream_count(), 2);
        drop(s0);
        assert_eq!(session.active_stream_count(), 1);
        drop(s2);
        assert_eq!(session.active_stream_count(), 0);
    }

    #[test]
    fn stream_outlives_use_of_session_reference() {
        let (_file, session) = session("stream-arc");
        let stream = session.open_stream().expect("stream opens");
        // The stream holds an Arc to the session internals, so using it
        // after other session activity is fine.
        assert_eq!(session.active_stream_count(), 1);
        assert_eq!(stream.id(), 0);
    }
}
