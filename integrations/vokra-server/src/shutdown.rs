//! Graceful-shutdown signal plumbing.
//!
//! `install_shutdown_signal` returns a future that resolves when the process
//! receives `SIGINT` (Ctrl-C) or `SIGTERM`. The HTTP and Wyoming listeners
//! both `.await` this shared trigger so they drain together (NFR-RL-07: no
//! task is silently killed mid-flight).

use tokio::sync::watch;

/// A cheap clonable handle observers `.await` to know shutdown was requested.
#[derive(Clone)]
pub struct ShutdownSignal {
    rx: watch::Receiver<bool>,
}

impl ShutdownSignal {
    /// Complete once shutdown has been requested. Cheap to call from many tasks.
    pub async fn wait(mut self) {
        // The initial value is `false`; we return as soon as it flips to `true`.
        while !*self.rx.borrow_and_update() {
            if self.rx.changed().await.is_err() {
                // Sender dropped — treat as shutdown request (defensive).
                return;
            }
        }
    }
}

/// Install signal handlers (ctrl_c always, SIGTERM on unix) and return
/// `(signal, trigger)`. The `signal` is cloned into every listener; calling
/// `trigger()` (or dropping it while the OS delivered a signal) initiates
/// shutdown. Tests call `trigger` directly to end the run without a real
/// signal.
pub fn install_shutdown_signal() -> (ShutdownSignal, ShutdownTrigger) {
    let (tx, rx) = watch::channel(false);
    let signal = ShutdownSignal { rx };
    let trigger = ShutdownTrigger { tx: tx.clone() };

    // Spawn the OS-signal watcher. It races ctrl_c against SIGTERM and flips
    // the watch on whichever arrives first. If both listeners are already
    // shutting down (e.g. tests), this task exits harmlessly on send-error.
    tokio::spawn(async move {
        wait_for_os_signal().await;
        let _ = tx.send(true);
    });

    (signal, trigger)
}

/// Owner-side handle: `trigger()` starts a shutdown without a real OS signal.
/// Used by tests and by `run_with_config` to fold listener errors into the
/// shutdown path.
pub struct ShutdownTrigger {
    tx: watch::Sender<bool>,
}

impl ShutdownTrigger {
    /// Request shutdown. Idempotent.
    pub fn trigger(&self) {
        let _ = self.tx.send(true);
    }
}

#[cfg(unix)]
async fn wait_for_os_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    // Both handlers may fail to install (e.g. in a container with limited
    // capabilities); in that case we fall back to ctrl_c only. That is safer
    // than crashing at boot.
    let mut term = signal(SignalKind::terminate()).ok();
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    if let Some(ref mut term) = term {
        tokio::select! {
            _ = &mut ctrl_c => {},
            _ = term.recv() => {},
        }
    } else {
        let _ = ctrl_c.await;
    }
}

#[cfg(not(unix))]
async fn wait_for_os_signal() {
    // Windows: ctrl_c only. NFR-RL-07 doc: SIGTERM is a POSIX concept; on
    // Windows the equivalent (SERVICE_CONTROL_STOP) is handled by the SCM,
    // not exercised in T03.
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn startup_trigger_wakes_waiters() {
        let (signal, trigger) = install_shutdown_signal();
        let handle = tokio::spawn(async move { signal.wait().await });
        trigger.trigger();
        // Must complete promptly. 1s is generous but bounded so the suite
        // never hangs if the plumbing regresses.
        tokio::time::timeout(std::time::Duration::from_secs(1), handle)
            .await
            .expect("shutdown wait did not resolve after trigger")
            .expect("wait task panicked");
    }
}
