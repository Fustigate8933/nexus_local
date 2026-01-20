//! Watch mode for automatic re-indexing on file changes.
//!
//! Uses the `notify` crate to watch directories for file system events.
//! Changes are debounced to avoid re-indexing on every keystroke.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;
use std::collections::HashSet;

use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use anyhow::Result;

use crate::config::WatchConfig;

/// File watcher that monitors directories for changes.
pub struct FileWatcher {
    watcher: RecommendedWatcher,
    receiver: Receiver<Result<Event, notify::Error>>,
    config: WatchConfig,
    watched_roots: Vec<PathBuf>,
}

/// A batch of changed files after debouncing.
#[derive(Debug, Clone)]
pub struct ChangeBatch {
    /// Files that were created or modified.
    pub modified: Vec<PathBuf>,
    /// Files that were deleted.
    pub deleted: Vec<PathBuf>,
}

impl FileWatcher {
    /// Create a new file watcher with the given configuration.
    pub fn new(config: WatchConfig) -> Result<Self> {
        let (tx, rx) = channel();
        
        let watcher = RecommendedWatcher::new(
            move |res| {
                let _ = tx.send(res);
            },
            Config::default().with_poll_interval(Duration::from_secs(1)),
        )?;
        
        Ok(Self {
            watcher,
            receiver: rx,
            config,
            watched_roots: vec![],
        })
    }

    /// Start watching a directory recursively.
    pub fn watch(&mut self, path: &PathBuf) -> Result<()> {
        self.watcher.watch(path, RecursiveMode::Recursive)?;
        self.watched_roots.push(path.clone());
        eprintln!("  watching: {}", path.display());
        Ok(())
    }

    /// Stop watching a directory.
    pub fn unwatch(&mut self, path: &PathBuf) -> Result<()> {
        self.watcher.unwatch(path)?;
        self.watched_roots.retain(|p| p != path);
        Ok(())
    }

    /// Wait for file changes and return a debounced batch.
    /// Blocks until changes are detected, then waits for `debounce_secs` of quiet.
    pub fn wait_for_changes(&self) -> Result<ChangeBatch> {
        let mut modified = HashSet::new();
        let mut deleted = HashSet::new();
        
        // Wait for first event
        let first_event = self.receiver.recv()?;
        self.process_event(first_event, &mut modified, &mut deleted);
        
        // Debounce: collect all events within the debounce window
        let debounce = Duration::from_secs(self.config.debounce_secs);
        loop {
            match self.receiver.recv_timeout(debounce) {
                Ok(event) => {
                    self.process_event(event, &mut modified, &mut deleted);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // Debounce period elapsed, return the batch
                    break;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("Watcher channel disconnected");
                }
            }
        }
        
        // Remove files that were both modified and deleted (deleted wins)
        for path in &deleted {
            modified.remove(path);
        }
        
        Ok(ChangeBatch {
            modified: modified.into_iter().collect(),
            deleted: deleted.into_iter().collect(),
        })
    }

    /// Process a single event into modified/deleted sets.
    fn process_event(
        &self,
        event: Result<Event, notify::Error>,
        modified: &mut HashSet<PathBuf>,
        deleted: &mut HashSet<PathBuf>,
    ) {
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                eprintln!("  watch error: {:?}", e);
                return;
            }
        };
        
        for path in event.paths {
            // Skip directories
            if path.is_dir() {
                continue;
            }
            
            // Check ignore patterns
            if self.should_ignore(&path) {
                continue;
            }
            
            use notify::EventKind;
            match event.kind {
                EventKind::Create(_) | EventKind::Modify(_) => {
                    modified.insert(path);
                }
                EventKind::Remove(_) => {
                    deleted.insert(path);
                }
                _ => {}
            }
        }
    }

    /// Check if a path matches any ignore pattern.
    fn should_ignore(&self, path: &PathBuf) -> bool {
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        
        for pattern in &self.config.ignore_patterns {
            if Self::glob_match(pattern, filename) {
                return true;
            }
        }
        
        false
    }

    /// Simple glob matching (supports * and ?).
    fn glob_match(pattern: &str, text: &str) -> bool {
        let pattern_chars: Vec<char> = pattern.chars().collect();
        let text_chars: Vec<char> = text.chars().collect();
        Self::glob_match_helper(&pattern_chars, &text_chars)
    }

    fn glob_match_helper(pattern: &[char], text: &[char]) -> bool {
        match (pattern.first(), text.first()) {
            (None, None) => true,
            (Some('*'), _) => {
                // * matches zero or more characters
                Self::glob_match_helper(&pattern[1..], text) ||
                (!text.is_empty() && Self::glob_match_helper(pattern, &text[1..]))
            }
            (Some('?'), Some(_)) => {
                // ? matches exactly one character
                Self::glob_match_helper(&pattern[1..], &text[1..])
            }
            (Some(p), Some(t)) if *p == *t => {
                Self::glob_match_helper(&pattern[1..], &text[1..])
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        assert!(FileWatcher::glob_match("*.tmp", "file.tmp"));
        assert!(FileWatcher::glob_match("*.tmp", ".tmp"));
        assert!(!FileWatcher::glob_match("*.tmp", "file.txt"));
        assert!(FileWatcher::glob_match("*~", "file.txt~"));
        assert!(FileWatcher::glob_match(".#*", ".#file"));
        assert!(FileWatcher::glob_match("*.swp", ".file.swp"));
        assert!(FileWatcher::glob_match("test?", "test1"));
        assert!(!FileWatcher::glob_match("test?", "test12"));
    }
}
