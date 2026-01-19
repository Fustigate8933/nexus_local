//! State Manager for tracking indexed files.
//!
//! Uses SQLite to track:
//! - File paths and modification timestamps
//! - Which files have been indexed and when
//! - Doc IDs associated with each file (for garbage collection)

use anyhow::{Result, Context};
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

/// File state in the index
#[derive(Debug, Clone, PartialEq)]
pub enum FileState {
    /// File has never been indexed
    NotIndexed,
    /// File is indexed and up-to-date
    Indexed,
    /// File has been modified since last index
    Modified,
    /// File was indexed but has been deleted from disk
    Deleted,
}

/// Information about an indexed file
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: PathBuf,
    pub file_state: FileState,
    pub indexed_at: Option<i64>,
    pub file_mtime: Option<i64>,
    pub doc_ids: Vec<String>,
}

/// SQLite-based state manager for tracking indexed files.
pub struct StateManager {
    conn: Mutex<Connection>,
}

impl StateManager {
    /// Create or open the state database at the given directory.
    pub fn new(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let db_path = data_dir.join("state.db");
        let conn = Connection::open(&db_path)
            .context("Failed to open state database")?;
        
        // Create tables
        conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                file_mtime INTEGER NOT NULL,
                indexed_at INTEGER NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS file_docs (
                path TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                PRIMARY KEY (path, doc_id),
                FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
            );
            
            CREATE INDEX IF NOT EXISTS idx_file_docs_path ON file_docs(path);
            CREATE INDEX IF NOT EXISTS idx_file_docs_doc_id ON file_docs(doc_id);
        "#).context("Failed to create tables")?;
        
        Ok(Self { conn: Mutex::new(conn) })
    }
    
    /// Mark a file as indexed with its current modification time.
    /// Also records the doc_ids generated for this file.
    pub fn mark_indexed(&self, path: &Path, mtime: SystemTime, doc_ids: &[String]) -> Result<()> {
        let mtime_secs = mtime
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        
        let path_str = path.to_string_lossy().to_string();
        let conn = self.conn.lock().unwrap();
        
        // Upsert file record
        conn.execute(
            "INSERT INTO files (path, file_mtime, indexed_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(path) DO UPDATE SET file_mtime = ?2, indexed_at = ?3",
            params![path_str, mtime_secs, now],
        )?;
        
        // Clear old doc_ids and insert new ones
        conn.execute("DELETE FROM file_docs WHERE path = ?1", params![path_str])?;
        
        for doc_id in doc_ids {
            conn.execute(
                "INSERT INTO file_docs (path, doc_id) VALUES (?1, ?2)",
                params![path_str, doc_id],
            )?;
        }
        
        Ok(())
    }
    
    /// Get the state of a file.
    pub fn get_file_state(&self, path: &Path) -> Result<FileState> {
        let path_str = path.to_string_lossy().to_string();
        let conn = self.conn.lock().unwrap();
        
        // Check if file exists in database
        let result: Option<i64> = conn
            .query_row(
                "SELECT file_mtime FROM files WHERE path = ?1",
                params![path_str],
                |row| row.get(0),
            )
            .ok();
        
        match result {
            None => Ok(FileState::NotIndexed),
            Some(stored_mtime) => {
                // Check if file still exists on disk
                if !path.exists() {
                    return Ok(FileState::Deleted);
                }
                
                // Get current mtime
                let current_mtime = path
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                
                if current_mtime > stored_mtime {
                    Ok(FileState::Modified)
                } else {
                    Ok(FileState::Indexed)
                }
            }
        }
    }
    
    /// Check if a file needs (re)indexing.
    pub fn needs_indexing(&self, path: &Path) -> Result<bool> {
        let state = self.get_file_state(path)?;
        Ok(matches!(state, FileState::NotIndexed | FileState::Modified))
    }
    
    /// Get all doc_ids for a file (for deletion during re-indexing or garbage collection).
    pub fn get_doc_ids(&self, path: &Path) -> Result<Vec<String>> {
        let path_str = path.to_string_lossy().to_string();
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare("SELECT doc_id FROM file_docs WHERE path = ?1")?;
        let doc_ids: Vec<String> = stmt
            .query_map(params![path_str], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        
        Ok(doc_ids)
    }
    
    /// Get all files that are marked as deleted (exist in DB but not on disk).
    pub fn get_deleted_files(&self) -> Result<Vec<PathBuf>> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare("SELECT path FROM files")?;
        let paths: Vec<PathBuf> = stmt
            .query_map([], |row| {
                let path_str: String = row.get(0)?;
                Ok(PathBuf::from(path_str))
            })?
            .filter_map(|r| r.ok())
            .filter(|p| !p.exists())
            .collect();
        
        Ok(paths)
    }
    
    /// Remove a file from the state database (after garbage collection).
    pub fn remove_file(&self, path: &Path) -> Result<Vec<String>> {
        let path_str = path.to_string_lossy().to_string();
        let conn = self.conn.lock().unwrap();
        
        // Get doc_ids before deletion
        let mut stmt = conn.prepare("SELECT doc_id FROM file_docs WHERE path = ?1")?;
        let doc_ids: Vec<String> = stmt
            .query_map(params![path_str], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);
        
        // Delete from both tables (cascade should handle file_docs)
        conn.execute("DELETE FROM file_docs WHERE path = ?1", params![path_str])?;
        conn.execute("DELETE FROM files WHERE path = ?1", params![path_str])?;
        
        Ok(doc_ids)
    }
    
    /// Get total number of tracked files.
    pub fn file_count(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;
        Ok(count as usize)
    }
    
    /// Get all tracked files with their info.
    pub fn get_all_files(&self) -> Result<Vec<FileInfo>> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare("SELECT path, file_mtime, indexed_at FROM files")?;
        let files: Vec<(String, i64, i64)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);
        
        let mut result = Vec::new();
        for (path_str, file_mtime, indexed_at) in files {
            let path = PathBuf::from(&path_str);
            
            // Get doc_ids
            let mut stmt = conn.prepare("SELECT doc_id FROM file_docs WHERE path = ?1")?;
            let doc_ids: Vec<String> = stmt
                .query_map(params![path_str], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            drop(stmt);
            
            // Determine state
            let file_state = if !path.exists() {
                FileState::Deleted
            } else {
                let current_mtime = path
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                
                if current_mtime > file_mtime {
                    FileState::Modified
                } else {
                    FileState::Indexed
                }
            };
            
            result.push(FileInfo {
                path,
                file_state,
                indexed_at: Some(indexed_at),
                file_mtime: Some(file_mtime),
                doc_ids,
            });
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    
    #[test]
    fn test_state_manager_basic() {
        let tmp = TempDir::new().unwrap();
        let state = StateManager::new(tmp.path()).unwrap();
        
        // Create a test file
        let test_file = tmp.path().join("test.txt");
        fs::write(&test_file, "hello").unwrap();
        let mtime = test_file.metadata().unwrap().modified().unwrap();
        
        // Initially not indexed
        assert_eq!(state.get_file_state(&test_file).unwrap(), FileState::NotIndexed);
        assert!(state.needs_indexing(&test_file).unwrap());
        
        // Mark as indexed
        let doc_ids = vec!["doc1".to_string(), "doc2".to_string()];
        state.mark_indexed(&test_file, mtime, &doc_ids).unwrap();
        
        // Now indexed
        assert_eq!(state.get_file_state(&test_file).unwrap(), FileState::Indexed);
        assert!(!state.needs_indexing(&test_file).unwrap());
        
        // Check doc_ids
        let stored_ids = state.get_doc_ids(&test_file).unwrap();
        assert_eq!(stored_ids.len(), 2);
        assert!(stored_ids.contains(&"doc1".to_string()));
        
        // File count
        assert_eq!(state.file_count().unwrap(), 1);
    }
    
    #[test]
    fn test_deleted_file_detection() {
        let tmp = TempDir::new().unwrap();
        let state = StateManager::new(tmp.path()).unwrap();
        
        // Create and index a file
        let test_file = tmp.path().join("delete_me.txt");
        fs::write(&test_file, "temporary").unwrap();
        let mtime = test_file.metadata().unwrap().modified().unwrap();
        state.mark_indexed(&test_file, mtime, &["doc1".to_string()]).unwrap();
        
        // Delete the file
        fs::remove_file(&test_file).unwrap();
        
        // Should be detected as deleted
        assert_eq!(state.get_file_state(&test_file).unwrap(), FileState::Deleted);
        
        // Should appear in deleted files list
        let deleted = state.get_deleted_files().unwrap();
        assert_eq!(deleted.len(), 1);
        assert_eq!(deleted[0], test_file);
        
        // Remove from state
        let removed_ids = state.remove_file(&test_file).unwrap();
        assert_eq!(removed_ids, vec!["doc1".to_string()]);
        
        // Now not tracked
        assert_eq!(state.file_count().unwrap(), 0);
    }
}
