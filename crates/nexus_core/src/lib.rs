//! Indexing pipeline and orchestration for Nexus Local.
//
// High-level API for orchestrating file indexing, chunking, and embedding.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use async_trait::async_trait;
use anyhow::Result;
use std::ffi::OsStr;
use sysinfo::System;
use rayon::prelude::*;
pub use store::{VectorStore, DocumentMetadata, SearchResult, StateManager, FileState};

/// Options for configuring the indexer.
pub struct IndexOptions {
	pub root: PathBuf,
	pub chunk_size: usize,
	/// Maximum file size to process (bytes). Files larger are skipped.
	pub max_file_size_bytes: u64,
	/// Maximum memory to use (bytes). Used for throttling.
	pub max_memory_bytes: u64,
}

impl Default for IndexOptions {
	fn default() -> Self {
		Self { 
			root: PathBuf::new(), 
			chunk_size: 512,
			max_file_size_bytes: 50 * 1024 * 1024, // 50MB
			max_memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB
		}
	}
}

/// Events emitted during indexing for progress reporting and resumability.
#[derive(Debug)]
pub enum IndexEvent {
	FileStarted(PathBuf),
	FileIndexed(PathBuf),
	FileError(PathBuf, String),
	FileSkipped(PathBuf, String),
	FileUnchanged(PathBuf), // File already indexed and not modified
	MemoryPressure(u64, u64), // (used_mb, limit_mb) - pausing due to memory pressure
	ChunkProcessed(PathBuf, usize),
	ChunkEmbedded(PathBuf, usize, String), // path, chunk_index, doc_id
	Done,
}

/// Summary of the indexing run.
pub struct IndexResult {
	pub files_indexed: usize,
	pub files_skipped: usize,
	pub files_unchanged: usize,
	pub chunks_indexed: usize,
	pub embeddings_stored: usize,
	pub errors: Vec<(PathBuf, String)>,
}

/// Result of garbage collection.
#[derive(Debug, Default)]
pub struct GcResult {
	/// Number of deleted files cleaned up
	pub deleted_files: usize,
	/// Number of modified files with old embeddings cleaned up
	pub modified_files: usize,
	/// Total embeddings removed from store
	pub embeddings_removed: usize,
}

/// Main orchestrator for the indexing pipeline.
/// Uses parallel text extraction with Rayon, followed by batched embedding.
pub struct Indexer<E: SyncTextExtractor, M: Embedder, S: VectorStore> {
	options: IndexOptions,
	extractor: Arc<E>,
	embedder: M,
	store: Arc<S>,
	state: Option<Arc<StateManager>>,
}

impl<E: SyncTextExtractor, M: Embedder, S: VectorStore> Indexer<E, M, S> {
	pub fn new(options: IndexOptions, extractor: E, embedder: M, store: Arc<S>) -> Self {
		Self { options, extractor: Arc::new(extractor), embedder, store, state: None }
	}
	
	/// Set the state manager for incremental indexing.
	pub fn with_state(mut self, state: Arc<StateManager>) -> Self {
		self.state = Some(state);
		self
	}

	/// Run the indexing pipeline (no progress reporting).
	pub async fn run(&mut self) -> Result<IndexResult> {
		self.run_with_progress(|_| ()).await
	}

	/// Run garbage collection to remove embeddings for deleted or modified files.
	/// This should be called before indexing to clean up stale data.
	pub async fn garbage_collect(&self) -> Result<GcResult> {
		let state = match &self.state {
			Some(s) => s,
			None => return Ok(GcResult::default()),
		};

		let mut result = GcResult::default();

		// 1. Clean up embeddings for deleted files
		let deleted_files = state.get_deleted_files()?;
		for path in &deleted_files {
			let doc_ids = state.remove_file(path)?;
			if !doc_ids.is_empty() {
				let removed = self.store.delete_by_doc_ids(&doc_ids).await?;
				result.embeddings_removed += removed;
				result.deleted_files += 1;
			}
		}

		// 2. Clean up old embeddings for modified files (they'll be re-indexed)
		let all_files = state.get_all_files()?;
		for file_info in all_files {
			if file_info.file_state == FileState::Modified && !file_info.doc_ids.is_empty() {
				let removed = self.store.delete_by_doc_ids(&file_info.doc_ids).await?;
				result.embeddings_removed += removed;
				result.modified_files += 1;
			}
		}

		Ok(result)
	}

	/// Run the indexing pipeline, reporting progress via callback.
	/// Uses parallel text extraction with Rayon for improved performance.
	pub async fn run_with_progress<F>(&mut self, mut cb: F) -> Result<IndexResult>
	where
		F: FnMut(IndexEvent) + Send,
	{
		let files = discover_files(&self.options.root)?;
		let chunk_size = self.options.chunk_size;
		let max_file_size = self.options.max_file_size_bytes;
		let max_memory = self.options.max_memory_bytes;

		// Counters for skipped/unchanged (used in parallel phase)
		let files_skipped = AtomicUsize::new(0);
		let files_unchanged = AtomicUsize::new(0);

		// Check memory before starting
		let mut sys = System::new();
		sys.refresh_memory();
		let used_mem = sys.used_memory();
		if used_mem > max_memory {
			let used_mb = used_mem / 1024 / 1024;
			let limit_mb = max_memory / 1024 / 1024;
			cb(IndexEvent::MemoryPressure(used_mb, limit_mb));
			// Continue anyway but warn - parallel extraction will proceed
		}

		// Phase 1: Parallel text extraction with Rayon
		// Each item: (path, chunks, file_type) or error
		let extractor = self.extractor.clone();
		let state = self.state.clone();
		
		let extraction_results: Vec<_> = files
			.par_iter()
			.filter_map(|path| {
				// Check file size
				if let Ok(metadata) = std::fs::metadata(path) {
					if metadata.len() > max_file_size {
						files_skipped.fetch_add(1, Ordering::Relaxed);
						return None;
					}
				}
				
				// Check if file needs indexing
				if let Some(ref state) = state {
					match state.needs_indexing(path) {
						Ok(false) => {
							files_unchanged.fetch_add(1, Ordering::Relaxed);
							return None;
						}
						Ok(true) => {}
						Err(_) => {} // Index anyway on error
					}
				}
				
				// Extract text (sync, CPU-bound)
				match extractor.extract_text_sync(path) {
					Ok(contents) => {
						let chunks = chunk_text(&contents, chunk_size);
						let file_type = path.extension()
							.and_then(|e| e.to_str())
							.unwrap_or("unknown")
							.to_string();
						Some(Ok((path.clone(), chunks, file_type)))
					}
					Err(e) => Some(Err((path.clone(), format!("{}", e))))
				}
			})
			.collect();

		// Phase 2: Sequential embedding and storage
		let mut files_indexed = 0;
		let mut chunks_indexed = 0;
		let mut embeddings_stored = 0;
		let mut errors: Vec<(PathBuf, String)> = vec![];

		for result in extraction_results {
			match result {
				Ok((path, chunks, file_type)) => {
					cb(IndexEvent::FileStarted(path.clone()));
					
					if chunks.is_empty() {
						cb(IndexEvent::FileIndexed(path));
						continue;
					}

					let mut file_doc_ids: Vec<String> = Vec::new();
					let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
					
					match self.embedder.embed_batch(&chunk_refs).await {
						Ok(embeddings) => {
							for (i, (chunk, embedding)) in chunks.iter().zip(embeddings.into_iter()).enumerate() {
								chunks_indexed += 1;
								cb(IndexEvent::ChunkProcessed(path.clone(), i));

								// Safe truncation for UTF-8 strings
								let snippet = if chunk.chars().count() > 200 {
									let truncated: String = chunk.chars().take(200).collect();
									Some(format!("{}...", truncated))
								} else {
									Some(chunk.clone())
								};

								let metadata = DocumentMetadata {
									doc_id: String::new(),
									file_path: path.clone(),
									file_type: file_type.clone(),
									chunk_index: i,
									snippet,
								};

								match self.store.add_embedding(embedding, metadata).await {
									Ok(doc_id) => {
										embeddings_stored += 1;
										file_doc_ids.push(doc_id.clone());
										cb(IndexEvent::ChunkEmbedded(path.clone(), i, doc_id));
									}
									Err(e) => {
										let err_str = format!("Failed to store embedding: {}", e);
										cb(IndexEvent::FileError(path.clone(), err_str.clone()));
										errors.push((path.clone(), err_str));
									}
								}
							}
						}
						Err(e) => {
							let err_str = format!("Embedding failed: {}", e);
							cb(IndexEvent::FileError(path.clone(), err_str.clone()));
							errors.push((path.clone(), err_str));
						}
					}
					
					// Mark file as indexed in state manager
					if !file_doc_ids.is_empty() {
						if let Some(ref state) = self.state {
							if let Ok(meta) = std::fs::metadata(&path) {
								if let Ok(mtime) = meta.modified() {
									if let Err(e) = state.mark_indexed(&path, mtime, &file_doc_ids) {
										eprintln!("  warning: failed to update state for {}: {}", path.display(), e);
									}
								}
							}
						}
						files_indexed += 1;
					}
					
					cb(IndexEvent::FileIndexed(path));
				}
				Err((path, err_str)) => {
					cb(IndexEvent::FileError(path.clone(), err_str.clone()));
					errors.push((path, err_str));
				}
			}
		}

		// Persist the store
		self.store.save().await?;

		cb(IndexEvent::Done);
		Ok(IndexResult {
			files_indexed,
			files_skipped: files_skipped.load(Ordering::Relaxed),
			files_unchanged: files_unchanged.load(Ordering::Relaxed),
			chunks_indexed,
			embeddings_stored,
			errors,
		})
	}
}

/// Recursively discover supported files in a directory.
fn discover_files(root: &PathBuf) -> Result<Vec<PathBuf>> {
	let mut files = Vec::new();
	let supported = ["txt", "md", "pdf", "png", "jpg", "jpeg"];
	for entry in walkdir::WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
		let path = entry.path();
		if path.is_file() {
			if let Some(ext) = path.extension().and_then(OsStr::to_str) {
				if supported.contains(&ext.to_lowercase().as_str()) {
					files.push(path.to_path_buf());
				}
			}
		}
	}
	Ok(files)
}

/// Split text into chunks of roughly `max_len` characters.
fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
	let mut chunks = Vec::new();
	let mut current = String::new();
	for line in text.lines() {
		if current.len() + line.len() > max_len && !current.is_empty() {
			chunks.push(current.clone());
			current.clear();
		}
		if !current.is_empty() {
			current.push('\n');
		}
		current.push_str(line);
	}
	if !current.is_empty() {
		chunks.push(current);
	}
	chunks
}

/// Trait for extracting text from files (plain, PDF, OCR, etc.)
#[async_trait]
pub trait TextExtractor: Send + Sync {
	async fn extract_text(&self, path: &PathBuf) -> Result<String>;
}

/// Sync version of TextExtractor for parallel processing with Rayon.
pub trait SyncTextExtractor: Send + Sync {
	fn extract_text_sync(&self, path: &PathBuf) -> Result<String>;
}

/// Trait for generating embeddings from text.
#[async_trait]
pub trait Embedder: Send + Sync {
	async fn embed(&self, text: &str) -> Result<Vec<f32>>;
	/// Embed multiple texts in a batch for efficiency.
	async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
	/// Return the embedding dimension.
	fn dimension(&self) -> usize;
}

