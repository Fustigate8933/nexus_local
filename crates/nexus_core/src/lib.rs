//! Indexing pipeline and orchestration for Nexus Local.
//
// High-level API for orchestrating file indexing, chunking, and embedding.

use std::path::PathBuf;
use std::sync::Arc;
use async_trait::async_trait;
use anyhow::Result;
use std::ffi::OsStr;
pub use store::{VectorStore, DocumentMetadata, SearchResult};

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
	ChunkProcessed(PathBuf, usize),
	ChunkEmbedded(PathBuf, usize, String), // path, chunk_index, doc_id
	Done,
}

/// Summary of the indexing run.
pub struct IndexResult {
	pub files_indexed: usize,
	pub files_skipped: usize,
	pub chunks_indexed: usize,
	pub embeddings_stored: usize,
	pub errors: Vec<(PathBuf, String)>,
}

/// Main orchestrator for the indexing pipeline.
pub struct Indexer<E: TextExtractor, M: Embedder, S: VectorStore> {
	options: IndexOptions,
	extractor: E,
	embedder: M,
	store: Arc<S>,
}

impl<E: TextExtractor, M: Embedder, S: VectorStore> Indexer<E, M, S> {
	pub fn new(options: IndexOptions, extractor: E, embedder: M, store: Arc<S>) -> Self {
		Self { options, extractor, embedder, store }
	}

	/// Run the indexing pipeline (no progress reporting).
	pub async fn run(&mut self) -> Result<IndexResult> {
		self.run_with_progress(|_| ()).await
	}

	/// Run the indexing pipeline, reporting progress via callback.
	pub async fn run_with_progress<F>(&mut self, mut cb: F) -> Result<IndexResult>
	where
		F: FnMut(IndexEvent) + Send,
	{
		let mut files_indexed = 0;
		let mut files_skipped = 0;
		let mut chunks_indexed = 0;
		let mut embeddings_stored = 0;
		let mut errors = vec![];

		let files = discover_files(&self.options.root)?;
		let chunk_size = self.options.chunk_size;
		let max_file_size = self.options.max_file_size_bytes;

		// Process files sequentially to allow mutable borrow of embedder
		for path in files {
			// Check file size before processing
			if let Ok(metadata) = std::fs::metadata(&path) {
				if metadata.len() > max_file_size {
					let reason = format!("file too large ({}MB > {}MB limit)", 
						metadata.len() / 1024 / 1024,
						max_file_size / 1024 / 1024);
					cb(IndexEvent::FileSkipped(path.clone(), reason));
					files_skipped += 1;
					continue;
				}
			}

			cb(IndexEvent::FileStarted(path.clone()));

			match self.extractor.extract_text(&path).await {
				Ok(contents) => {
					files_indexed += 1;
					let chunks = chunk_text(&contents, chunk_size);

					// Batch embed for efficiency
					if !chunks.is_empty() {
						let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
						match self.embedder.embed_batch(&chunk_refs).await {
							Ok(embeddings) => {
								for (i, (chunk, embedding)) in chunks.iter().zip(embeddings.into_iter()).enumerate() {
									chunks_indexed += 1;
									cb(IndexEvent::ChunkProcessed(path.clone(), i));

									// Store embedding with metadata
									let file_type = path.extension()
										.and_then(|e| e.to_str())
										.unwrap_or("unknown")
										.to_string();

									// Safe truncation for UTF-8 strings
									let snippet = if chunk.chars().count() > 200 {
										let truncated: String = chunk.chars().take(200).collect();
										Some(format!("{}...", truncated))
									} else {
										Some(chunk.clone())
									};

									let metadata = DocumentMetadata {
										doc_id: String::new(), // Will be assigned by store
										file_path: path.clone(),
										file_type,
										chunk_index: i,
										snippet,
									};

									match self.store.add_embedding(embedding, metadata).await {
										Ok(doc_id) => {
											embeddings_stored += 1;
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
					}
					cb(IndexEvent::FileIndexed(path));
				}
				Err(ref e) => {
					let err_str = format!("{}", e);
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
			files_skipped,
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

/// Trait for generating embeddings from text.
#[async_trait]
pub trait Embedder: Send + Sync {
	async fn embed(&self, text: &str) -> Result<Vec<f32>>;
	/// Embed multiple texts in a batch for efficiency.
	async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
	/// Return the embedding dimension.
	fn dimension(&self) -> usize;
}

