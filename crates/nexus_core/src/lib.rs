//! Indexing pipeline and orchestration for Nexus Local.
//
// High-level API for orchestrating file indexing, chunking, and embedding.

use std::path::PathBuf;
use async_trait::async_trait;
use anyhow::Result;
use std::fs;
use std::ffi::OsStr;
use futures::stream::{self, StreamExt};

/// Options for configuring the indexer.
pub struct IndexOptions {
	pub root: PathBuf,
	// TODO: Add more options (file filters, chunk size, etc.)
}

/// Events emitted during indexing for progress reporting and resumability.
#[derive(Debug)]
pub enum IndexEvent {
	FileStarted(PathBuf),
	FileIndexed(PathBuf),
	FileError(PathBuf, String),
	ChunkProcessed(PathBuf, usize),
	Done,
}

/// Summary of the indexing run.
pub struct IndexResult {
	pub files_indexed: usize,
	pub chunks_indexed: usize,
	pub errors: Vec<(PathBuf, String)>,
}

/// Main orchestrator for the indexing pipeline.
pub struct Indexer<E: TextExtractor, M: Embedder> {
	options: IndexOptions,
	extractor: E,
	embedder: M,
}

impl<E: TextExtractor, M: Embedder> Indexer<E, M> {
	pub fn new(options: IndexOptions, extractor: E, embedder: M) -> Self {
		Self { options, extractor, embedder }
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
		let mut chunks_indexed = 0;
		let mut errors = vec![];

		let files = discover_files(&self.options.root)?;
		let mut stream = stream::iter(files)
			.map(|path| async move {
				(path.clone(), fs::read_to_string(&path))
			})
			.buffer_unordered(4);

		while let Some((path, res)) = stream.next().await {
			cb(IndexEvent::FileStarted(path.clone()));
			match res {
				Ok(contents) => {
					files_indexed += 1;
					let chunks = chunk_text(&contents, 512);
					for (i, _chunk) in chunks.iter().enumerate() {
						chunks_indexed += 1;
						cb(IndexEvent::ChunkProcessed(path.clone(), i));
						// TODO: Call embedder and store
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
		cb(IndexEvent::Done);
		Ok(IndexResult {
			files_indexed,
			chunks_indexed,
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
}
