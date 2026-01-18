//! Indexing pipeline and orchestration for Nexus Local.
//
// High-level API for orchestrating file indexing, chunking, and embedding.

use std::path::PathBuf;
use async_trait::async_trait;
use anyhow::Result;

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
	FileError(PathBuf, anyhow::Error),
	ChunkProcessed(PathBuf, usize),
	Done,
}

/// Summary of the indexing run.
pub struct IndexResult {
	pub files_indexed: usize,
	pub chunks_indexed: usize,
	pub errors: Vec<(PathBuf, anyhow::Error)>,
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
		// TODO: Walk directory, extract text, chunk, embed, store
		cb(IndexEvent::Done);
		Ok(IndexResult {
			files_indexed: 0,
			chunks_indexed: 0,
			errors: vec![],
		})
	}
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
