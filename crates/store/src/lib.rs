//! Vector + metadata storage for Nexus Local.
//
// Provides a trait for storing and querying embeddings and metadata.

use async_trait::async_trait;
use anyhow::Result;
use std::path::PathBuf;

/// Metadata associated with a document or chunk.
#[derive(Debug, Clone)]
pub struct DocumentMetadata {
	pub doc_id: String,
	pub file_path: PathBuf,
	pub file_type: String,
	pub chunk_index: usize,
	// TODO: Add more fields as needed (snippet, etc.)
}

/// Result of a search query.
#[derive(Debug, Clone)]
pub struct SearchResult {
	pub doc_id: String,
	pub score: f32,
	pub snippet: Option<String>,
	pub metadata: DocumentMetadata,
}

/// Trait for a vector + metadata store.
#[async_trait]
pub trait VectorStore: Send + Sync {
	async fn add_embedding(&self, embedding: Vec<f32>, metadata: DocumentMetadata) -> Result<()>;
	async fn search(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>>;
	async fn get_metadata(&self, doc_id: &str) -> Result<Option<DocumentMetadata>>;
}

// Example stub implementation (to be replaced with real LanceDB backend)
pub struct DummyStore;

#[async_trait]
impl VectorStore for DummyStore {
	async fn add_embedding(&self, _embedding: Vec<f32>, _metadata: DocumentMetadata) -> Result<()> {
		Ok(())
	}

	async fn search(&self, _query: Vec<f32>, _top_k: usize) -> Result<Vec<SearchResult>> {
		Ok(vec![])
	}

	async fn get_metadata(&self, _doc_id: &str) -> Result<Option<DocumentMetadata>> {
		Ok(None)
	}
}
