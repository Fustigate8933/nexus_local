//! Hybrid search + ranking for Nexus Local.
//
// Provides a trait for hybrid (vector + lexical) search and ranking.

use async_trait::async_trait;
use anyhow::Result;
use std::path::PathBuf;

/// Query for hybrid search (text, embedding, options).
pub struct HybridSearchQuery {
	pub text: String,
	pub embedding: Option<Vec<f32>>,
	pub top_k: usize,
	// TODO: Add filters, etc.
}

/// Result of a hybrid search.
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
	pub file_path: PathBuf,
	pub score: f32,
	pub snippet: Option<String>,
	pub metadata: Option<String>, // TODO: Replace with richer metadata
}

/// Trait for hybrid search and ranking.
#[async_trait]
pub trait HybridSearch: Send + Sync {
	async fn search(&self, query: HybridSearchQuery) -> Result<Vec<HybridSearchResult>>;
}

// Example stub implementation (to be replaced with real hybrid search backend)
pub struct DummyHybridSearch;

#[async_trait]
impl HybridSearch for DummyHybridSearch {
	async fn search(&self, _query: HybridSearchQuery) -> Result<Vec<HybridSearchResult>> {
		Ok(vec![])
	}
}
