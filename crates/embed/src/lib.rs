//! Embedding abstraction for Nexus Local.
//
// Provides a trait for generating vector embeddings from text.

use async_trait::async_trait;
use anyhow::Result;

/// Trait for generating embeddings from text.
#[async_trait]
pub trait Embedder: Send + Sync {
	async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

// Example stub implementation (to be replaced with real embedding backend)
pub struct DummyEmbedder;

#[async_trait]
impl Embedder for DummyEmbedder {
	async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
		Ok(vec![0.0, 1.0, 2.0])
	}
}
