//! Vector + metadata storage for Nexus Local.
//
// Provides a trait for storing and querying embeddings and metadata.
// Uses a simple in-memory store with disk persistence.

use async_trait::async_trait;
use anyhow::Result;
use std::path::PathBuf;
use std::fs;
use std::collections::HashMap;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Metadata associated with a document or chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
	pub doc_id: String,
	pub file_path: PathBuf,
	pub file_type: String,
	pub chunk_index: usize,
	pub snippet: Option<String>,
}

/// Result of a search query.
#[derive(Debug, Clone)]
pub struct SearchResult {
	pub doc_id: String,
	pub score: f32,
	pub snippet: Option<String>,
	pub metadata: DocumentMetadata,
}

/// Stored embedding entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmbeddingEntry {
	doc_id: String,
	embedding: Vec<f32>,
	metadata: DocumentMetadata,
}

/// Trait for a vector + metadata store.
#[async_trait]
pub trait VectorStore: Send + Sync {
	async fn add_embedding(&self, embedding: Vec<f32>, metadata: DocumentMetadata) -> Result<String>;
	async fn search(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>>;
	async fn get_metadata(&self, doc_id: &str) -> Result<Option<DocumentMetadata>>;
	async fn save(&self) -> Result<()>;
	async fn count(&self) -> usize;
}

/// Local disk-backed vector store using brute-force cosine similarity.
/// Suitable for small-to-medium datasets (thousands of documents).
pub struct LocalVectorStore {
	data_dir: PathBuf,
	entries: RwLock<Vec<EmbeddingEntry>>,
	metadata_index: RwLock<HashMap<String, usize>>,
}

impl LocalVectorStore {
	/// Create or load a LocalVectorStore from the given directory.
	pub fn new(data_dir: PathBuf) -> Result<Self> {
		fs::create_dir_all(&data_dir)?;
		let store_path = data_dir.join("vectors.bin");
		let entries: Vec<EmbeddingEntry> = if store_path.exists() {
			let data = fs::read(&store_path)?;
			bincode::deserialize(&data).unwrap_or_default()
		} else {
			Vec::new()
		};
		let metadata_index: HashMap<String, usize> = entries
			.iter()
			.enumerate()
			.map(|(i, e)| (e.doc_id.clone(), i))
			.collect();
		Ok(Self {
			data_dir,
			entries: RwLock::new(entries),
			metadata_index: RwLock::new(metadata_index),
		})
	}

	/// Compute cosine similarity between two vectors.
	fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
		let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
		let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
		let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
		if norm_a == 0.0 || norm_b == 0.0 {
			0.0
		} else {
			dot / (norm_a * norm_b)
		}
	}
}

#[async_trait]
impl VectorStore for LocalVectorStore {
	async fn add_embedding(&self, embedding: Vec<f32>, metadata: DocumentMetadata) -> Result<String> {
		let doc_id = if metadata.doc_id.is_empty() {
			Uuid::new_v4().to_string()
		} else {
			metadata.doc_id.clone()
		};
		let entry = EmbeddingEntry {
			doc_id: doc_id.clone(),
			embedding,
			metadata: DocumentMetadata { doc_id: doc_id.clone(), ..metadata },
		};
		let mut entries = self.entries.write();
		let idx = entries.len();
		entries.push(entry);
		self.metadata_index.write().insert(doc_id.clone(), idx);
		Ok(doc_id)
	}

	async fn search(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>> {
		let entries = self.entries.read();
		let mut scored: Vec<(f32, &EmbeddingEntry)> = entries
			.iter()
			.map(|e| (Self::cosine_similarity(&query, &e.embedding), e))
			.collect();
		scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
		let results: Vec<SearchResult> = scored
			.into_iter()
			.take(top_k)
			.map(|(score, e)| SearchResult {
				doc_id: e.doc_id.clone(),
				score,
				snippet: e.metadata.snippet.clone(),
				metadata: e.metadata.clone(),
			})
			.collect();
		Ok(results)
	}

	async fn get_metadata(&self, doc_id: &str) -> Result<Option<DocumentMetadata>> {
		let index = self.metadata_index.read();
		if let Some(&idx) = index.get(doc_id) {
			let entries = self.entries.read();
			Ok(entries.get(idx).map(|e| e.metadata.clone()))
		} else {
			Ok(None)
		}
	}

	async fn save(&self) -> Result<()> {
		let entries = self.entries.read();
		let data = bincode::serialize(&*entries)?;
		fs::write(self.data_dir.join("vectors.bin"), data)?;
		Ok(())
	}

	async fn count(&self) -> usize {
		self.entries.read().len()
	}
}

// Stub implementation for testing without persistence
pub struct DummyStore;

#[async_trait]
impl VectorStore for DummyStore {
	async fn add_embedding(&self, _embedding: Vec<f32>, metadata: DocumentMetadata) -> Result<String> {
		Ok(metadata.doc_id)
	}

	async fn search(&self, _query: Vec<f32>, _top_k: usize) -> Result<Vec<SearchResult>> {
		Ok(vec![])
	}

	async fn get_metadata(&self, _doc_id: &str) -> Result<Option<DocumentMetadata>> {
		Ok(None)
	}

	async fn save(&self) -> Result<()> {
		Ok(())
	}

	async fn count(&self) -> usize {
		0
	}
}

