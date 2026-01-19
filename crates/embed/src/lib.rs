//! Embedding abstraction for Nexus Local.
//
// Provides a trait for generating vector embeddings from text.

use std::sync::Mutex;
use async_trait::async_trait;
use anyhow::Result;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

/// Trait for generating embeddings from text.
#[async_trait]
pub trait Embedder: Send + Sync {
	async fn embed(&self, text: &str) -> Result<Vec<f32>>;
	/// Embed multiple texts in a batch for efficiency.
	async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
	/// Return the embedding dimension.
	fn dimension(&self) -> usize;
}

/// Local embedder using fastembed (runs entirely offline).
pub struct LocalEmbedder {
	model: Mutex<TextEmbedding>,
	dim: usize,
}

impl LocalEmbedder {
	/// Create a new LocalEmbedder with the default model (all-MiniLM-L6-v2, 384 dimensions).
	pub fn new() -> Result<Self> {
		let options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
			.with_show_download_progress(true);
		let model = TextEmbedding::try_new(options)?;
		Ok(Self { model: Mutex::new(model), dim: 384 })
	}

	/// Create a LocalEmbedder, optionally with GPU acceleration.
	/// When GPU is requested, tries CUDA first, then falls back to CPU.
	pub fn new_with_options(use_gpu: bool) -> Result<Self> {
		if use_gpu {
			#[cfg(feature = "cuda")]
			{
				use ort::execution_providers::CUDAExecutionProvider;
				use fastembed::ExecutionProviderDispatch;
				
				eprintln!("  Attempting GPU (CUDA) acceleration...");
				
				let cuda_ep: ExecutionProviderDispatch = CUDAExecutionProvider::default().into();
				let options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
					.with_show_download_progress(true)
					.with_execution_providers(vec![cuda_ep]);
				
				match TextEmbedding::try_new(options) {
					Ok(model) => {
						eprintln!("  ✓ CUDA acceleration enabled");
						return Ok(Self { model: Mutex::new(model), dim: 384 });
					}
					Err(e) => {
						eprintln!("  ✗ CUDA init failed: {}", e);
						eprintln!("    Falling back to CPU...");
					}
				}
			}
			
			#[cfg(not(feature = "cuda"))]
			{
				eprintln!("  Note: GPU support requires building with --features cuda");
				eprintln!("        Also requires CUDA toolkit installed on system");
				eprintln!("        Using CPU...");
			}
		}
		
		Self::new()
	}

	/// Create a LocalEmbedder with a specific model.
	pub fn with_model(model_name: EmbeddingModel, dim: usize) -> Result<Self> {
		let options = InitOptions::new(model_name)
			.with_show_download_progress(true);
		let model = TextEmbedding::try_new(options)?;
		Ok(Self { model: Mutex::new(model), dim })
	}
}

#[async_trait]
impl Embedder for LocalEmbedder {
	async fn embed(&self, text: &str) -> Result<Vec<f32>> {
		let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
		let embeddings = model.embed(vec![text], None)?;
		Ok(embeddings.into_iter().next().unwrap_or_default())
	}

	async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
		let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
		let embeddings = model.embed(texts.to_vec(), None)?;
		Ok(embeddings)
	}

	fn dimension(&self) -> usize {
		self.dim
	}
}

// Example stub implementation (for testing without model download)
pub struct DummyEmbedder;

#[async_trait]
impl Embedder for DummyEmbedder {
	async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
		Ok(vec![0.0; 384])
	}

	async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
		Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
	}

	fn dimension(&self) -> usize {
		384
	}
}

