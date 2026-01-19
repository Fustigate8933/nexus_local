extern crate nexus_core;
use nexus_core::{IndexOptions, Indexer, TextExtractor, Embedder, IndexEvent, VectorStore, DocumentMetadata, SearchResult};
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;

struct DummyExtractor;
#[async_trait]
impl TextExtractor for DummyExtractor {
    async fn extract_text(&self, _path: &PathBuf) -> Result<String> {
        Ok("dummy text".to_string())
    }
}

struct DummyEmbedder;
#[async_trait]
impl Embedder for DummyEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.0, 1.0, 2.0])
    }
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0, 1.0, 2.0]).collect())
    }
    fn dimension(&self) -> usize { 3 }
}

struct DummyStore;
#[async_trait]
impl VectorStore for DummyStore {
    async fn add_embedding(&self, _embedding: Vec<f32>, mut metadata: DocumentMetadata) -> Result<String> {
        let id = format!("doc-{}", metadata.chunk_index);
        metadata.doc_id = id.clone();
        Ok(id)
    }
    async fn search(&self, _query: Vec<f32>, _top_k: usize) -> Result<Vec<SearchResult>> {
        Ok(vec![])
    }
    async fn get_metadata(&self, _doc_id: &str) -> Result<Option<DocumentMetadata>> {
        Ok(None)
    }
    async fn save(&self) -> Result<()> { Ok(()) }
    async fn count(&self) -> usize { 0 }
}

#[tokio::test]
async fn test_file_discovery_and_chunking() -> Result<()> {
    let root = PathBuf::from(".");
    let options = IndexOptions { 
        root, 
        chunk_size: 512,
        max_file_size_bytes: 50 * 1024 * 1024,
        max_memory_bytes: 4 * 1024 * 1024 * 1024,
    };
    let extractor = DummyExtractor;
    let embedder = DummyEmbedder;
    let store = Arc::new(DummyStore);
    let mut indexer = Indexer::new(options, extractor, embedder, store);
    let mut events = Vec::new();
    let result = indexer.run_with_progress(|e| events.push(e)).await?;
    // Should finish and emit Done event
    assert!(events.iter().any(|e| matches!(e, IndexEvent::Done)));
    // Should not panic or error
    assert!(result.errors.is_empty());
    Ok(())
}
