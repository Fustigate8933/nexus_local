extern crate nexus_core;
use nexus_core::{IndexOptions, Indexer, TextExtractor, Embedder, IndexEvent};
use std::path::PathBuf;
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
}

#[tokio::test]
async fn test_file_discovery_and_chunking() -> Result<()> {
    let root = PathBuf::from(".");
    let options = IndexOptions { root };
    let extractor = DummyExtractor;
    let embedder = DummyEmbedder;
    let mut indexer = Indexer::new(options, extractor, embedder);
    let mut events = Vec::new();
    let result = indexer.run_with_progress(|e| events.push(e)).await?;
    // Should finish and emit Done event
    assert!(events.iter().any(|e| matches!(e, IndexEvent::Done)));
    // Should not panic or error
    assert!(result.errors.is_empty());
    Ok(())
}
