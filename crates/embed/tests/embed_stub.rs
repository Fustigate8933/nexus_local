use embed::{Embedder, DummyEmbedder};
use anyhow::Result;
use async_trait::async_trait;

#[tokio::test]
async fn test_dummy_embedder() -> Result<()> {
    let embedder = DummyEmbedder;
    let vec = embedder.embed("hello world").await?;
    assert_eq!(vec, vec![0.0, 1.0, 2.0]);
    Ok(())
}
