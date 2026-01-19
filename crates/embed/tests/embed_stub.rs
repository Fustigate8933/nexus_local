use embed::{Embedder, DummyEmbedder};
use anyhow::Result;

#[tokio::test]
async fn test_dummy_embedder() -> Result<()> {
    let embedder = DummyEmbedder;
    let vec = embedder.embed("hello world").await?;
    assert_eq!(vec.len(), 384);
    assert_eq!(embedder.dimension(), 384);
    Ok(())
}
