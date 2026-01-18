use search::{HybridSearch, DummyHybridSearch, HybridSearchQuery};
use anyhow::Result;
use async_trait::async_trait;

#[tokio::test]
async fn test_dummy_hybrid_search() -> Result<()> {
    let searcher = DummyHybridSearch;
    let query = HybridSearchQuery {
        text: "test query".to_string(),
        embedding: Some(vec![1.0, 2.0, 3.0]),
        top_k: 5,
    };
    let results = searcher.search(query).await?;
    assert!(results.is_empty()); // DummyHybridSearch always returns empty
    Ok(())
}
