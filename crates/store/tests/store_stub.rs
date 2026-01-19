use store::{VectorStore, DummyStore, DocumentMetadata};
use anyhow::Result;
use std::path::PathBuf;

#[tokio::test]
async fn test_dummy_store_add_and_search() -> Result<()> {
    let store = DummyStore;
    let meta = DocumentMetadata {
        doc_id: "doc1".to_string(),
        file_path: PathBuf::from("file.txt"),
        file_type: "txt".to_string(),
        chunk_index: 0,
        snippet: None,
    };
    store.add_embedding(vec![1.0, 2.0, 3.0], meta.clone()).await?;
    let results = store.search(vec![1.0, 2.0, 3.0], 5).await?;
    assert!(results.is_empty()); // DummyStore always returns empty
    let meta_result = store.get_metadata("doc1").await?;
    assert!(meta_result.is_none()); // DummyStore always returns None
    Ok(())
}
