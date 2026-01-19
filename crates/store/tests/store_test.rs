use store::{VectorStore, LocalVectorStore, DocumentMetadata};
use std::path::PathBuf;
use std::fs;

#[tokio::test]
async fn test_local_store_add_and_search() {
    let tmp_dir = std::env::temp_dir().join("nexus_store_test");
    let _ = fs::remove_dir_all(&tmp_dir);
    
    let store = LocalVectorStore::new(tmp_dir.clone()).unwrap();
    
    // Add some embeddings
    let meta1 = DocumentMetadata {
        doc_id: String::new(),
        file_path: PathBuf::from("/test/file1.txt"),
        file_type: "txt".to_string(),
        chunk_index: 0,
        snippet: Some("Hello world".to_string()),
    };
    let embedding1 = vec![1.0, 0.0, 0.0];
    let doc_id1 = store.add_embedding(embedding1.clone(), meta1).await.unwrap();
    assert!(!doc_id1.is_empty());
    
    let meta2 = DocumentMetadata {
        doc_id: String::new(),
        file_path: PathBuf::from("/test/file2.txt"),
        file_type: "txt".to_string(),
        chunk_index: 0,
        snippet: Some("Goodbye world".to_string()),
    };
    let embedding2 = vec![0.0, 1.0, 0.0];
    let doc_id2 = store.add_embedding(embedding2.clone(), meta2).await.unwrap();
    
    // Search with query similar to embedding1
    let query = vec![0.9, 0.1, 0.0];
    let results = store.search(query, 2).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].doc_id, doc_id1, "First result should be most similar");
    assert!(results[0].score > results[1].score);
    
    // Test count
    assert_eq!(store.count().await, 2);
    
    // Test get_metadata
    let meta = store.get_metadata(&doc_id1).await.unwrap();
    assert!(meta.is_some());
    assert_eq!(meta.unwrap().file_path, PathBuf::from("/test/file1.txt"));
    
    fs::remove_dir_all(&tmp_dir).unwrap();
}

#[tokio::test]
async fn test_local_store_persistence() {
    let tmp_dir = std::env::temp_dir().join("nexus_store_persist_test");
    let _ = fs::remove_dir_all(&tmp_dir);
    
    // Create store and add data
    {
        let store = LocalVectorStore::new(tmp_dir.clone()).unwrap();
        let meta = DocumentMetadata {
            doc_id: "test-doc-1".to_string(),
            file_path: PathBuf::from("/test/persist.txt"),
            file_type: "txt".to_string(),
            chunk_index: 0,
            snippet: Some("Persisted content".to_string()),
        };
        store.add_embedding(vec![1.0, 2.0, 3.0], meta).await.unwrap();
        store.save().await.unwrap();
    }
    
    // Reload store and verify data persisted
    {
        let store = LocalVectorStore::new(tmp_dir.clone()).unwrap();
        assert_eq!(store.count().await, 1);
        let meta = store.get_metadata("test-doc-1").await.unwrap();
        assert!(meta.is_some());
        assert_eq!(meta.unwrap().snippet, Some("Persisted content".to_string()));
    }
    
    fs::remove_dir_all(&tmp_dir).unwrap();
}

#[tokio::test]
async fn test_cosine_similarity_search() {
    let tmp_dir = std::env::temp_dir().join("nexus_store_cosine_test");
    let _ = fs::remove_dir_all(&tmp_dir);
    
    let store = LocalVectorStore::new(tmp_dir.clone()).unwrap();
    
    // Add orthogonal vectors
    for i in 0..3 {
        let mut embedding = vec![0.0; 3];
        embedding[i] = 1.0;
        let meta = DocumentMetadata {
            doc_id: format!("doc-{}", i),
            file_path: PathBuf::from(format!("/test/file{}.txt", i)),
            file_type: "txt".to_string(),
            chunk_index: 0,
            snippet: Some(format!("Document {}", i)),
        };
        store.add_embedding(embedding, meta).await.unwrap();
    }
    
    // Search for vector aligned with doc-1
    let query = vec![0.0, 1.0, 0.0];
    let results = store.search(query, 3).await.unwrap();
    assert_eq!(results[0].doc_id, "doc-1");
    assert!((results[0].score - 1.0).abs() < 0.001, "Perfect match should have score ~1.0");
    assert!((results[1].score - 0.0).abs() < 0.001, "Orthogonal should have score ~0.0");
    
    fs::remove_dir_all(&tmp_dir).unwrap();
}
