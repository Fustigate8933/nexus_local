use store::{VectorStore, LanceVectorStore, DocumentMetadata};
use std::path::PathBuf;
use std::fs;

/// Create a 384-dimensional test embedding with given seed values
fn make_embedding(seed: &[f32]) -> Vec<f32> {
    let mut emb = vec![0.0f32; 384];
    for (i, &v) in seed.iter().enumerate() {
        if i < 384 {
            emb[i] = v;
        }
    }
    emb
}

#[tokio::test]
async fn test_lance_store_add_and_search() {
    let tmp_dir = std::env::temp_dir().join("nexus_lance_store_test");
    let _ = fs::remove_dir_all(&tmp_dir);
    
    let store = LanceVectorStore::new(tmp_dir.clone()).await.unwrap();
    
    // Add some embeddings
    let meta1 = DocumentMetadata {
        doc_id: String::new(),
        file_path: PathBuf::from("/test/file1.txt"),
        file_type: "txt".to_string(),
        chunk_index: 0,
        snippet: Some("Hello world".to_string()),
    };
    let embedding1 = make_embedding(&[1.0, 0.0, 0.0]);
    let doc_id1 = store.add_embedding(embedding1.clone(), meta1).await.unwrap();
    assert!(!doc_id1.is_empty());
    
    let meta2 = DocumentMetadata {
        doc_id: String::new(),
        file_path: PathBuf::from("/test/file2.txt"),
        file_type: "txt".to_string(),
        chunk_index: 0,
        snippet: Some("Goodbye world".to_string()),
    };
    let embedding2 = make_embedding(&[0.0, 1.0, 0.0]);
    let doc_id2 = store.add_embedding(embedding2.clone(), meta2).await.unwrap();
    
    // Search with query similar to embedding1
    let query = make_embedding(&[0.9, 0.1, 0.0]);
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
    
    let _ = fs::remove_dir_all(&tmp_dir);
}

#[tokio::test]
async fn test_lance_store_persistence() {
    let tmp_dir = std::env::temp_dir().join("nexus_lance_persist_test");
    let _ = fs::remove_dir_all(&tmp_dir);
    
    let doc_id: String;
    
    // Create store and add data
    {
        let store = LanceVectorStore::new(tmp_dir.clone()).await.unwrap();
        let meta = DocumentMetadata {
            doc_id: String::new(),
            file_path: PathBuf::from("/test/persist.txt"),
            file_type: "txt".to_string(),
            chunk_index: 0,
            snippet: Some("Persisted content".to_string()),
        };
        doc_id = store.add_embedding(make_embedding(&[1.0, 2.0, 3.0]), meta).await.unwrap();
        store.save().await.unwrap();
    }
    
    // Reload store and verify data persisted
    {
        let store = LanceVectorStore::new(tmp_dir.clone()).await.unwrap();
        assert_eq!(store.count().await, 1);
        let meta = store.get_metadata(&doc_id).await.unwrap();
        assert!(meta.is_some());
        assert_eq!(meta.unwrap().snippet, Some("Persisted content".to_string()));
    }
    
    let _ = fs::remove_dir_all(&tmp_dir);
}

#[tokio::test]
async fn test_l2_similarity_search() {
    let tmp_dir = std::env::temp_dir().join("nexus_lance_l2_test");
    let _ = fs::remove_dir_all(&tmp_dir);
    
    let store = LanceVectorStore::new(tmp_dir.clone()).await.unwrap();
    
    // Add vectors pointing in different directions
    let mut doc_ids = Vec::new();
    for i in 0..3 {
        let mut seed = vec![0.0f32; 3];
        seed[i] = 1.0;
        let meta = DocumentMetadata {
            doc_id: String::new(),
            file_path: PathBuf::from(format!("/test/file{}.txt", i)),
            file_type: "txt".to_string(),
            chunk_index: 0,
            snippet: Some(format!("Document {}", i)),
        };
        let id = store.add_embedding(make_embedding(&seed), meta).await.unwrap();
        doc_ids.push(id);
    }
    
    // Search for vector aligned with doc-1 (second document)
    let query = make_embedding(&[0.0, 1.0, 0.0]);
    let results = store.search(query, 3).await.unwrap();
    
    // The closest (smallest L2 distance) should be doc_ids[1]
    assert_eq!(results[0].doc_id, doc_ids[1]);
    // Score should be high for exact match (1 / (1 + 0) = 1.0)
    assert!(results[0].score > 0.9, "Exact match should have score > 0.9, got {}", results[0].score);
    
    let _ = fs::remove_dir_all(&tmp_dir);
}
