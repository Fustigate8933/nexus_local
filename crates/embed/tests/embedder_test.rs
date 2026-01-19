use embed::{Embedder, LocalEmbedder, DummyEmbedder};

#[tokio::test]
async fn test_dummy_embedder() {
    let embedder = DummyEmbedder;
    let vec = embedder.embed("hello world").await.unwrap();
    assert_eq!(vec.len(), 384);
    assert_eq!(embedder.dimension(), 384);
}

#[tokio::test]
async fn test_dummy_embedder_batch() {
    let embedder = DummyEmbedder;
    let vecs = embedder.embed_batch(&["hello", "world"]).await.unwrap();
    assert_eq!(vecs.len(), 2);
    assert_eq!(vecs[0].len(), 384);
}

#[tokio::test]
async fn test_local_embedder() {
    // This test downloads the model on first run (~23MB)
    let embedder = LocalEmbedder::new();
    if embedder.is_err() {
        eprintln!("Skipping LocalEmbedder test (model download may have failed)");
        return;
    }
    let embedder = embedder.unwrap();
    let vec = embedder.embed("The quick brown fox jumps over the lazy dog.").await.unwrap();
    assert_eq!(vec.len(), 384);
    assert_eq!(embedder.dimension(), 384);
    // Check that the vector is not all zeros (i.e., real embedding)
    assert!(vec.iter().any(|&x| x != 0.0), "Embedding should not be all zeros");
}

#[tokio::test]
async fn test_local_embedder_batch() {
    let embedder = LocalEmbedder::new();
    if embedder.is_err() {
        eprintln!("Skipping LocalEmbedder batch test");
        return;
    }
    let embedder = embedder.unwrap();
    let vecs = embedder.embed_batch(&["hello", "world", "rust"]).await.unwrap();
    assert_eq!(vecs.len(), 3);
    for v in &vecs {
        assert_eq!(v.len(), 384);
    }
}
