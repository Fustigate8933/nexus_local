//! Vector + metadata storage for Nexus Local.
//!
//! Uses LanceDB for disk-based vector storage with ANN indexing.
//! Scales to millions of embeddings without loading everything into RAM.

use async_trait::async_trait;
use anyhow::{Result, Context};
use std::path::PathBuf;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use lancedb::connect;
use lancedb::query::{QueryBase, ExecutableQuery};
use arrow_array::{
    RecordBatch, RecordBatchIterator, StringArray, Float32Array, Int32Array,
    ArrayRef, Array,
};
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_schema::{Schema, Field, DataType};
use futures::TryStreamExt;
use tokio::sync::RwLock;

/// Metadata associated with a document or chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub doc_id: String,
    pub file_path: PathBuf,
    pub file_type: String,
    pub chunk_index: usize,
    pub snippet: Option<String>,
}

/// Result of a search query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: String,
    pub score: f32,
    pub snippet: Option<String>,
    pub metadata: DocumentMetadata,
}

/// Trait for a vector + metadata store.
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add_embedding(&self, embedding: Vec<f32>, metadata: DocumentMetadata) -> Result<String>;
    async fn search(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>>;
    async fn get_metadata(&self, doc_id: &str) -> Result<Option<DocumentMetadata>>;
    async fn save(&self) -> Result<()>;
    async fn count(&self) -> usize;
}

const TABLE_NAME: &str = "embeddings";
const EMBEDDING_DIM: i32 = 384; // all-MiniLM-L6-v2

/// LanceDB-backed vector store.
/// Data is stored on disk with efficient ANN search.
pub struct LanceVectorStore {
    db: Arc<lancedb::Connection>,
    table: RwLock<Option<lancedb::Table>>,
    #[allow(dead_code)]
    data_dir: PathBuf,
}

impl LanceVectorStore {
    /// Create or open a LanceDB store at the given directory.
    pub async fn new(data_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        let db_path = data_dir.to_string_lossy().to_string();
        let db = connect(&db_path).execute().await
            .context("Failed to connect to LanceDB")?;
        
        // Try to open existing table
        let table = match db.open_table(TABLE_NAME).execute().await {
            Ok(t) => Some(t),
            Err(_) => None, // Table doesn't exist yet
        };
        
        Ok(Self {
            db: Arc::new(db),
            table: RwLock::new(table),
            data_dir,
        })
    }

    /// Get the Arrow schema for the embeddings table.
    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("file_type", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new("snippet", DataType::Utf8, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIM,
                ),
                false,
            ),
        ]))
    }

    /// Create a RecordBatch from a single embedding + metadata.
    fn create_batch(embedding: Vec<f32>, metadata: &DocumentMetadata) -> Result<RecordBatch> {
        let schema = Self::schema();
        
        let doc_id = StringArray::from(vec![metadata.doc_id.as_str()]);
        let file_path = StringArray::from(vec![metadata.file_path.to_string_lossy().to_string()]);
        let file_type = StringArray::from(vec![metadata.file_type.as_str()]);
        let chunk_index = Int32Array::from(vec![metadata.chunk_index as i32]);
        let snippet = StringArray::from(vec![metadata.snippet.as_deref()]);
        
        // Create FixedSizeList for the embedding vector using builder
        let mut list_builder = FixedSizeListBuilder::new(Float32Builder::new(), EMBEDDING_DIM);
        let values_builder = list_builder.values();
        for v in &embedding {
            values_builder.append_value(*v);
        }
        list_builder.append(true);
        let vector = list_builder.finish();
        
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(doc_id) as ArrayRef,
                Arc::new(file_path) as ArrayRef,
                Arc::new(file_type) as ArrayRef,
                Arc::new(chunk_index) as ArrayRef,
                Arc::new(snippet) as ArrayRef,
                Arc::new(vector) as ArrayRef,
            ],
        )?;
        
        Ok(batch)
    }
}

#[async_trait]
impl VectorStore for LanceVectorStore {
    async fn add_embedding(&self, embedding: Vec<f32>, metadata: DocumentMetadata) -> Result<String> {
        let doc_id = if metadata.doc_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            metadata.doc_id.clone()
        };
        
        let metadata = DocumentMetadata { doc_id: doc_id.clone(), ..metadata };
        let batch = Self::create_batch(embedding, &metadata)?;
        
        let mut table_guard = self.table.write().await;
        
        if let Some(ref table) = *table_guard {
            // Add to existing table
            table.add(
                RecordBatchIterator::new(vec![Ok(batch)], Self::schema())
            ).execute().await?;
        } else {
            // Create new table
            let new_table = self.db.create_table(
                TABLE_NAME,
                RecordBatchIterator::new(vec![Ok(batch)], Self::schema()),
            ).execute().await?;
            *table_guard = Some(new_table);
        }
        
        Ok(doc_id)
    }

    async fn search(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>> {
        let table_guard = self.table.read().await;
        
        let table = match &*table_guard {
            Some(t) => t,
            None => return Ok(vec![]), // No table means no results
        };
        
        let results = table
            .vector_search(query)?
            .limit(top_k)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        
        let mut search_results = Vec::new();
        
        for batch in results {
            let doc_ids = batch
                .column_by_name("doc_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let file_paths = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let file_types = batch
                .column_by_name("file_type")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let chunk_indices = batch
                .column_by_name("chunk_index")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let snippets = batch
                .column_by_name("snippet")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            
            if let (Some(doc_ids), Some(file_paths), Some(file_types), Some(chunk_indices), Some(snippets), Some(distances)) 
                = (doc_ids, file_paths, file_types, chunk_indices, snippets, distances) 
            {
                for i in 0..batch.num_rows() {
                    let doc_id = doc_ids.value(i).to_string();
                    let file_path = PathBuf::from(file_paths.value(i));
                    let file_type = file_types.value(i).to_string();
                    let chunk_index = chunk_indices.value(i) as usize;
                    let snippet = if snippets.is_null(i) { None } else { Some(snippets.value(i).to_string()) };
                    let distance = distances.value(i);
                    
                    // Convert L2 distance to similarity score (1 / (1 + distance))
                    let score = 1.0 / (1.0 + distance);
                    
                    search_results.push(SearchResult {
                        doc_id: doc_id.clone(),
                        score,
                        snippet: snippet.clone(),
                        metadata: DocumentMetadata {
                            doc_id,
                            file_path,
                            file_type,
                            chunk_index,
                            snippet,
                        },
                    });
                }
            }
        }
        
        Ok(search_results)
    }

    async fn get_metadata(&self, doc_id: &str) -> Result<Option<DocumentMetadata>> {
        let table_guard = self.table.read().await;
        
        let table = match &*table_guard {
            Some(t) => t,
            None => return Ok(None),
        };
        
        // Support prefix matching for partial doc IDs
        let filter = format!("doc_id LIKE '{}%'", doc_id.replace('\'', "''"));
        let results = table
            .query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        
        for batch in results {
            if batch.num_rows() == 0 {
                continue;
            }
            
            let doc_ids = batch.column_by_name("doc_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let file_paths = batch.column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let file_types = batch.column_by_name("file_type")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let chunk_indices = batch.column_by_name("chunk_index")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let snippets = batch.column_by_name("snippet")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            
            if let (Some(doc_ids), Some(file_paths), Some(file_types), Some(chunk_indices), Some(snippets))
                = (doc_ids, file_paths, file_types, chunk_indices, snippets)
            {
                return Ok(Some(DocumentMetadata {
                    doc_id: doc_ids.value(0).to_string(),
                    file_path: PathBuf::from(file_paths.value(0)),
                    file_type: file_types.value(0).to_string(),
                    chunk_index: chunk_indices.value(0) as usize,
                    snippet: if snippets.is_null(0) { None } else { Some(snippets.value(0).to_string()) },
                }));
            }
        }
        
        Ok(None)
    }

    async fn save(&self) -> Result<()> {
        // LanceDB automatically persists to disk, no explicit save needed
        Ok(())
    }

    async fn count(&self) -> usize {
        let table_guard = self.table.read().await;
        
        match &*table_guard {
            Some(table) => table.count_rows(None).await.unwrap_or(0) as usize,
            None => 0,
        }
    }
}

// Stub implementation for testing without persistence
pub struct DummyStore;

#[async_trait]
impl VectorStore for DummyStore {
    async fn add_embedding(&self, _embedding: Vec<f32>, metadata: DocumentMetadata) -> Result<String> {
        Ok(metadata.doc_id)
    }

    async fn search(&self, _query: Vec<f32>, _top_k: usize) -> Result<Vec<SearchResult>> {
        Ok(vec![])
    }

    async fn get_metadata(&self, _doc_id: &str) -> Result<Option<DocumentMetadata>> {
        Ok(None)
    }

    async fn save(&self) -> Result<()> {
        Ok(())
    }

    async fn count(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_lance_store_basic() {
        let dir = tempdir().unwrap();
        let store = LanceVectorStore::new(dir.path().to_path_buf()).await.unwrap();
        
        // Add an embedding
        let embedding = vec![0.1f32; 384];
        let metadata = DocumentMetadata {
            doc_id: String::new(),
            file_path: PathBuf::from("/test/file.txt"),
            file_type: "txt".to_string(),
            chunk_index: 0,
            snippet: Some("test snippet".to_string()),
        };
        
        let doc_id = store.add_embedding(embedding.clone(), metadata).await.unwrap();
        assert!(!doc_id.is_empty());
        
        // Count should be 1
        assert_eq!(store.count().await, 1);
        
        // Search should return the document
        let results = store.search(embedding, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, doc_id);
    }

    #[tokio::test]
    async fn test_lance_store_get_metadata() {
        let dir = tempdir().unwrap();
        let store = LanceVectorStore::new(dir.path().to_path_buf()).await.unwrap();
        
        let embedding = vec![0.5f32; 384];
        let metadata = DocumentMetadata {
            doc_id: String::new(),
            file_path: PathBuf::from("/test/doc.pdf"),
            file_type: "pdf".to_string(),
            chunk_index: 5,
            snippet: Some("hello world".to_string()),
        };
        
        let doc_id = store.add_embedding(embedding, metadata).await.unwrap();
        
        let retrieved = store.get_metadata(&doc_id).await.unwrap();
        assert!(retrieved.is_some());
        let m = retrieved.unwrap();
        assert_eq!(m.file_type, "pdf");
        assert_eq!(m.chunk_index, 5);
    }
}

