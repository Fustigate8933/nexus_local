//! Lexical (full-text) search index using Tantivy.
//!
//! Provides BM25-based keyword search to complement vector similarity search.

use anyhow::{Result, Context};
use std::path::PathBuf;
use std::sync::RwLock;
use tantivy::{
    schema::{Schema, STRING, STORED, Field, TextOptions, TextFieldIndexing, IndexRecordOption, Value},
    Index, IndexWriter, IndexReader, TantivyDocument,
    query::QueryParser,
    collector::TopDocs,
};

/// A document stored in the lexical index.
#[derive(Debug, Clone)]
pub struct LexicalDoc {
    pub doc_id: String,
    pub file_path: String,
    pub content: String,
    pub chunk_index: usize,
}

/// Result of a lexical search.
#[derive(Debug, Clone)]
pub struct LexicalSearchResult {
    pub doc_id: String,
    pub file_path: String,
    pub chunk_index: usize,
    pub score: f32,
}

/// Tantivy-based lexical (BM25) search index.
pub struct LexicalIndex {
    index: Index,
    writer: RwLock<IndexWriter>,
    reader: RwLock<IndexReader>,
    // Schema fields
    doc_id_field: Field,
    file_path_field: Field,
    content_field: Field,
    chunk_index_field: Field,
}

impl LexicalIndex {
    /// Create or open a lexical index at the given directory.
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        let index_path = data_dir.join("tantivy_index");
        std::fs::create_dir_all(&index_path)?;
        
        // Build schema
        let mut schema_builder = Schema::builder();
        
        // doc_id: stored and indexed for exact match lookup
        let doc_id_field = schema_builder.add_text_field("doc_id", STRING | STORED);
        
        // file_path: stored for retrieval
        let file_path_field = schema_builder.add_text_field("file_path", STRING | STORED);
        
        // content: full-text indexed with positions for phrase queries
        // NOT stored - we use LanceDB snippets for display, this is just for BM25 scoring
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
            );
        let content_field = schema_builder.add_text_field("content", text_options);
        
        // chunk_index: stored as text (Tantivy doesn't have native i32 in older versions)
        let chunk_index_field = schema_builder.add_text_field("chunk_index", STRING | STORED);
        
        let schema = schema_builder.build();
        
        // Open or create index
        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path)
                .context("Failed to open existing Tantivy index")?
        } else {
            Index::create_in_dir(&index_path, schema.clone())
                .context("Failed to create Tantivy index")?
        };
        
        // Create writer with 50MB heap
        let writer = index.writer(50_000_000)
            .context("Failed to create index writer")?;
        
        let reader = index.reader()
            .context("Failed to create index reader")?;
        
        Ok(Self {
            index,
            writer: RwLock::new(writer),
            reader: RwLock::new(reader),
            doc_id_field,
            file_path_field,
            content_field,
            chunk_index_field,
        })
    }
    
    /// Add a document to the lexical index.
    pub fn add_document(&self, doc: LexicalDoc) -> Result<()> {
        let writer = self.writer.write()
            .map_err(|e| anyhow::anyhow!("Writer lock poisoned: {}", e))?;
        
        let mut tantivy_doc = TantivyDocument::default();
        tantivy_doc.add_text(self.doc_id_field, &doc.doc_id);
        tantivy_doc.add_text(self.file_path_field, &doc.file_path);
        tantivy_doc.add_text(self.content_field, &doc.content);
        tantivy_doc.add_text(self.chunk_index_field, &doc.chunk_index.to_string());
        
        writer.add_document(tantivy_doc)?;
        Ok(())
    }
    
    /// Add multiple documents in batch.
    pub fn add_documents(&self, docs: Vec<LexicalDoc>) -> Result<()> {
        let writer = self.writer.write()
            .map_err(|e| anyhow::anyhow!("Writer lock poisoned: {}", e))?;
        
        for doc in docs {
            let mut tantivy_doc = TantivyDocument::default();
            tantivy_doc.add_text(self.doc_id_field, &doc.doc_id);
            tantivy_doc.add_text(self.file_path_field, &doc.file_path);
            tantivy_doc.add_text(self.content_field, &doc.content);
            tantivy_doc.add_text(self.chunk_index_field, &doc.chunk_index.to_string());
            
            writer.add_document(tantivy_doc)?;
        }
        Ok(())
    }
    
    /// Commit pending changes to the index.
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.write()
            .map_err(|e| anyhow::anyhow!("Writer lock poisoned: {}", e))?;
        writer.commit()?;
        
        // Reload reader to see new documents
        let reader = self.reader.write()
            .map_err(|e| anyhow::anyhow!("Reader lock poisoned: {}", e))?;
        reader.reload()?;
        
        Ok(())
    }
    
    /// Search for documents matching the query.
    pub fn search(&self, query_str: &str, top_k: usize) -> Result<Vec<LexicalSearchResult>> {
        let reader = self.reader.read()
            .map_err(|e| anyhow::anyhow!("Reader lock poisoned: {}", e))?;
        
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        
        // Parse query, fall back to match-all if empty
        let query = if query_str.trim().is_empty() {
            return Ok(vec![]);
        } else {
            query_parser.parse_query(query_str)
                .unwrap_or_else(|_| {
                    // If query parsing fails, try as a simple term query
                    Box::new(tantivy::query::AllQuery)
                })
        };
        
        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;
        
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            
            let doc_id = doc.get_first(self.doc_id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            
            let file_path = doc.get_first(self.file_path_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            
            let chunk_index: usize = doc.get_first(self.chunk_index_field)
                .and_then(|v| v.as_str())
                .and_then(|s: &str| s.parse().ok())
                .unwrap_or(0);
            
            results.push(LexicalSearchResult {
                doc_id,
                file_path,
                chunk_index,
                score,
            });
        }
        
        Ok(results)
    }
    
    /// Delete documents by their doc_ids.
    pub fn delete_by_doc_ids(&self, doc_ids: &[String]) -> Result<usize> {
        if doc_ids.is_empty() {
            return Ok(0);
        }
        
        let writer = self.writer.write()
            .map_err(|e| anyhow::anyhow!("Writer lock poisoned: {}", e))?;
        
        let mut deleted = 0;
        for doc_id in doc_ids {
            let term = tantivy::Term::from_field_text(self.doc_id_field, doc_id);
            writer.delete_term(term);
            deleted += 1;
        }
        
        Ok(deleted)
    }
    
    /// Get the number of documents in the index.
    pub fn count(&self) -> Result<usize> {
        let reader = self.reader.read()
            .map_err(|e| anyhow::anyhow!("Reader lock poisoned: {}", e))?;
        let searcher = reader.searcher();
        Ok(searcher.num_docs() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_lexical_index_basic() {
        let dir = tempdir().unwrap();
        let index = LexicalIndex::new(dir.path().to_path_buf()).unwrap();
        
        // Add a document
        index.add_document(LexicalDoc {
            doc_id: "doc1".to_string(),
            file_path: "/test/file.txt".to_string(),
            content: "The quick brown fox jumps over the lazy dog".to_string(),
            chunk_index: 0,
        }).unwrap();
        
        index.commit().unwrap();
        
        // Search for "fox"
        let results = index.search("fox", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "doc1");
        
        // Search for "cat" (not in document)
        let results = index.search("cat", 10).unwrap();
        assert_eq!(results.len(), 0);
    }
    
    #[test]
    fn test_lexical_index_multiple_docs() {
        let dir = tempdir().unwrap();
        let index = LexicalIndex::new(dir.path().to_path_buf()).unwrap();
        
        index.add_documents(vec![
            LexicalDoc {
                doc_id: "doc1".to_string(),
                file_path: "/a.txt".to_string(),
                content: "Rust programming language".to_string(),
                chunk_index: 0,
            },
            LexicalDoc {
                doc_id: "doc2".to_string(),
                file_path: "/b.txt".to_string(),
                content: "Python programming language".to_string(),
                chunk_index: 0,
            },
            LexicalDoc {
                doc_id: "doc3".to_string(),
                file_path: "/c.txt".to_string(),
                content: "JavaScript web development".to_string(),
                chunk_index: 0,
            },
        ]).unwrap();
        
        index.commit().unwrap();
        
        // Search for "programming" - should match doc1 and doc2
        let results = index.search("programming", 10).unwrap();
        assert_eq!(results.len(), 2);
        
        // Search for "Rust" - should match doc1
        let results = index.search("Rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "doc1");
    }
}
