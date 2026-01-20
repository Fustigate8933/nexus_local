use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tauri::Emitter;

use nexus_core::{
    IndexOptions, Indexer, Embedder, IndexEvent, SyncTextExtractor, VectorStore, 
    PagedExtractor, ExtractedPage, LexicalIndex
};
use ocr::{PlainTextExtractor, SyncOcrEngine};
use embed::{LocalEmbedder, Embedder as EmbedderTrait};
use store::{LanceVectorStore, StateManager};

// Result types for frontend
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub file_path: String,
    pub chunk_index: usize,
    pub snippet: Option<String>,
    pub score: f32,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexStatus {
    pub store_path: String,
    pub vector_embeddings: u64,
    pub lexical_documents: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexProgress {
    pub files_indexed: usize,
    pub files_unchanged: usize,
    pub files_skipped: usize,
    pub chunks_indexed: usize,
    pub embeddings_stored: usize,
    pub errors: Vec<String>,
}

// Wrapper to adapt PlainTextExtractor to SyncTextExtractor trait
struct OcrExtractor(PlainTextExtractor);

impl SyncTextExtractor for OcrExtractor {
    fn extract_text_sync(&self, path: &PathBuf) -> anyhow::Result<String> {
        self.0.extract_text_sync(path)
    }
}

impl PagedExtractor for OcrExtractor {
    fn extract_pages(&self, path: &PathBuf) -> anyhow::Result<Vec<ExtractedPage>> {
        ocr::PagedExtractor::extract_pages(&self.0, path)
    }
    
    fn is_paged(&self, path: &PathBuf) -> bool {
        ocr::PagedExtractor::is_paged(&self.0, path)
    }
}

// Wrapper to adapt LocalEmbedder to nexus_core::Embedder trait
struct EmbedWrapper(LocalEmbedder);

#[async_trait::async_trait]
impl Embedder for EmbedWrapper {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.0.embed(text).await
    }
    async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.0.embed_batch(texts).await
    }
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
}

#[tauri::command]
async fn search(
    query: String,
    mode: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<SearchResult>, String> {
    let mode = mode.unwrap_or_else(|| "hybrid".to_string());
    let limit = limit.unwrap_or(5);

    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("nexus_local");

    if !data_dir.exists() {
        return Err("No index found. Please index a directory first.".to_string());
    }

    let embedder = LocalEmbedder::new()
        .map_err(|e| format!("Failed to load embedder: {}", e))?;
    let store = Arc::new(LanceVectorStore::new(data_dir.clone()).await
        .map_err(|e| format!("Failed to open store: {}", e))?);
    let lexical = LexicalIndex::new(data_dir)
        .map_err(|e| format!("Failed to open lexical index: {}", e))?;

    let results = match mode.as_str() {
        "semantic" | "vector" => {
            let query_embedding = embedder.embed(&query).await
                .map_err(|e| format!("Failed to embed query: {}", e))?;
            let vector_results = store.search(query_embedding, limit).await
                .map_err(|e| format!("Failed to search: {}", e))?;
            vector_results.into_iter().map(|r| SearchResult {
                doc_id: r.doc_id,
                file_path: r.metadata.file_path.to_string_lossy().to_string(),
                chunk_index: r.metadata.chunk_index,
                snippet: r.snippet,
                score: r.score,
                source: "semantic".to_string(),
            }).collect()
        }
        "lexical" | "keyword" => {
            let lexical_results = lexical.search(&query, limit)
                .map_err(|e| format!("Failed to search: {}", e))?;
            let mut results = Vec::new();
            for r in lexical_results {
                let snippet = store.get_metadata(&r.doc_id).await
                    .ok()
                    .flatten()
                    .and_then(|m| m.snippet);
                results.push(SearchResult {
                    doc_id: r.doc_id,
                    file_path: r.file_path,
                    chunk_index: r.chunk_index,
                    snippet,
                    score: r.score,
                    source: "lexical".to_string(),
                });
            }
            results
        }
        "hybrid" | _ => {
            let query_embedding = embedder.embed(&query).await
                .map_err(|e| format!("Failed to embed query: {}", e))?;
            let vector_results = store.search(query_embedding, limit * 2).await
                .map_err(|e| format!("Failed to search: {}", e))?;
            let lexical_results = lexical.search(&query, limit * 2)
                .map_err(|e| format!("Failed to search: {}", e))?;
            
            // Apply Reciprocal Rank Fusion (RRF)
            let k = 60.0;
            let mut doc_scores: std::collections::HashMap<String, (f32, Option<String>, PathBuf, usize)> = 
                std::collections::HashMap::new();
            
            for (rank, r) in vector_results.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                let entry = doc_scores.entry(r.doc_id.clone()).or_insert((
                    0.0,
                    r.snippet.clone(),
                    r.metadata.file_path.clone(),
                    r.metadata.chunk_index,
                ));
                entry.0 += rrf_score;
            }
            
            for (rank, r) in lexical_results.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                let entry = doc_scores.entry(r.doc_id.clone()).or_insert((
                    0.0,
                    None,
                    PathBuf::from(&r.file_path),
                    r.chunk_index,
                ));
                entry.0 += rrf_score;
            }
            
            let mut sorted: Vec<_> = doc_scores.into_iter().collect();
            sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap_or(std::cmp::Ordering::Equal));
            
            sorted.into_iter()
                .take(limit)
                .map(|(doc_id, (score, snippet, file_path, chunk_index))| SearchResult {
                    doc_id,
                    file_path: file_path.to_string_lossy().to_string(),
                    chunk_index,
                    snippet,
                    score,
                    source: "hybrid".to_string(),
                })
                .collect()
        }
    };

    Ok(results)
}

#[tauri::command]
async fn get_status() -> Result<IndexStatus, String> {
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("nexus_local");

    if !data_dir.exists() {
        return Ok(IndexStatus {
            store_path: data_dir.to_string_lossy().to_string(),
            vector_embeddings: 0,
            lexical_documents: 0,
        });
    }

    let store = Arc::new(LanceVectorStore::new(data_dir.clone()).await
        .map_err(|e| format!("Failed to open store: {}", e))?);
    let lexical = LexicalIndex::new(data_dir.clone())
        .map_err(|e| format!("Failed to open lexical index: {}", e))?;

    let count = store.count().await;
    let lexical_count = lexical.count().unwrap_or(0);

    Ok(IndexStatus {
        store_path: data_dir.to_string_lossy().to_string(),
        vector_embeddings: count as u64,
        lexical_documents: lexical_count as u64,
    })
}

#[tauri::command]
async fn index_directory(
    app: tauri::AppHandle,
    path: String,
    gpu: Option<bool>,
    max_file_mb: Option<u64>,
    max_memory_mb: Option<u64>,
) -> Result<IndexProgress, String> {
    let path = shellexpand::tilde(&path).to_string();
    let root = PathBuf::from(&path);

    if !root.exists() {
        return Err(format!("Directory does not exist: {}", path));
    }

    let gpu = gpu.unwrap_or(false);
    let max_file_mb = max_file_mb.unwrap_or(50);
    let max_memory_mb = max_memory_mb.unwrap_or_else(|| {
        let sys = sysinfo::System::new_all();
        (sys.total_memory() / 1024 / 1024 * 3 / 4) as u64
    });

    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("nexus_local");
    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("Failed to create data directory: {}", e))?;

    let embedder = LocalEmbedder::new_with_options(gpu)
        .map_err(|e| format!("Failed to load embedder: {}", e))?;
    let store = Arc::new(LanceVectorStore::new(data_dir.clone()).await
        .map_err(|e| format!("Failed to open store: {}", e))?);
    let state = Arc::new(StateManager::new(&data_dir)
        .map_err(|e| format!("Failed to create state manager: {}", e))?);
    let lexical = Arc::new(LexicalIndex::new(data_dir.clone())
        .map_err(|e| format!("Failed to create lexical index: {}", e))?);

    let options = IndexOptions {
        root,
        chunk_size: 1500,
        max_file_size_bytes: max_file_mb * 1024 * 1024,
        max_memory_bytes: max_memory_mb * 1024 * 1024,
        max_chunks_per_file: 500,
        skip_extensions: vec![],
        skip_files: vec![],
    };

    let extractor = OcrExtractor(PlainTextExtractor);
    let embed_wrapper = EmbedWrapper(embedder);
    let indexer = Indexer::new(options, extractor, embed_wrapper, store.clone())
        .with_state(state)
        .with_lexical(lexical);

    // Run garbage collection first
    let _ = indexer.garbage_collect().await;

    let app_handle = app.clone();
    let mut indexer = indexer;
    let result = indexer.run_with_progress(move |event| {
        let app = app_handle.clone();
        let event_name = "index-progress".to_string();
        
        let payload = match event {
            IndexEvent::FileStarted(path) => {
                serde_json::json!({
                    "type": "file-started",
                    "path": path.to_string_lossy().to_string()
                })
            }
            IndexEvent::FileIndexed(path) => {
                serde_json::json!({
                    "type": "file-indexed",
                    "path": path.to_string_lossy().to_string()
                })
            }
            IndexEvent::FileSkipped(path, reason) => {
                serde_json::json!({
                    "type": "file-skipped",
                    "path": path.to_string_lossy().to_string(),
                    "reason": reason
                })
            }
            IndexEvent::FileUnchanged(path) => {
                serde_json::json!({
                    "type": "file-unchanged",
                    "path": path.to_string_lossy().to_string()
                })
            }
            IndexEvent::ChunkEmbedded(_, _, _) => {
                serde_json::json!({
                    "type": "chunk-embedded"
                })
            }
            IndexEvent::PageProcessed(path, page, total) => {
                serde_json::json!({
                    "type": "page-processed",
                    "path": path.to_string_lossy().to_string(),
                    "page": page,
                    "total": total
                })
            }
            IndexEvent::FileError(path, error) => {
                serde_json::json!({
                    "type": "error",
                    "path": path.to_string_lossy().to_string(),
                    "error": error
                })
            }
            IndexEvent::Done => {
                serde_json::json!({
                    "type": "done"
                })
            }
            _ => return, // Skip other events
        };

        // Emit event to frontend
        let _ = app.emit(&event_name, payload);
    }).await.map_err(|e| format!("Indexing failed: {}", e))?;

    // Emit final done event
    let _ = app.emit("index-progress", serde_json::json!({ "type": "done" }));

    Ok(IndexProgress {
        files_indexed: result.files_indexed,
        files_unchanged: result.files_unchanged,
        files_skipped: result.files_skipped,
        chunks_indexed: result.chunks_indexed,
        embeddings_stored: result.embeddings_stored,
        errors: result.errors.into_iter().map(|(_, e)| e).collect(),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            search,
            get_status,
            index_directory,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
