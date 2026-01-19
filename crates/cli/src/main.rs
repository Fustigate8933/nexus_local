//! CLI entrypoint for Nexus Local.


use clap::{Parser, Subcommand};
use anyhow::Result;
use nexus_core::{IndexOptions, Indexer, Embedder, IndexEvent, TextExtractor, VectorStore};
use ocr::{PlainTextExtractor, OcrEngine};
use embed::{LocalEmbedder, Embedder as EmbedderTrait};
use store::{LanceVectorStore, StateManager};
use std::path::PathBuf;
use std::sync::Arc;
use async_trait::async_trait;
use sysinfo::System;

#[derive(Parser)]
#[command(name = "nexus")]
#[command(about = "Nexus Local: Local-first, privacy-preserving second brain", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index a directory
    Index {
        path: String,
        /// Maximum memory usage in MB (default: 75% of system RAM)
        #[arg(long)]
        max_memory_mb: Option<u64>,
        /// Skip files larger than this size in MB (default: 50)
        #[arg(long, default_value = "50")]
        max_file_mb: u64,
    },
    /// Show indexer/search status
    Status,
    /// Search for a query
    Search {
        query: String,
        #[arg(long)]
        json: bool,
    },
    /// Explain a document by ID
    Explain {
        doc_id: String,
    },
}

/// Wrapper to adapt PlainTextExtractor (OcrEngine) to TextExtractor trait.
struct OcrExtractor(PlainTextExtractor);

#[async_trait]
impl TextExtractor for OcrExtractor {
    async fn extract_text(&self, path: &PathBuf) -> anyhow::Result<String> {
        self.0.extract_text(path).await
    }
}

/// Wrapper to adapt LocalEmbedder to nexus_core::Embedder trait.
struct EmbedWrapper(LocalEmbedder);

#[async_trait]
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

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { path, max_memory_mb, max_file_mb } => {
            // Get system memory info
            let sys = System::new_all();
            let total_mem_mb = sys.total_memory() / 1024 / 1024;
            let max_mem = max_memory_mb.unwrap_or(total_mem_mb * 3 / 4);
            
            eprintln!("info: indexing {}", path);
            eprintln!("info: memory limit {}MB (system: {}MB)\", max file: {}MB", 
                max_mem, total_mem_mb, max_file_mb);

            // Initialize data directory
            let data_dir = dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("nexus_local");
            std::fs::create_dir_all(&data_dir)?;

            eprintln!("info: loading embedding model...");
            let embedder = LocalEmbedder::new()?;
            eprintln!("info: model loaded (dim={})", embedder.dimension());

            eprintln!("info: opening store at {:?}", data_dir);
            let store = Arc::new(LanceVectorStore::new(data_dir.clone()).await?);
            eprintln!("info: {} existing embeddings", store.count().await);

            // Initialize state manager
            let state = Arc::new(StateManager::new(&data_dir)?);
            eprintln!("info: state manager ready");

            let options = IndexOptions { 
                root: PathBuf::from(&path), 
                chunk_size: 512,
                max_file_size_bytes: max_file_mb * 1024 * 1024,
                max_memory_bytes: max_mem * 1024 * 1024,
            };
            let extractor = OcrExtractor(PlainTextExtractor);
            let embedder = EmbedWrapper(embedder);
            let indexer = Indexer::new(options, extractor, embedder, store.clone())
                .with_state(state);

            // Run garbage collection first to clean up stale embeddings
            eprintln!("info: running garbage collection...");
            let gc_result = indexer.garbage_collect().await?;
            if gc_result.embeddings_removed > 0 {
                eprintln!("  gc: removed {} embeddings ({} deleted files, {} modified files)",
                    gc_result.embeddings_removed,
                    gc_result.deleted_files,
                    gc_result.modified_files
                );
            }

            let mut indexer = indexer; // Make mutable for run_with_progress
            let mut memory_skipped = 0usize;
            let result = indexer.run_with_progress(|e| {
                match &e {
                    IndexEvent::FileStarted(p) => eprintln!("  processing {}", p.display()),
                    IndexEvent::FileIndexed(p) => eprintln!("  indexed {}", p.display()),
                    IndexEvent::FileSkipped(_, reason) if reason.contains("memory pressure") => {
                        memory_skipped += 1;
                    }
                    IndexEvent::FileSkipped(p, reason) => eprintln!("  skipped {} ({})", p.display(), reason),
                    IndexEvent::FileUnchanged(p) => eprintln!("  unchanged {}", p.display()),
                    IndexEvent::MemoryPressure(_, _) => {} // Handled via FileSkipped
                    IndexEvent::ChunkEmbedded(_, i, id) => eprintln!("    chunk {} -> {}", i, &id[..8]),
                    IndexEvent::FileError(p, err) => eprintln!("  error: {} - {}", p.display(), err),
                    IndexEvent::Done => {},
                    _ => {}
                }
            }).await?;

            eprintln!("done: {} indexed, {} unchanged, {} skipped, {} chunks, {} embeddings, {} errors",
                result.files_indexed,
                result.files_unchanged,
                result.files_skipped,
                result.chunks_indexed,
                result.embeddings_stored,
                result.errors.len()
            );
            if memory_skipped > 0 {
                eprintln!("warning: {} files skipped due to memory pressure", memory_skipped);
                eprintln!("  hint: increase limit with --max-memory-mb or re-run later");
            }
            eprintln!("info: total embeddings in store: {}", store.count().await);
        }
        Commands::Status => {
            // Initialize data directory
            let data_dir = dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("nexus_local");

            if !data_dir.exists() {
                eprintln!("error: no index found, run 'nexus index <path>' first");
                return Ok(());
            }

            let store = Arc::new(LanceVectorStore::new(data_dir.clone()).await?);
            let count = store.count().await;
            println!("nexus status");
            println!("  store: {:?}", data_dir);
            println!("  embeddings: {}", count);
        }
        Commands::Search { query, json } => {
            // Initialize data directory
            let data_dir = dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("nexus_local");

            if !data_dir.exists() {
                eprintln!("error: no index found, run 'nexus index <path>' first");
                return Ok(());
            }

            // Load embedder and store
            let embedder = LocalEmbedder::new()?;
            let store = Arc::new(LanceVectorStore::new(data_dir).await?);

            // Embed the query
            let query_embedding = embedder.embed(&query).await?;

            // Search
            let results = store.search(query_embedding, 5).await?;

            if json {
                // JSON output
                let json_results: Vec<_> = results.iter().map(|r| {
                    serde_json::json!({
                        "doc_id": r.doc_id,
                        "score": r.score,
                        "file_path": r.metadata.file_path,
                        "chunk_index": r.metadata.chunk_index,
                        "snippet": r.snippet
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&json_results)?);
            } else {
                // Human-readable output
                println!("search: \"{}\"", query);

                if results.is_empty() {
                    println!("  (no results)");
                } else {
                    for (i, result) in results.iter().enumerate() {
                        println!();
                        println!("  {}. {} (score: {:.4})", 
                            i + 1, 
                            result.metadata.file_path.display(),
                            result.score
                        );
                        println!("     chunk {} | id {}", 
                            result.metadata.chunk_index, 
                            &result.doc_id[..8]
                        );
                        if let Some(snippet) = &result.snippet {
                            let preview: String = snippet.chars().take(80).collect();
                            println!("     > {}...", preview.replace('\n', " "));
                        }
                    }
                    println!();
                }
            }
        }
        Commands::Explain { doc_id } => {
            // Initialize data directory
            let data_dir = dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("nexus_local");

            if !data_dir.exists() {
                eprintln!("error: no index found, run 'nexus index <path>' first");
                return Ok(());
            }

            let store = Arc::new(LanceVectorStore::new(data_dir).await?);

            // Find matching documents (partial ID match)
            if let Some(meta) = store.get_metadata(&doc_id).await? {
                println!("document: {}", doc_id);
                println!("  path: {}", meta.file_path.display());
                println!("  type: {}", meta.file_type);
                println!("  chunk: {}", meta.chunk_index);
                if let Some(snippet) = &meta.snippet {
                    println!("  content:");
                    for line in snippet.lines() {
                        println!("    {}", line);
                    }
                }
            } else {
                eprintln!("error: document not found: {}", doc_id);
            }
        }
    }
    Ok(())
}
