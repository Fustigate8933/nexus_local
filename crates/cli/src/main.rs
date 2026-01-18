//! CLI entrypoint for Nexus Local.


use clap::{Parser, Subcommand};
use anyhow::Result;
use nexus_core::{IndexOptions, Indexer, TextExtractor, Embedder, IndexEvent};
use std::path::PathBuf;
use async_trait::async_trait;

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

struct DummyExtractor;
#[async_trait]
impl TextExtractor for DummyExtractor {
    async fn extract_text(&self, _path: &PathBuf) -> anyhow::Result<String> {
        Ok("dummy text".to_string())
    }
}

struct DummyEmbedder;
#[async_trait]
impl Embedder for DummyEmbedder {
    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0, 1.0, 2.0])
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { path } => {
            println!("Indexing directory: {}", path);
            let options = IndexOptions { root: PathBuf::from(path) };
            let extractor = DummyExtractor;
            let embedder = DummyEmbedder;
            let mut indexer = Indexer::new(options, extractor, embedder);
            let mut events = Vec::new();
            let result = indexer.run_with_progress(|e| {
                println!("Event: {:?}", e);
                events.push(e);
            }).await?;
            println!("Indexed {} files, {} chunks, {} errors", result.files_indexed, result.chunks_indexed, result.errors.len());
        }
        Commands::Status => {
            println!("Indexer/search status: (stub)");
        }
        Commands::Search { query, json } => {
            println!("Searching for: {} (json: {})", query, json);
        }
        Commands::Explain { doc_id } => {
            println!("Explaining document: {}", doc_id);
        }
    }
    Ok(())
}
