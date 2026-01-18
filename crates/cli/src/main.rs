//! CLI entrypoint for Nexus Local.

use clap::{Parser, Subcommand};
use anyhow::Result;

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

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { path } => {
            println!("Indexing directory: {}", path);
            // TODO: Call core indexing logic
        }
        Commands::Status => {
            println!("Indexer/search status: (stub)");
            // TODO: Show status
        }
        Commands::Search { query, json } => {
            println!("Searching for: {} (json: {})", query, json);
            // TODO: Call search logic
        }
        Commands::Explain { doc_id } => {
            println!("Explaining document: {}", doc_id);
            // TODO: Call explain logic
        }
    }
    Ok(())
}
