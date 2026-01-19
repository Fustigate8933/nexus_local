//! Indexing pipeline and orchestration for Nexus Local.
//
// High-level API for orchestrating file indexing, chunking, and embedding.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use async_trait::async_trait;
use anyhow::Result;
use std::ffi::OsStr;
use sysinfo::System;
use rayon::prelude::*;
pub use store::{VectorStore, DocumentMetadata, SearchResult, StateManager, FileState, LexicalIndex, LexicalDoc, LexicalSearchResult};
// Re-export paged extraction types from ocr crate
pub use ocr::{ExtractedPage, PagedExtractor};

/// Options for configuring the indexer.
pub struct IndexOptions {
	pub root: PathBuf,
	pub chunk_size: usize,
	/// Maximum file size to process (bytes). Files larger are skipped.
	pub max_file_size_bytes: u64,
	/// Maximum memory to use (bytes). Used for throttling.
	pub max_memory_bytes: u64,
	/// Maximum chunks per file. Files generating more chunks are skipped.
	/// Prevents dictionary/wordlist files from creating thousands of embeddings.
	pub max_chunks_per_file: usize,
	/// File extensions to skip (e.g., ["png", "jpg"] to skip images).
	pub skip_extensions: Vec<String>,
	/// File name patterns to skip (substring match).
	pub skip_files: Vec<String>,
}

impl Default for IndexOptions {
	fn default() -> Self {
		Self { 
			root: PathBuf::new(), 
			chunk_size: 1500, // ~375 tokens, good balance of context vs granularity
			max_file_size_bytes: 50 * 1024 * 1024, // 50MB
			max_memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB
			max_chunks_per_file: 500, // Skip files that would create >500 chunks
			skip_extensions: Vec::new(),
			skip_files: Vec::new(),
		}
	}
}

/// Events emitted during indexing for progress reporting and resumability.
#[derive(Debug)]
pub enum IndexEvent {
	FileStarted(PathBuf),
	FileIndexed(PathBuf),
	FileError(PathBuf, String),
	FileSkipped(PathBuf, String),
	FileUnchanged(PathBuf), // File already indexed and not modified
	MemoryPressure(u64, u64), // (used_mb, limit_mb) - pausing due to memory pressure
	PageProcessed(PathBuf, usize, usize), // (path, page_num, total_pages)
	ChunkProcessed(PathBuf, usize),
	ChunkEmbedded(PathBuf, usize, String), // path, chunk_index, doc_id
	Done,
}

/// Summary of the indexing run.
pub struct IndexResult {
	pub files_indexed: usize,
	pub files_skipped: usize,
	pub files_unchanged: usize,
	pub chunks_indexed: usize,
	pub embeddings_stored: usize,
	pub errors: Vec<(PathBuf, String)>,
}

/// Result of garbage collection.
#[derive(Debug, Default)]
pub struct GcResult {
	/// Number of deleted files cleaned up
	pub deleted_files: usize,
	/// Number of modified files with old embeddings cleaned up
	pub modified_files: usize,
	/// Total embeddings removed from store
	pub embeddings_removed: usize,
}

/// Main orchestrator for the indexing pipeline.
/// Uses parallel text extraction with Rayon, followed by batched embedding.
/// Supports page-by-page PDF processing for reduced memory usage and resumability.
pub struct Indexer<E: SyncTextExtractor + PagedExtractor, M: Embedder, S: VectorStore> {
	options: IndexOptions,
	extractor: Arc<E>,
	embedder: M,
	store: Arc<S>,
	state: Option<Arc<StateManager>>,
	lexical: Option<Arc<LexicalIndex>>,
}

impl<E: SyncTextExtractor + PagedExtractor, M: Embedder, S: VectorStore> Indexer<E, M, S> {
	pub fn new(options: IndexOptions, extractor: E, embedder: M, store: Arc<S>) -> Self {
		Self { options, extractor: Arc::new(extractor), embedder, store, state: None, lexical: None }
	}
	
	/// Set the state manager for incremental indexing.
	pub fn with_state(mut self, state: Arc<StateManager>) -> Self {
		self.state = Some(state);
		self
	}
	
	/// Set the lexical index for full-text search.
	pub fn with_lexical(mut self, lexical: Arc<LexicalIndex>) -> Self {
		self.lexical = Some(lexical);
		self
	}

	/// Run the indexing pipeline (no progress reporting).
	pub async fn run(&mut self) -> Result<IndexResult> {
		self.run_with_progress(|_| ()).await
	}

	/// Run garbage collection to remove embeddings for deleted or modified files.
	/// This should be called before indexing to clean up stale data.
	pub async fn garbage_collect(&self) -> Result<GcResult> {
		let state = match &self.state {
			Some(s) => s,
			None => return Ok(GcResult::default()),
		};

		let mut result = GcResult::default();

		// 1. Clean up embeddings for deleted files
		let deleted_files = state.get_deleted_files()?;
		for path in &deleted_files {
			let doc_ids = state.remove_file(path)?;
			if !doc_ids.is_empty() {
				let removed = self.store.delete_by_doc_ids(&doc_ids).await?;
				result.embeddings_removed += removed;
				result.deleted_files += 1;
			}
		}

		// 2. Clean up old embeddings for modified files (they'll be re-indexed)
		let all_files = state.get_all_files()?;
		for file_info in all_files {
			if file_info.file_state == FileState::Modified && !file_info.doc_ids.is_empty() {
				let removed = self.store.delete_by_doc_ids(&file_info.doc_ids).await?;
				result.embeddings_removed += removed;
				result.modified_files += 1;
			}
		}

		Ok(result)
	}

	/// Run the indexing pipeline, reporting progress via callback.
	/// Uses parallel text extraction with Rayon for non-paged files.
	/// For paged files (PDFs), processes page-by-page with checkpoints.
	pub async fn run_with_progress<F>(&mut self, mut cb: F) -> Result<IndexResult>
	where
		F: FnMut(IndexEvent) + Send,
	{
		let files = discover_files(&self.options.root, &self.options.skip_extensions, &self.options.skip_files)?;
		let chunk_size = self.options.chunk_size;
		let max_file_size = self.options.max_file_size_bytes;
		let max_memory = self.options.max_memory_bytes;
		let max_chunks = self.options.max_chunks_per_file;

		// Counters for skipped/unchanged (used in parallel phase)
		let files_skipped = AtomicUsize::new(0);
		let files_unchanged = AtomicUsize::new(0);

		// Check memory before starting
		let mut sys = System::new();
		sys.refresh_memory();
		let used_mem = sys.used_memory();
		if used_mem > max_memory {
			let used_mb = used_mem / 1024 / 1024;
			let limit_mb = max_memory / 1024 / 1024;
			cb(IndexEvent::MemoryPressure(used_mb, limit_mb));
			// Continue anyway but warn - parallel extraction will proceed
		}

		// Separate paged files (PDFs) from non-paged files
		let (paged_files, non_paged_files): (Vec<_>, Vec<_>) = files
			.into_iter()
			.partition(|path| self.extractor.is_paged(path));

		// Phase 1: Parallel text extraction with Rayon for non-paged files
		let extractor = self.extractor.clone();
		let state = self.state.clone();
		
		let extraction_results: Vec<_> = non_paged_files
			.par_iter()
			.filter_map(|path| {
				// Check file size
				if let Ok(metadata) = std::fs::metadata(path) {
					if metadata.len() > max_file_size {
						files_skipped.fetch_add(1, Ordering::Relaxed);
						return None;
					}
				}
				
				// Check if file needs indexing
				if let Some(ref state) = state {
					match state.needs_indexing(path) {
						Ok(false) => {
							files_unchanged.fetch_add(1, Ordering::Relaxed);
							return None;
						}
						Ok(true) => {}
						Err(_) => {} // Index anyway on error
					}
				}
				
				// Extract text (sync, CPU-bound)
				match extractor.extract_text_sync(path) {
					Ok(contents) => {
						let chunks = chunk_text(&contents, chunk_size);
						
						// Skip files with too many chunks (e.g., dictionaries, wordlists)
						if chunks.len() > max_chunks {
							files_skipped.fetch_add(1, Ordering::Relaxed);
							return None;
						}
						
						let file_type = path.extension()
							.and_then(|e| e.to_str())
							.unwrap_or("unknown")
							.to_string();
						Some(Ok((path.clone(), chunks, file_type)))
					}
					Err(e) => Some(Err((path.clone(), format!("{}", e))))
				}
			})
			.collect();

		// Phase 2: Sequential embedding and batch storage for non-paged files
		let mut files_indexed = 0;
		let mut chunks_indexed = 0;
		let mut embeddings_stored = 0;
		let mut errors: Vec<(PathBuf, String)> = vec![];

		for result in extraction_results {
			match result {
				Ok((path, chunks, file_type)) => {
					cb(IndexEvent::FileStarted(path.clone()));
					
					if chunks.is_empty() {
						cb(IndexEvent::FileIndexed(path));
						continue;
					}

					let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
					
					match self.embedder.embed_batch(&chunk_refs).await {
						Ok(embeddings) => {
							chunks_indexed += chunks.len();
							
							// Prepare all metadata for batch insert
							let metadata_batch: Vec<DocumentMetadata> = chunks.iter()
								.enumerate()
								.map(|(i, chunk)| {
									let snippet = if chunk.chars().count() > 200 {
										let truncated: String = chunk.chars().take(200).collect();
										Some(format!("{}...", truncated))
									} else {
										Some(chunk.clone())
									};
									DocumentMetadata {
										doc_id: String::new(),
										file_path: path.clone(),
										file_type: file_type.clone(),
										chunk_index: i,
										snippet,
									}
								})
								.collect();

							// Batch insert all embeddings for this file at once
							match self.store.add_embeddings_batch(embeddings, metadata_batch).await {
								Ok(doc_ids) => {
									embeddings_stored += doc_ids.len();
									
									// Batch add to lexical index if configured
									if let Some(ref lexical) = self.lexical {
										let lexical_docs: Vec<LexicalDoc> = doc_ids.iter()
											.zip(chunks.iter())
											.enumerate()
											.map(|(i, (doc_id, chunk))| LexicalDoc {
												doc_id: doc_id.clone(),
												file_path: path.to_string_lossy().to_string(),
												content: chunk.clone(),
												chunk_index: i,
											})
											.collect();
										if let Err(e) = lexical.add_documents(lexical_docs) {
											cb(IndexEvent::FileError(path.clone(), format!("Lexical index error: {}", e)));
										}
									}
									
									// Report progress for each chunk
									for (i, doc_id) in doc_ids.iter().enumerate() {
										cb(IndexEvent::ChunkEmbedded(path.clone(), i, doc_id.clone()));
									}
									
									// Mark file as indexed in state manager
									if let Some(ref state) = self.state {
										if let Ok(meta) = std::fs::metadata(&path) {
											if let Ok(mtime) = meta.modified() {
												if let Err(e) = state.mark_indexed(&path, mtime, &doc_ids) {
													eprintln!("  warning: failed to update state for {}: {}", path.display(), e);
												}
											}
										}
									}
									files_indexed += 1;
								}
								Err(e) => {
									let err_str = format!("Failed to store embeddings: {}", e);
									cb(IndexEvent::FileError(path.clone(), err_str.clone()));
									errors.push((path.clone(), err_str));
								}
							}
						}
						Err(e) => {
							let err_str = format!("Embedding failed: {}", e);
							cb(IndexEvent::FileError(path.clone(), err_str.clone()));
							errors.push((path.clone(), err_str));
						}
					}
					
					cb(IndexEvent::FileIndexed(path));
				}
				Err((path, err_str)) => {
					cb(IndexEvent::FileError(path.clone(), err_str.clone()));
					errors.push((path, err_str));
				}
			}
		}

		// Phase 3: Page-by-page processing for paged files (PDFs)
		for path in paged_files {
			// Check file size
			if let Ok(metadata) = std::fs::metadata(&path) {
				if metadata.len() > max_file_size {
					files_skipped.fetch_add(1, Ordering::Relaxed);
					continue;
				}
			}
			
			// Get mtime for state tracking
			let mtime = match std::fs::metadata(&path).and_then(|m| m.modified()) {
				Ok(t) => t,
				Err(_) => {
					errors.push((path.clone(), "Failed to get file mtime".to_string()));
					continue;
				}
			};
			
			// Check if file needs indexing (for full file)
			if let Some(ref state) = self.state {
				match state.needs_indexing(&path) {
					Ok(false) => {
						files_unchanged.fetch_add(1, Ordering::Relaxed);
						continue;
					}
					Ok(true) => {}
					Err(_) => {} // Index anyway on error
				}
			}

			cb(IndexEvent::FileStarted(path.clone()));
			
			// Get resume page if interrupted previously
			let resume_page = self.state.as_ref()
				.and_then(|s| s.get_resume_page(&path, mtime).ok())
				.flatten()
				.unwrap_or(0);

			// Extract all pages
			let pages = match self.extractor.extract_pages(&path) {
				Ok(p) => p,
				Err(e) => {
					let err_str = format!("Failed to extract pages: {}", e);
					cb(IndexEvent::FileError(path.clone(), err_str.clone()));
					errors.push((path.clone(), err_str));
					continue;
				}
			};

			if pages.is_empty() {
				cb(IndexEvent::FileIndexed(path));
				continue;
			}

			let total_pages = pages.len();
			let file_type = path.extension()
				.and_then(|e| e.to_str())
				.unwrap_or("pdf")
				.to_string();

			// Process each page
			for page in pages.into_iter().skip(resume_page) {
				// Skip already indexed pages
				let page_num = page.page_num;
				
				if page.text.trim().is_empty() {
					cb(IndexEvent::PageProcessed(path.clone(), page_num, total_pages));
					continue;
				}

				// Chunk the page text
				let chunks = chunk_text(&page.text, chunk_size);
				let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
				
				match self.embedder.embed_batch(&chunk_refs).await {
					Ok(embeddings) => {
						chunks_indexed += chunks.len();
						
						// Prepare metadata for batch insert
						let metadata_batch: Vec<DocumentMetadata> = chunks.iter()
							.enumerate()
							.map(|(i, chunk)| {
								let global_chunk_idx = page_num * 1000 + i;
								let snippet = if chunk.chars().count() > 200 {
									let truncated: String = chunk.chars().take(200).collect();
									Some(format!("{}...", truncated))
								} else {
									Some(chunk.clone())
								};
								DocumentMetadata {
									doc_id: String::new(),
									file_path: path.clone(),
									file_type: file_type.clone(),
									chunk_index: global_chunk_idx,
									snippet,
								}
							})
							.collect();

						// Batch insert all page embeddings at once
						match self.store.add_embeddings_batch(embeddings, metadata_batch).await {
							Ok(doc_ids) => {
								embeddings_stored += doc_ids.len();
								
								// Batch add to lexical index if configured
								if let Some(ref lexical) = self.lexical {
									let lexical_docs: Vec<LexicalDoc> = doc_ids.iter()
										.zip(chunks.iter())
										.enumerate()
										.map(|(i, (doc_id, chunk))| {
											let global_chunk_idx = page_num * 1000 + i;
											LexicalDoc {
												doc_id: doc_id.clone(),
												file_path: path.to_string_lossy().to_string(),
												content: chunk.clone(),
												chunk_index: global_chunk_idx,
											}
										})
										.collect();
									if let Err(e) = lexical.add_documents(lexical_docs) {
										cb(IndexEvent::FileError(path.clone(), format!("Lexical index error: {}", e)));
									}
								}
								
								// Report progress
								for (i, doc_id) in doc_ids.iter().enumerate() {
									let global_chunk_idx = page_num * 1000 + i;
									cb(IndexEvent::ChunkEmbedded(path.clone(), global_chunk_idx, doc_id.clone()));
								}

								// Checkpoint: mark this page as indexed
								if let Some(ref state) = self.state {
									if let Err(e) = state.mark_page_indexed(&path, mtime, page_num, total_pages, &doc_ids) {
										eprintln!("  warning: failed to checkpoint page {} of {}: {}", 
											page_num, path.display(), e);
									}
								}
							}
							Err(e) => {
								let err_str = format!("Failed to store page {} embeddings: {}", page_num, e);
								cb(IndexEvent::FileError(path.clone(), err_str.clone()));
								errors.push((path.clone(), err_str));
							}
						}
					}
					Err(e) => {
						let err_str = format!("Embedding page {} failed: {}", page_num, e);
						cb(IndexEvent::FileError(path.clone(), err_str.clone()));
						errors.push((path.clone(), err_str));
						continue;
					}
				}

				cb(IndexEvent::PageProcessed(path.clone(), page_num, total_pages));
			}

			files_indexed += 1;
			cb(IndexEvent::FileIndexed(path));
		}

		// Persist the store
		self.store.save().await?;
		
		// Commit the lexical index if configured
		if let Some(ref lexical) = self.lexical {
			lexical.commit()?;
		}

		cb(IndexEvent::Done);
		Ok(IndexResult {
			files_indexed,
			files_skipped: files_skipped.load(Ordering::Relaxed),
			files_unchanged: files_unchanged.load(Ordering::Relaxed),
			chunks_indexed,
			embeddings_stored,
			errors,
		})
	}
}

/// Recursively discover supported files in a directory.
fn discover_files(root: &PathBuf, skip_extensions: &[String], skip_files: &[String]) -> Result<Vec<PathBuf>> {
	let mut files = Vec::new();
	let supported = ["txt", "md", "pdf", "png", "jpg", "jpeg"];
	for entry in walkdir::WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
		let path = entry.path();
		if path.is_file() {
			// Skip if filename matches any skip pattern
			if let Some(filename) = path.file_name().and_then(OsStr::to_str) {
				if skip_files.iter().any(|pattern| filename.contains(pattern)) {
					continue;
				}
			}
			if let Some(ext) = path.extension().and_then(OsStr::to_str) {
				let ext_lower = ext.to_lowercase();
				// Skip if extension in skip list
				if skip_extensions.iter().any(|s| s.to_lowercase() == ext_lower) {
					continue;
				}
				if supported.contains(&ext_lower.as_str()) {
					files.push(path.to_path_buf());
				}
			}
		}
	}
	Ok(files)
}

/// Split text into chunks of roughly `max_len` characters.
/// Uses a smarter strategy:
/// 1. First try to split by paragraphs (double newlines)
/// 2. For content with many short lines, group them more aggressively
/// 3. Never break mid-word if possible
fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
	// First, try paragraph-based chunking (split on double newlines)
	let paragraphs: Vec<&str> = text.split("\n\n").collect();
	
	// If we have reasonable paragraphs, use them
	if paragraphs.len() > 1 && paragraphs.len() < text.len() / 100 {
		return chunk_by_paragraphs(&paragraphs, max_len);
	}
	
	// Otherwise, use character-based chunking (better for short-line content)
	chunk_by_chars(text, max_len)
}

/// Chunk by paragraphs, merging small ones and splitting large ones.
fn chunk_by_paragraphs(paragraphs: &[&str], max_len: usize) -> Vec<String> {
	let mut chunks = Vec::new();
	let mut current = String::new();
	
	for para in paragraphs {
		let para = para.trim();
		if para.is_empty() {
			continue;
		}
		
		// If adding this paragraph would exceed limit
		if !current.is_empty() && current.len() + para.len() + 2 > max_len {
			chunks.push(current.clone());
			current.clear();
		}
		
		// If single paragraph is too long, split it
		if para.len() > max_len {
			if !current.is_empty() {
				chunks.push(current.clone());
				current.clear();
			}
			chunks.extend(chunk_by_chars(para, max_len));
			continue;
		}
		
		if !current.is_empty() {
			current.push_str("\n\n");
		}
		current.push_str(para);
	}
	
	if !current.is_empty() {
		chunks.push(current);
	}
	chunks
}

/// Character-based chunking that respects word boundaries.
/// Much better for short-line content (poetry, lyrics, code).
fn chunk_by_chars(text: &str, max_len: usize) -> Vec<String> {
	let mut chunks = Vec::new();
	let mut start = 0;
	let chars: Vec<char> = text.chars().collect();
	let len = chars.len();
	
	while start < len {
		let mut end = (start + max_len).min(len);
		
		// If we're not at the end, try to break at a word boundary
		if end < len {
			// Look back for a space or newline
			let mut break_pos = end;
			while break_pos > start && !chars[break_pos].is_whitespace() {
				break_pos -= 1;
			}
			// If we found a good break point (not all the way back to start)
			if break_pos > start + max_len / 2 {
				end = break_pos;
			}
		}
		
		let chunk: String = chars[start..end].iter().collect();
		let trimmed = chunk.trim();
		if !trimmed.is_empty() {
			chunks.push(trimmed.to_string());
		}
		start = end;
		
		// Skip leading whitespace for next chunk
		while start < len && chars[start].is_whitespace() {
			start += 1;
		}
	}
	
	chunks
}

/// Trait for extracting text from files (plain, PDF, OCR, etc.)
#[async_trait]
pub trait TextExtractor: Send + Sync {
	async fn extract_text(&self, path: &PathBuf) -> Result<String>;
}

/// Sync version of TextExtractor for parallel processing with Rayon.
pub trait SyncTextExtractor: Send + Sync {
	fn extract_text_sync(&self, path: &PathBuf) -> Result<String>;
}

/// Trait for generating embeddings from text.
#[async_trait]
pub trait Embedder: Send + Sync {
	async fn embed(&self, text: &str) -> Result<Vec<f32>>;
	/// Embed multiple texts in a batch for efficiency.
	async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
	/// Return the embedding dimension.
	fn dimension(&self) -> usize;
}

