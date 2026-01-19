//! OCR abstraction for Nexus Local.
//!
//! Provides a trait for extracting text from images and scanned documents using OCR.
//! Images are automatically resized before OCR to limit memory usage.
//! PDFs are processed page-by-page to reduce memory footprint.

use std::path::PathBuf;
use std::fs;
use async_trait::async_trait;
use anyhow::Result;

use leptess::LepTess;
use poppler::PopplerDocument;
use image::GenericImageView;
use tempfile::NamedTempFile;
use dotext::{MsDoc, Docx, Xlsx, Pptx, Odt, Odp};
use dotext::doc::OpenOfficeDoc;
use std::io::Read;

/// Maximum dimension (width or height) for images before OCR.
/// Larger images are downscaled to fit within this limit.
const MAX_IMAGE_DIMENSION: u32 = 2000;

/// Trait for OCR text extraction from images or PDFs.
#[async_trait]
pub trait OcrEngine: Send + Sync {
    async fn extract_text(&self, path: &PathBuf) -> Result<String>;
}

/// Sync trait for parallel text extraction with Rayon.
pub trait SyncOcrEngine: Send + Sync {
    fn extract_text_sync(&self, path: &PathBuf) -> Result<String>;
}

/// A single page extracted from a document.
#[derive(Debug, Clone)]
pub struct ExtractedPage {
    /// Page number (0-indexed)
    pub page_num: usize,
    /// Total pages in document
    pub total_pages: usize,
    /// Extracted text content
    pub text: String,
}

/// Trait for page-by-page extraction (for PDFs and multi-page documents).
pub trait PagedExtractor: Send + Sync {
    /// Extract pages one at a time. Returns iterator of pages.
    /// For non-paged documents (txt, images), returns single page with all content.
    fn extract_pages(&self, path: &PathBuf) -> Result<Vec<ExtractedPage>>;
    
    /// Check if this file type supports paged extraction.
    fn is_paged(&self, path: &PathBuf) -> bool;
}

/// Preprocesses an image: loads it, resizes if needed, saves to temp file.
/// Returns the path to use for OCR (either original or temp file).
fn preprocess_image(path: &PathBuf) -> Result<(PathBuf, Option<NamedTempFile>)> {
    let img = image::open(path)?;
    let (width, height) = img.dimensions();
    
    // Check if resizing is needed
    if width <= MAX_IMAGE_DIMENSION && height <= MAX_IMAGE_DIMENSION {
        // Image is small enough, use original
        return Ok((path.clone(), None));
    }
    
    // Calculate new dimensions preserving aspect ratio
    let scale = if width > height {
        MAX_IMAGE_DIMENSION as f64 / width as f64
    } else {
        MAX_IMAGE_DIMENSION as f64 / height as f64
    };
    
    let new_width = (width as f64 * scale) as u32;
    let new_height = (height as f64 * scale) as u32;
    
    eprintln!("  resizing: {}x{} -> {}x{}", width, height, new_width, new_height);
    
    // Resize using Lanczos3 for quality
    let resized = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);
    
    // Save to temp file (PNG for lossless)
    let temp_file = NamedTempFile::with_suffix(".png")?;
    resized.save(temp_file.path())?;
    
    Ok((temp_file.path().to_path_buf(), Some(temp_file)))
}

/// Text-based file extensions (code, config, docs)
const TEXT_EXTENSIONS: &[&str] = &[
    // Documents
    "txt", "md", "markdown", "rst", "org", "tex", "rtf",
    // Programming languages
    "py", "rs", "js", "ts", "jsx", "tsx", "cpp", "c", "h", "hpp", "cc", "cxx",
    "go", "java", "kt", "kts", "scala", "rb", "php", "swift", "m", "mm",
    "cs", "fs", "vb", "r", "lua", "pl", "pm", "tcl", "zig", "nim", "d",
    "hs", "ml", "mli", "ex", "exs", "erl", "hrl", "clj", "cljs", "lisp", "el",
    "v", "sv", "vhd", "vhdl", "asm", "s",
    // Shell/scripts
    "sh", "bash", "zsh", "fish", "ps1", "psm1", "bat", "cmd",
    // Config/data
    "json", "yaml", "yml", "toml", "xml", "ini", "cfg", "conf", "config",
    "env", "properties", "plist",
    // Web
    "html", "htm", "css", "scss", "sass", "less", "svg",
    // Database/query
    "sql", "graphql", "gql",
    // Build/CI
    "cmake", "make", "gradle", "sbt", "cabal",
    // Other
    "csv", "tsv", "log", "diff", "patch",
];

/// Known no-extension filenames that are text
const TEXT_FILENAMES: &[&str] = &[
    "Makefile", "makefile", "GNUmakefile",
    "Dockerfile", "dockerfile",
    "Containerfile",
    "Vagrantfile",
    "Gemfile", "Rakefile",
    "LICENSE", "LICENCE", "COPYING",
    "README", "CHANGELOG", "HISTORY", "AUTHORS", "CONTRIBUTORS",
    "TODO", "NOTES", "INSTALL", "NEWS",
    ".gitignore", ".gitattributes", ".gitmodules",
    ".dockerignore", ".editorconfig", ".env",
    ".bashrc", ".zshrc", ".profile", ".bash_profile",
    "requirements.txt", "Pipfile", "Cargo.toml", "go.mod", "package.json",
];

/// Check if a file is likely text by trying to read it as UTF-8
fn is_valid_utf8_file(path: &PathBuf, max_bytes: usize) -> bool {
    if let Ok(file) = fs::File::open(path) {
        use std::io::Read;
        let mut reader = std::io::BufReader::new(file);
        let mut buffer = vec![0u8; max_bytes.min(8192)];
        if let Ok(n) = reader.read(&mut buffer) {
            buffer.truncate(n);
            // Check for null bytes (binary indicator) and valid UTF-8
            return !buffer.contains(&0) && std::str::from_utf8(&buffer).is_ok();
        }
    }
    false
}

/// Implementation for extracting text from various file types.
pub struct PlainTextExtractor;

impl PlainTextExtractor {
    /// Check if file is a supported text file
    pub fn is_text_file(path: &PathBuf) -> bool {
        // Check extension
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_lowercase();
            if TEXT_EXTENSIONS.contains(&ext_lower.as_str()) {
                return true;
            }
        }
        
        // Check filename (for no-extension files)
        if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
            if TEXT_FILENAMES.contains(&filename) {
                return true;
            }
            // If no extension, check if it's valid UTF-8
            if path.extension().is_none() {
                return is_valid_utf8_file(path, 4096);
            }
        }
        
        false
    }
    
    /// Core sync extraction logic, used by both async and sync traits.
    fn do_extract(&self, path: &PathBuf) -> Result<String> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        
        // Check for text files first (including code, config, no-extension)
        if Self::is_text_file(path) && !matches!(ext.as_str(), "pdf" | "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tiff" | "tif" | "docx" | "xlsx" | "pptx" | "odt" | "odp") {
            let text = fs::read_to_string(path)?;
            return Ok(text);
        }
        
        match ext.as_str() {
            // Microsoft Office formats (dotext)
            "docx" => {
                let mut doc = Docx::open(path)?;
                let mut text = String::new();
                doc.read_to_string(&mut text)?;
                Ok(text)
            }
            "xlsx" => {
                let mut doc = Xlsx::open(path)?;
                let mut text = String::new();
                doc.read_to_string(&mut text)?;
                Ok(text)
            }
            "pptx" => {
                let mut doc = Pptx::open(path)?;
                let mut text = String::new();
                doc.read_to_string(&mut text)?;
                Ok(text)
            }
            // OpenDocument formats (dotext)
            "odt" => {
                let mut doc = Odt::open(path)?;
                let mut text = String::new();
                doc.read_to_string(&mut text)?;
                Ok(text)
            }
            "odp" => {
                let mut doc = Odp::open(path)?;
                let mut text = String::new();
                doc.read_to_string(&mut text)?;
                Ok(text)
            }
            // HTML extraction
            "html" | "htm" => {
                let html_content = fs::read_to_string(path)?;
                let text = html2text::from_read(html_content.as_bytes(), 100)?;
                Ok(text)
            }
            // Images
            "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tiff" | "tif" => {
                // Preprocess image (resize if needed)
                let (ocr_path, _temp_file) = preprocess_image(path)?;
                
                let mut lt = LepTess::new(None, "eng")?;
                lt.set_image(&ocr_path)?;
                let text = lt.get_utf8_text()?;
                
                // _temp_file is dropped here, cleaning up the temp file
                Ok(text)
            }
            "pdf" => {
                let mut data = fs::read(path)?;
                let doc = PopplerDocument::new_from_data(&mut data, None)
                    .map_err(|e| anyhow::anyhow!("Failed to open PDF: {:?}", e))?;
                
                let mut text = String::new();
                for page in doc.pages() {
                    if let Some(page_text) = page.get_text() {
                        text.push_str(page_text);
                        text.push_str("\n");
                    }
                }
                Ok(text)
            }
            _ => Ok(String::new()),
        }
    }
}

#[async_trait]
impl OcrEngine for PlainTextExtractor {
    async fn extract_text(&self, path: &PathBuf) -> Result<String> {
        self.do_extract(path)
    }
}

impl SyncOcrEngine for PlainTextExtractor {
    fn extract_text_sync(&self, path: &PathBuf) -> Result<String> {
        self.do_extract(path)
    }
}

impl PagedExtractor for PlainTextExtractor {
    fn extract_pages(&self, path: &PathBuf) -> Result<Vec<ExtractedPage>> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        
        match ext.as_str() {
            "pdf" => {
                // Memory-mapped file reading would be ideal here, but poppler needs the data
                // For now, we still read the file but process pages individually
                let mut data = fs::read(path)?;
                let doc = PopplerDocument::new_from_data(&mut data, None)
                    .map_err(|e| anyhow::anyhow!("Failed to open PDF: {:?}", e))?;
                
                let pages: Vec<_> = doc.pages().collect();
                let total_pages = pages.len();
                
                let mut result = Vec::with_capacity(total_pages);
                for (page_num, page) in pages.into_iter().enumerate() {
                    let text = page.get_text().unwrap_or_default().to_string();
                    result.push(ExtractedPage {
                        page_num,
                        total_pages,
                        text,
                    });
                }
                Ok(result)
            }
            _ => {
                // Non-paged documents: return single page with all content
                let text = self.do_extract(path)?;
                Ok(vec![ExtractedPage {
                    page_num: 0,
                    total_pages: 1,
                    text,
                }])
            }
        }
    }
    
    fn is_paged(&self, path: &PathBuf) -> bool {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        ext == "pdf"
    }
}

/// Stub for future PDF/image OCR implementation
pub struct StubOcr;

#[async_trait]
impl OcrEngine for StubOcr {
    async fn extract_text(&self, _path: &PathBuf) -> Result<String> {
        Ok("[OCR not implemented]".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_preprocess_small_image_no_resize() {
        // A small image should not be resized
        // This test would need a real small image file
    }

    #[tokio::test]
    async fn test_plain_text_extraction() {
        let extractor = PlainTextExtractor;
        let path = PathBuf::from("src/lib.rs");
        let result = extractor.extract_text(&path).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("OcrEngine"));
    }
}
