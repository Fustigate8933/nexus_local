use std::path::PathBuf;
use std::fs;

#[tokio::test]
async fn test_index_txt_file() {
    // Create a temp directory with a .txt file
    let tmp_dir = std::env::temp_dir().join("nexus_ocr_test_txt");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).unwrap();
    let txt_path = tmp_dir.join("hello.txt");
    fs::write(&txt_path, "Hello, Nexus!").unwrap();

    // Use OcrExtractor to extract text
    use ocr::{PlainTextExtractor, OcrEngine};
    let extractor = PlainTextExtractor;
    let text = extractor.extract_text(&txt_path).await.unwrap();
    assert!(text.contains("Hello, Nexus!"), "Should extract text from .txt file");

    fs::remove_dir_all(&tmp_dir).unwrap();
}

#[tokio::test]
async fn test_index_md_file() {
    let tmp_dir = std::env::temp_dir().join("nexus_ocr_test_md");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).unwrap();
    let md_path = tmp_dir.join("readme.md");
    fs::write(&md_path, "# Title\nSome **markdown** content.").unwrap();

    use ocr::{PlainTextExtractor, OcrEngine};
    let extractor = PlainTextExtractor;
    let text = extractor.extract_text(&md_path).await.unwrap();
    assert!(text.contains("# Title"), "Should extract text from .md file");

    fs::remove_dir_all(&tmp_dir).unwrap();
}

#[tokio::test]
async fn test_index_pdf_file() {
    // Use the existing brainfuck.pdf test file
    let pdf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ocr/tests/brainfuck.pdf");
    if !pdf_path.exists() {
        eprintln!("brainfuck.pdf not found, skipping PDF integration test");
        return;
    }

    use ocr::{PlainTextExtractor, OcrEngine};
    let extractor = PlainTextExtractor;
    let text = extractor.extract_text(&pdf_path).await.unwrap();
    assert!(text.to_lowercase().contains("brainfuck"), "Should extract text from PDF file");
}

#[tokio::test]
async fn test_index_image_file() {
    // Use the existing ocr_test.png test file
    let img_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ocr/tests/ocr_test.png");
    if !img_path.exists() {
        eprintln!("ocr_test.png not found, skipping image OCR integration test");
        return;
    }

    use ocr::{PlainTextExtractor, OcrEngine};
    let extractor = PlainTextExtractor;
    let text = extractor.extract_text(&img_path).await.unwrap();
    assert!(text.to_uppercase().contains("HOW TO COMBINE"), "Should extract text from image file");
}
