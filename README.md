# Nexus Local

**Local-first, privacy-preserving semantic search for your documents.**

Nexus Local indexes your files (PDFs, text, images) and lets you search them using natural language queries. All processing happens locally; your data never leaves your machine.

## Features

- **Hybrid Search** — Combines semantic (vector) and lexical (keyword) search using Reciprocal Rank Fusion
- **GPU Acceleration** — Optional CUDA support for 5-10x faster embedding generation
- **Incremental Indexing** — Only processes new or modified files
- **PDF Page-by-Page** — Memory-efficient processing with resumable checkpoints
- **OCR Support** — Extract text from images and scanned PDFs (via Tesseract)
- **Privacy First** — 100% local, no cloud dependencies, no telemetry

## Installation

### Prerequisites

- **Rust** (nightly recommended): `rustup default nightly`
- **Tesseract OCR** (optional, for images): `sudo apt install tesseract-ocr`
- **CUDA Toolkit** (optional, for GPU): CUDA 11.8+ with cuDNN

### Build

```bash
git clone https://github.com/yourusername/nexus-local.git
cd nexus-local/nexus_local
cargo build --release
```

The binary will be at `./target/release/cli`.

## Usage

### Index Documents

```bash
# Basic indexing
./target/release/cli index ~/Documents

# With GPU acceleration
./target/release/cli index ~/Documents --gpu

# Skip images (faster, no OCR)
./target/release/cli index ~/Documents --skip-images

# Custom memory limit (MB)
./target/release/cli index ~/Documents --max-memory-mb 8000
```

### Search

```bash
# Hybrid search (default) — combines semantic + keyword
./target/release/cli search "machine learning optimization"

# Semantic only — meaning-based similarity
./target/release/cli search "ML optimization" --mode semantic

# Lexical only — exact keyword matching (BM25)
./target/release/cli search "gradient descent" --mode lexical

# More results
./target/release/cli search "neural networks" -n 10

# JSON output (for scripting)
./target/release/cli search "transformers" --json
```

### Check Status

```bash
./target/release/cli status
```

Output:
```
nexus status
  store: /home/user/.local/share/nexus_local
  vector embeddings: 1234
  lexical documents: 1234
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI                                 │
├─────────────────────────────────────────────────────────────┤
│                      nexus_core                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Discover │→ │ Extract  │→ │  Chunk   │→ │   Embed     │  │
│  │  Files   │  │  Text    │  │  Text    │  │  (GPU/CPU)  │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   LanceDB       │  │    Tantivy      │  │   SQLite    │  │
│  │ (Vector Store)  │  │ (Lexical Index) │  │   (State)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Crates

| Crate | Description |
|-------|-------------|
| `cli` | Command-line interface |
| `nexus_core` | Indexing pipeline, file discovery, chunking |
| `embed` | Embedding model (fastembed + ONNX Runtime) |
| `store` | LanceDB vector store, Tantivy lexical index, SQLite state |
| `ocr` | Text extraction (PDF, images via Tesseract) |
| `search` | (Reserved for future search abstractions) |

### Storage

All data is stored in `~/.local/share/nexus_local/`:

```
~/.local/share/nexus_local/
├── *.lance/           # LanceDB vector embeddings
├── tantivy_index/     # Tantivy inverted index (BM25)
└── state.db           # SQLite (file mtimes, doc_ids)
```

## How Search Works

### Hybrid Search (default)

1. **Semantic**: Query is embedded, ANN search finds similar vectors
2. **Lexical**: Query is tokenized, BM25 scores matching documents
3. **Fusion**: Results are combined using Reciprocal Rank Fusion (RRF):

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Where $k = 60$ (constant) and $R$ = {semantic, lexical}

## Configuration

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gpu` | Enable CUDA acceleration | Off |
| `--skip-images` | Skip PNG/JPG files (no OCR) | Off |
| `--skip-ext <ext>` | Skip specific extensions | None |
| `--max-memory-mb <MB>` | Memory limit for throttling | 75% of RAM |
| `--max-file-mb <MB>` | Skip files larger than this | 50 |
| `--mode <mode>` | Search mode: semantic, lexical, hybrid | hybrid |
| `-n <count>` | Number of search results | 5 |
| `--max-chunks <N>` | Skip files with >N chunks (edge cases) | 500 |

### Supported File Types

| Category | Extensions |
|----------|------------|
| **Documents** | txt, md, markdown, rst, org, tex, rtf |
| **Office** | docx, xlsx, pptx (Microsoft), odt, odp (OpenDocument) |
| **Code** | py, rs, js, ts, jsx, tsx, cpp, c, h, hpp, go, java, kt, scala, rb, php, swift, cs, fs, r, lua, pl, hs, ml, ex, erl, clj, lisp, zig, nim, d, v, vhd, asm... |
| **Shell** | sh, bash, zsh, fish, ps1, bat, cmd |
| **Config** | json, yaml, yml, toml, xml, ini, cfg, conf, env, properties, plist |
| **Web** | html, htm, css, scss, sass, less, svg |
| **Data** | sql, graphql, csv, tsv, log, diff, patch |
| **PDF** | pdf (text layer via poppler) |
| **Images** | png, jpg, jpeg, webp, bmp, tiff (OCR via Tesseract) |
| **No extension** | Makefile, Dockerfile, LICENSE, README, .gitignore, etc. |

## Development

```bash
# Run tests
cargo test

# Check for errors
cargo check

# Build debug version
cargo build

# Run directly
cargo run -p cli -- index ~/Documents
cargo run -p cli -- search "your query"
```

## License

MIT
