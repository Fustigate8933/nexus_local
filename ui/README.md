# Nexus Local - Desktop UI

Desktop UI for Nexus Local built with Tauri, Vue 3, and Tailwind CSS.

## Features

- 🔍 **Search**: Semantic, lexical, and hybrid search with real-time results
- 📁 **Indexing**: Index directories with progress tracking
- 📊 **Status Dashboard**: View index statistics (embeddings, documents, etc.)

## Development

### Prerequisites

- Rust (for building the backend)
- Node.js and pnpm
- System dependencies for Tauri (see [Tauri prerequisites](https://tauri.app/start/prerequisites/))

### Run in Development Mode

```bash
cd ui
pnpm install
pnpm tauri dev
```

### Build for Production

```bash
cd ui
pnpm install
pnpm tauri build
```

This creates platform-specific installers in `src-tauri/target/release/bundle/`:
- **Windows**: `.exe` or `.msi`
- **macOS**: `.dmg` or `.app`
- **Linux**: `.deb`, `.rpm`, or `.AppImage`

## Project Structure

```
ui/
├── src/                  # Vue frontend
│   ├── App.vue          # Main component
│   ├── main.js          # Entry point
│   └── style.css        # Tailwind CSS imports
├── src-tauri/           # Rust backend (Tauri)
│   ├── src/
│   │   ├── main.rs      # Entry point
│   │   └── lib.rs       # Tauri commands
│   └── Cargo.toml       # Rust dependencies
├── package.json         # Frontend dependencies
├── vite.config.js       # Vite configuration
├── tailwind.config.js   # Tailwind CSS configuration
└── postcss.config.js    # PostCSS configuration
```

## Tauri Commands

The Rust backend exposes these commands to the frontend:

- `search(query, mode, limit)` - Search the index
- `get_status()` - Get index status and statistics
- `index_directory(path, gpu, max_file_mb, max_memory_mb)` - Index a directory

## Architecture

The UI uses:
- **Tauri 2.0** for the desktop framework
- **Vue 3** with Composition API for the frontend
- **Tailwind CSS** for styling
- **Vite** for build tooling

The Rust backend imports and uses all the existing Nexus Local crates:
- `nexus_core` - Indexing pipeline and file discovery
- `embed` - Embedding generation
- `store` - Vector and lexical storage
- `ocr` - Text extraction

## User Installation (No Rust Required!)

Once built, users can install the app without needing Rust, Node.js, or any development tools. They simply:

1. Download the installer for their platform
2. Install like any normal desktop app
3. Use the app!

The Tauri build process bundles everything needed into the installer.
