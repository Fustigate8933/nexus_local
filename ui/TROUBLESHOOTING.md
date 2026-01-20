# Troubleshooting Tauri Development

## Common Issues

### Missing WebKit Dependencies (Linux)

**Error:**
```
The system library `javascriptcoregtk-4.1` required by crate `javascriptcore-rs-sys` was not found.
```

**Solution for Arch Linux:**

Install the required development packages:

```bash
sudo pacman -S webkit2gtk-4.1 base-devel
```

Or run the provided script:

```bash
./install-deps.sh
```

**Solution for Ubuntu/Debian:**

```bash
sudo apt install libwebkit2gtk-4.1-dev \
  build-essential \
  curl \
  wget \
  libssl-dev \
  libgtk-3-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev
```

**Verify installation:**

```bash
pkg-config --exists webkit2gtk-4.1 && echo "Found" || echo "Not found"
pkg-config --exists javascriptcoregtk-4.1 && echo "Found" || echo "Not found"
```

### Rust Compilation Errors

If you see Rust compilation errors:

1. **Clean and rebuild:**
```bash
cd ui
cargo clean
pnpm tauri dev
```

2. **Check Rust toolchain:**
```bash
rustup update
rustup component add rust-src
```

### Frontend Build Issues

1. **Clean node_modules:**
```bash
cd ui
rm -rf node_modules
pnpm install
```

2. **Check Vite is running:**
```bash
pnpm dev
# Should start on http://localhost:1420
```

### Port Already in Use

If port 1420 is already in use:

1. Change the port in `vite.config.js`:
```javascript
server: {
  port: 1421,  // Change this
  // ...
}
```

2. Update `tauri.conf.json`:
```json
{
  "build": {
    "devUrl": "http://localhost:1421"
  }
}
```

### Permission Issues

If you see permission errors:

```bash
# Make scripts executable
chmod +x install-deps.sh

# Fix cargo permissions
cargo install --locked cargo-binstall
```

## Getting More Debug Information

### Verbose Tauri Output

```bash
RUST_BACKTRACE=1 pnpm tauri dev
```

### Check Cargo Build

```bash
cd ui/src-tauri
cargo build --verbose
```

### Check Frontend Build

```bash
cd ui
pnpm build
```

## Still Having Issues?

1. Check Tauri prerequisites: https://tauri.app/start/prerequisites/
2. Check Tauri issues: https://github.com/tauri-apps/tauri/issues
3. Verify all dependencies are installed correctly
