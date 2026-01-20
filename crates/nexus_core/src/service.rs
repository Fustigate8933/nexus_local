//! System service installation for watch mode.
//!
//! Generates platform-specific service files:
//! - Linux: systemd user service
//! - macOS: launchd plist
//! - Windows: Startup folder shortcut (via PowerShell)

use std::path::PathBuf;
use std::fs;
use std::env;
use anyhow::Result;

/// Service manager for the current platform.
pub struct ServiceManager {
    binary_path: PathBuf,
}

impl ServiceManager {
    /// Create a new service manager.
    pub fn new() -> Result<Self> {
        let binary_path = env::current_exe()?;
        Ok(Self { binary_path })
    }

    /// Create with a specific binary path.
    pub fn with_binary(binary_path: PathBuf) -> Self {
        Self { binary_path }
    }

    /// Install the service for the current platform.
    pub fn install(&self) -> Result<String> {
        #[cfg(target_os = "linux")]
        return self.install_linux();

        #[cfg(target_os = "macos")]
        return self.install_macos();

        #[cfg(target_os = "windows")]
        return self.install_windows();

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        anyhow::bail!("Service installation not supported on this platform");
    }

    /// Uninstall the service for the current platform.
    pub fn uninstall(&self) -> Result<String> {
        #[cfg(target_os = "linux")]
        return self.uninstall_linux();

        #[cfg(target_os = "macos")]
        return self.uninstall_macos();

        #[cfg(target_os = "windows")]
        return self.uninstall_windows();

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        anyhow::bail!("Service uninstallation not supported on this platform");
    }

    /// Get the service status.
    pub fn status(&self) -> Result<String> {
        #[cfg(target_os = "linux")]
        return self.status_linux();

        #[cfg(target_os = "macos")]
        return self.status_macos();

        #[cfg(target_os = "windows")]
        return self.status_windows();

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        anyhow::bail!("Service status not supported on this platform");
    }

    // ========== Linux (systemd) ==========

    #[cfg(target_os = "linux")]
    fn systemd_service_path(&self) -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("~/.config"))
            .join("systemd/user/nexus.service")
    }

    #[cfg(target_os = "linux")]
    fn generate_systemd_service(&self) -> String {
        format!(
            r#"[Unit]
Description=Nexus Local - File Watcher
Documentation=https://github.com/yourusername/nexus-local
After=default.target

[Service]
Type=simple
ExecStart={} watch
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
"#,
            self.binary_path.display()
        )
    }

    #[cfg(target_os = "linux")]
    fn install_linux(&self) -> Result<String> {
        let service_path = self.systemd_service_path();
        let service_content = self.generate_systemd_service();

        // Create directory if needed
        if let Some(parent) = service_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&service_path, service_content)?;

        Ok(format!(
            r#"Installed systemd service at: {}

To enable and start:
  systemctl --user daemon-reload
  systemctl --user enable nexus
  systemctl --user start nexus

To check status:
  systemctl --user status nexus

To view logs:
  journalctl --user -u nexus -f"#,
            service_path.display()
        ))
    }

    #[cfg(target_os = "linux")]
    fn uninstall_linux(&self) -> Result<String> {
        let service_path = self.systemd_service_path();

        // Stop and disable first
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "stop", "nexus"])
            .output();
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "disable", "nexus"])
            .output();

        if service_path.exists() {
            fs::remove_file(&service_path)?;
        }

        let _ = std::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .output();

        Ok(format!(
            "Uninstalled systemd service from: {}",
            service_path.display()
        ))
    }

    #[cfg(target_os = "linux")]
    fn status_linux(&self) -> Result<String> {
        let output = std::process::Command::new("systemctl")
            .args(["--user", "status", "nexus"])
            .output()?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    // ========== macOS (launchd) ==========

    #[cfg(target_os = "macos")]
    fn launchd_plist_path(&self) -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("~"))
            .join("Library/LaunchAgents/com.nexus.watch.plist")
    }

    #[cfg(target_os = "macos")]
    fn generate_launchd_plist(&self) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nexus.watch</string>
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
        <string>watch</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/nexus.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/nexus.err</string>
</dict>
</plist>
"#,
            self.binary_path.display()
        )
    }

    #[cfg(target_os = "macos")]
    fn install_macos(&self) -> Result<String> {
        let plist_path = self.launchd_plist_path();
        let plist_content = self.generate_launchd_plist();

        // Create directory if needed
        if let Some(parent) = plist_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&plist_path, plist_content)?;

        Ok(format!(
            r#"Installed launchd service at: {}

To load and start:
  launchctl load {}

To check status:
  launchctl list | grep nexus

To view logs:
  tail -f /tmp/nexus.log"#,
            plist_path.display(),
            plist_path.display()
        ))
    }

    #[cfg(target_os = "macos")]
    fn uninstall_macos(&self) -> Result<String> {
        let plist_path = self.launchd_plist_path();

        // Unload first
        let _ = std::process::Command::new("launchctl")
            .args(["unload", &plist_path.to_string_lossy()])
            .output();

        if plist_path.exists() {
            fs::remove_file(&plist_path)?;
        }

        Ok(format!(
            "Uninstalled launchd service from: {}",
            plist_path.display()
        ))
    }

    #[cfg(target_os = "macos")]
    fn status_macos(&self) -> Result<String> {
        let output = std::process::Command::new("launchctl")
            .args(["list"])
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let nexus_line = stdout
            .lines()
            .find(|l| l.contains("nexus"))
            .unwrap_or("Nexus service not loaded");

        Ok(nexus_line.to_string())
    }

    // ========== Windows (Startup folder) ==========

    #[cfg(target_os = "windows")]
    fn startup_shortcut_path(&self) -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("~"))
            .join(r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\nexus-watch.bat")
    }

    #[cfg(target_os = "windows")]
    fn generate_startup_batch(&self) -> String {
        format!(
            r#"@echo off
start /min "" "{}" watch
"#,
            self.binary_path.display()
        )
    }

    #[cfg(target_os = "windows")]
    fn install_windows(&self) -> Result<String> {
        let shortcut_path = self.startup_shortcut_path();
        let batch_content = self.generate_startup_batch();

        // Create directory if needed
        if let Some(parent) = shortcut_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&shortcut_path, batch_content)?;

        Ok(format!(
            r#"Installed startup script at: {}

The watcher will start automatically on next login.
To start now, run:
  nexus watch"#,
            shortcut_path.display()
        ))
    }

    #[cfg(target_os = "windows")]
    fn uninstall_windows(&self) -> Result<String> {
        let shortcut_path = self.startup_shortcut_path();

        if shortcut_path.exists() {
            fs::remove_file(&shortcut_path)?;
        }

        Ok(format!(
            "Uninstalled startup script from: {}",
            shortcut_path.display()
        ))
    }

    #[cfg(target_os = "windows")]
    fn status_windows(&self) -> Result<String> {
        let shortcut_path = self.startup_shortcut_path();

        if shortcut_path.exists() {
            Ok(format!("Nexus startup script installed at: {}", shortcut_path.display()))
        } else {
            Ok("Nexus startup script not installed".to_string())
        }
    }
}
