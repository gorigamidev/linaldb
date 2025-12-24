use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: PathBuf,
    pub default_db: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                data_dir: PathBuf::from("./data"),
                default_db: "default".to_string(),
            },
        }
    }
}

impl EngineConfig {
    pub fn load() -> Self {
        let config_path = "linal.toml";
        if let Ok(content) = fs::read_to_string(config_path) {
            match toml::from_str(&content) {
                Ok(config) => return config,
                Err(e) => eprintln!(
                    "Warning: Failed to parse linal.toml: {}. Using defaults.",
                    e
                ),
            }
        }
        Self::default()
    }
}
