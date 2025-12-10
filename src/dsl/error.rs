use crate::engine::EngineError;

/// Errores del lenguaje de alto nivel (DSL)
#[derive(Debug)]
pub enum DslError {
    Parse { line: usize, msg: String },
    Engine { line: usize, source: EngineError },
}

impl std::fmt::Display for DslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DslError::Parse { line, msg } => {
                write!(f, "[line {}] Parse error: {}", line, msg)
            }
            DslError::Engine { line, source } => {
                write!(f, "[line {}] Engine error: {}", line, source)
            }
        }
    }
}

impl std::error::Error for DslError {}
