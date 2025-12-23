use clap::{Parser, Subcommand};
use std::fs;
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use vector_db_rs::dsl::{execute_line, DslOutput};
use vector_db_rs::engine::TensorDb;
use toon_format::encode_default;
use vector_db_rs::server::start_server;

#[derive(Parser)]
#[command(name = "VectorDB")]
#[command(version = "0.1")]
#[command(about = "A vector database", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start REPL (default)
    Repl {
        /// Output format: 'display' (default, human-readable) or 'toon' (machine-readable)
        #[arg(long, default_value = "display")]
        format: String,
    },
    /// Run a script file
    Run {
        /// Path to the script file (.vdb)
        file: String,
        /// Output format: 'display' (default, human-readable) or 'toon' (machine-readable)
        #[arg(long, default_value = "display")]
        format: String,
    },
    /// Start HTTP server
    Server {
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mut db = TensorDb::new();

    match cli.command {
        Some(Commands::Run { file, format }) => {
            let content = fs::read_to_string(&file)?;
            let use_toon = format == "toon";
            
            // Execute script line by line with appropriate output format
            for (line_num, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                
                match execute_line(&mut db, line, line_num + 1) {
                    Ok(output) => {
                        if !matches!(output, DslOutput::None) {
                            if use_toon {
                                let toon = encode_default(&output)
                                    .unwrap_or_else(|e| format!("Error encoding TOON: {}", e));
                                println!("{}", toon);
                            } else {
                                println!("{}", output);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error on line {}: {}", line_num + 1, e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::Server { port }) => {
            // Need Arc<Mutex<TensorDb>>
            let db_arc = Arc::new(Mutex::new(db));
            start_server(db_arc, port).await;
        }
        Some(Commands::Repl { format }) => {
            let use_toon = format == "toon";
            
            println!("VectorDB REPL v0.1");
            if use_toon {
                println!("Output format: TOON (machine-readable)");
            } else {
                println!("Output format: Display (human-readable)");
            }
            println!("Type 'EXIT' to quit.");
            
            let stdin = io::stdin();
            let mut handle = stdin.lock();
            let mut buffer = String::new();

            loop {
                print!(">_>  ");
                io::stdout().flush()?;
                buffer.clear();
                if handle.read_line(&mut buffer)? == 0 {
                    break;
                }
                let line = buffer.trim();
                if line.eq_ignore_ascii_case("EXIT") {
                    break;
                }
                match execute_line(&mut db, line, 1) {
                    Ok(output) => {
                        if !matches!(output, DslOutput::None) {
                            if use_toon {
                                let toon = encode_default(&output)
                                    .unwrap_or_else(|e| format!("Error encoding TOON: {}", e));
                                println!("{}", toon);
                            } else {
                                println!("{}", output);
                            }
                        }
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
        }
        None => {
            // Default REPL with display format
            println!("VectorDB REPL v0.1");
            println!("Output format: Display (human-readable)");
            println!("Type 'EXIT' to quit.");
            
            let stdin = io::stdin();
            let mut handle = stdin.lock();
            let mut buffer = String::new();

            loop {
                print!(">_>  ");
                io::stdout().flush()?;
                buffer.clear();
                if handle.read_line(&mut buffer)? == 0 {
                    break;
                }
                let line = buffer.trim();
                if line.eq_ignore_ascii_case("EXIT") {
                    break;
                }
                match execute_line(&mut db, line, 1) {
                    Ok(output) => {
                        if !matches!(output, DslOutput::None) {
                            println!("{}", output);
                        }
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
        }
    }

    Ok(())
}
