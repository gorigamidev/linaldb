use clap::{Parser, Subcommand};
use std::fs;
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use vector_db_rs::dsl::{execute_line, execute_script, DslOutput};
use vector_db_rs::engine::TensorDb;
use vector_db_rs::server::start_server;

#[derive(Parser)]
#[command(name = "VectorDB")]
#[command(version = "0.1")]
#[command(about = "A toy vector database", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start REPL (default)
    Repl,
    /// Run a script file
    Run {
        /// Path to the script file (.vdb)
        file: String,
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
        Some(Commands::Run { file }) => {
            let content = fs::read_to_string(&file)?;
            // execute_script runs logic and prints output if any (via mod.rs logic we added)
            // Wait, execute_script returns Result<(), DslError> in current mod.rs implementation (Step 1172 updated it?)
            // Step 1172 updated `execute_line` to Result<DslOutput>.
            // And `execute_script` calls `execute_line` loop.
            // In Step 1172, `execute_script` prints output if not None.
            // So we just call it.
            if let Err(e) = execute_script(&mut db, &content) {
                eprintln!("Error executing script: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Server { port }) => {
            // Need Arc<Mutex<TensorDb>>
            let db_arc = Arc::new(Mutex::new(db));
            start_server(db_arc, port).await;
        }
        Some(Commands::Repl) | None => {
            println!("VectorDB REPL v0.1");
            println!("Type 'EXIT' to quit.");
            let stdin = io::stdin();
            let mut handle = stdin.lock();
            let mut buffer = String::new();

            loop {
                print!("vdb>>> ");
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
