// src/main.rs

use std::env;
use std::fs;
use std::io::{self, Write};

use vector_db_rs::{TensorDb, execute_script, execute_line};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = TensorDb::new();
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Modo "ejecutar archivo de script"
        let path = &args[1];
        let script = fs::read_to_string(path)?;
        println!("Running script from '{}'\n", path);

        if let Err(e) = execute_script(&mut db, &script) {
            eprintln!("{}", e);
            std::process::exit(1);
        }
        return Ok(());
    }

    // Modo interactivo (REPL)
    println!("TensorDB CLI");
    println!("Type commands like:");
    println!("  DEFINE a AS TENSOR [3] VALUES [1, 0, 0]");
    println!("  LET c = ADD a b");
    println!("  LET s = CORRELATE a WITH b");
    println!("  LET sim = SIMILARITY a WITH b");
    println!("  LET half = SCALE a BY 0.5");
    println!("  SHOW a");
    println!("  SHOW ALL");
    println!("Type EXIT or QUIT to leave.\n");

    let mut input = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;

        input.clear();
        if io::stdin().read_line(&mut input)? == 0 {
            // EOF
            break;
        }

        let line = input.trim();
        if line.eq_ignore_ascii_case("EXIT") || line.eq_ignore_ascii_case("QUIT") {
            break;
        }
        if line.is_empty() {
            continue;
        }

        if let Err(e) = execute_line(&mut db, line, 1) {
            // En modo interactivo usamos siempre line_no = 1 (no importa tanto)
            eprintln!("{}", e);
        }
    }

    Ok(())
}
