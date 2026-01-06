use crate::dsl::{DslError, DslOutput};
use crate::engine::context::ExecutionContext;
use crate::engine::{BinaryOp, TensorDb, UnaryOp};

/// LET c = ADD a b
/// LET score = CORRELATE a WITH b
/// LET sim = SIMILARITY a WITH b
/// LET half = SCALE a BY 0.5
pub fn handle_let(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
    ctx: Option<&mut ExecutionContext>,
) -> Result<DslOutput, DslError> {
    // If no context provided, create a transient one
    let mut local_ctx;
    let ctx = match ctx {
        Some(c) => c,
        None => {
            local_ctx = ExecutionContext::new();
            &mut local_ctx
        }
    };

    // Quitamos LET
    let rest = line.trim_start_matches("LET").trim();
    // ... (rest of function body until return) ...
    // Note: I can't replace the signature and return in one go easily if body is long.
    // I will replace valid chunks.
    // Start with signature.

    // output = ...
    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: LET <name> = ...".into(),
        });
    }

    let output_name = parts[0].trim();
    if output_name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Missing output name in LET".into(),
        });
    }

    let expr = parts[1].trim();
    let tokens: Vec<&str> = expr.split_whitespace().collect();

    if tokens.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Missing expression in LET".into(),
        });
    }

    // Check if the command starts with a known operation keyword
    // This prevents valid commands like "RESHAPE ... [ ... ]" from being misidentified as indexing
    let is_keyword = matches!(
        tokens[0],
        "ADD"
            | "SUBTRACT"
            | "MULTIPLY"
            | "DIVIDE"
            | "CORRELATE"
            | "SIMILARITY"
            | "DISTANCE"
            | "MATMUL"
            | "TRANSPOSE"
            | "RESHAPE"
            | "STACK"
            | "SCALE"
            | "NORMALIZE"
            | "FLATTEN"
    );

    if !is_keyword {
        // Check for infix operator: a + b
        if let Some((left, op, right)) = parse_infix_op(expr) {
            return handle_infix_op(db, output_name, left, op, right, line_no, ctx);
        }

        // Check for dot notation: name.field
        if expr.contains('.') && !expr.contains('[') {
            return handle_dot_notation(db, output_name, expr, line_no, ctx);
        }

        // Check for indexing syntax: name[indices]
        if expr.contains('[') && expr.contains(']') {
            return handle_indexing(db, output_name, expr, line_no, ctx);
        }
    }

    // Check for constructor functions: dataset("name")
    if tokens[0].starts_with("dataset(") {
        let paren_start = tokens[0].find('(').unwrap();
        let paren_end = tokens[0].rfind(')').ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: "Expected ')' in dataset() call".into(),
        })?;
        let name = tokens[0][paren_start + 1..paren_end]
            .trim_matches('"')
            .trim_matches('\'');

        // Create new tensor dataset
        let ds = crate::core::dataset::Dataset::new(name);
        db.register_tensor_dataset(ds.clone());
        db.register_dataset_var(output_name.to_string(), name.to_string());

        return Ok(DslOutput::TensorTable(ds, Vec::new()));
    }

    match tokens[0] {
        "ADD" => {
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = ADD a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_binary(ctx, output_name, left, right, BinaryOp::Add)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "SUBTRACT" => {
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = SUBTRACT a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_binary(ctx, output_name, left, right, BinaryOp::Subtract)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "MULTIPLY" => {
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = MULTIPLY a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_binary(ctx, output_name, left, right, BinaryOp::Multiply)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "DIVIDE" => {
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = DIVIDE a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_binary(ctx, output_name, left, right, BinaryOp::Divide)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "CORRELATE" => {
            // CORRELATE a WITH b
            if tokens.len() != 4 || tokens[2] != "WITH" {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = CORRELATE a WITH b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[3];
            db.eval_binary(ctx, output_name, left, right, BinaryOp::Correlate)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "SIMILARITY" => {
            // SIMILARITY a WITH b
            if tokens.len() != 4 || tokens[2] != "WITH" {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = SIMILARITY a WITH b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[3];
            db.eval_binary(ctx, output_name, left, right, BinaryOp::Similarity)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "DISTANCE" => {
            // DISTANCE a TO b
            if tokens.len() != 4 || tokens[2] != "TO" {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = DISTANCE a TO b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[3];
            db.eval_binary(ctx, output_name, left, right, BinaryOp::Distance)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "SCALE" => {
            // SCALE a BY 0.5
            if tokens.len() != 4 || tokens[2] != "BY" {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = SCALE a BY <number>".into(),
                });
            }
            let input_name = tokens[1];
            let factor: f32 = tokens[3].parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid scale factor: {}", tokens[3]),
            })?;
            db.eval_unary(ctx, output_name, input_name, UnaryOp::Scale(factor))
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "NORMALIZE" => {
            // NORMALIZE a
            if tokens.len() != 2 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = NORMALIZE a".into(),
                });
            }
            let input_name = tokens[1];
            db.eval_unary(ctx, output_name, input_name, UnaryOp::Normalize)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "MATMUL" => {
            // MATMUL a b
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = MATMUL a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_matmul(ctx, output_name, left, right)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "RESHAPE" => {
            // RESHAPE a TO [2, 3]
            if tokens.len() < 4 || tokens[2] != "TO" {
                // Tokens for shape might be split if they contain spaces inside brackets, but basic split is by whitespace.
                // Assuming simple case: RESHAPE a TO [2,3] (one token for shape if no spaces)
                // Or RESHAPE a TO [ 2, 3 ] (multiple tokens).
                // Parsing shape from tokens is tricky if split by whitespace.
                // Better to parse from the original string part.
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = RESHAPE a TO [dims]".into(),
                });
            }
            // Need to re-parse from expr string to handle shape properly vs tokens
            // expr is "RESHAPE a TO [2, 3]"
            // We can find " TO " and parse what's after.
            // But tokens[0] is RESHAPE.

            // Let's simplified approach based on tokens if shape has no spaces or we join them.
            // But parse_usize_list expects string like "[2, 3]".
            // Let's regex or find "TO".

            let to_index = expr.find(" TO ");
            if let Some(idx) = to_index {
                let shape_part = expr[idx + 4..].trim();
                let input_name = tokens[1];

                let dims = crate::utils::parsing::parse_usize_list(shape_part)
                    .map_err(|msg| DslError::Parse { line: line_no, msg })?;

                let shape = crate::core::tensor::Shape::new(dims);
                db.eval_reshape(ctx, output_name, input_name, shape)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })
            } else {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = RESHAPE a TO [dims]".into(),
                });
            }
        }
        "TRANSPOSE" => {
            // LET x = TRANSPOSE a
            if tokens.len() != 2 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = TRANSPOSE a".into(),
                });
            }
            let input_name = tokens[1];
            db.eval_unary(ctx, output_name, input_name, UnaryOp::Transpose)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "FLATTEN" => {
            // LET x = FLATTEN a
            if tokens.len() != 2 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = FLATTEN a".into(),
                });
            }
            let input_name = tokens[1];
            // Flatten is equivalent to reshape to [len]
            // We can resolve checking dims or use a eval_flatten helper if exists.
            // Using eval_reshape requires knowing the size -> need lookup?
            // Does db have eval_flatten? Assuming yes based on engine split.
            db.eval_unary(ctx, output_name, input_name, UnaryOp::Flatten)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        "STACK" => {
            // LET x = STACK a b c ...
            if tokens.len() < 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = STACK t1 t2 ...".into(),
                });
            }
            // tokens[0] is STACK
            let input_names: Vec<&str> = tokens[1..].to_vec();
            db.eval_stack(ctx, output_name, input_names, 0) // Axis 0 fixed for now
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })
        }
        other => Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown LET operation: {}", other),
        }),
    }
    .map(|_| DslOutput::Message(format!("Defined variable: {}", output_name)))
}

/// Handle indexing expressions: name[i, j, ...] or name[i:j, *]
fn handle_indexing(
    db: &mut TensorDb,
    output_name: &str,
    expr: &str,
    line_no: usize,
    ctx: &mut ExecutionContext,
) -> Result<DslOutput, DslError> {
    use crate::engine::kernels::SliceSpec;

    // Parse: tensor_name[spec1, spec2, ...]
    let bracket_start = expr.find('[').ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: "Expected '[' in indexing expression".into(),
    })?;

    let bracket_end = expr.find(']').ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: "Expected ']' in indexing expression".into(),
    })?;

    if bracket_end <= bracket_start {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Invalid bracket positions in indexing expression".into(),
        });
    }

    let tensor_name = expr[..bracket_start].trim();
    let specs_str = &expr[bracket_start + 1..bracket_end];

    // Parse specs: "0, 1" or "0:2, *" or "*, 1"
    let spec_parts: Vec<&str> = specs_str.split(',').map(|s| s.trim()).collect();
    let mut specs = Vec::new();
    let mut all_indices = true; // Track if all specs are simple indices

    for part in spec_parts {
        if part == "*" || part == ":" {
            // Wildcard: entire dimension
            specs.push(SliceSpec::All);
            all_indices = false;
        } else if part.contains(':') {
            // Range: start:end
            let range_parts: Vec<&str> = part.split(':').collect();
            if range_parts.len() != 2 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: format!("Invalid range spec: {}", part),
                });
            }

            let start = range_parts[0]
                .trim()
                .parse::<usize>()
                .map_err(|_| DslError::Parse {
                    line: line_no,
                    msg: format!("Invalid range start: {}", range_parts[0]),
                })?;

            let end = range_parts[1]
                .trim()
                .parse::<usize>()
                .map_err(|_| DslError::Parse {
                    line: line_no,
                    msg: format!("Invalid range end: {}", range_parts[1]),
                })?;

            specs.push(SliceSpec::Range(start, end));
            all_indices = false;
        } else {
            // Single index
            let idx = part.parse::<usize>().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid index: {}", part),
            })?;
            specs.push(SliceSpec::Index(idx));
        }
    }

    // If all specs are simple indices, use eval_index for backwards compatibility
    if all_indices {
        let indices: Vec<usize> = specs
            .iter()
            .filter_map(|s| {
                if let SliceSpec::Index(idx) = s {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect();

        db.eval_index(ctx, output_name, tensor_name, indices)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    } else {
        // Use eval_slice for advanced indexing
        db.eval_slice(ctx, output_name, tensor_name, specs)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    }

    Ok(DslOutput::Message(format!(
        "Defined variable: {}",
        output_name
    )))
}

/// Handle dot notation: name.field
fn handle_dot_notation(
    db: &mut TensorDb,
    output_name: &str,
    expr: &str,
    line_no: usize,
    ctx: &mut ExecutionContext,
) -> Result<DslOutput, DslError> {
    // Parse: object_name.field_name
    let parts: Vec<&str> = expr.split('.').collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: object.field".into(),
        });
    }

    let object_name = parts[0].trim();
    let field_name = parts[1].trim();

    // Try as dataset column access first (Legacy or Tensor-First)
    if db.get_dataset(object_name).is_ok() || db.get_tensor_dataset(object_name).is_some() {
        db.eval_column_access(ctx, output_name, object_name, field_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    } else {
        // Try as tuple field access
        db.eval_field_access(ctx, output_name, object_name, field_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    }

    Ok(DslOutput::Message(format!(
        "Defined variable: {}",
        output_name
    )))
}

fn parse_infix_op(expr: &str) -> Option<(&str, crate::engine::BinaryOp, &str)> {
    use crate::engine::BinaryOp;
    let ops = [
        ("+", BinaryOp::Add),
        ("-", BinaryOp::Subtract),
        ("*", BinaryOp::Multiply),
        ("/", BinaryOp::Divide),
    ];

    // Find the first operator that is NOT inside brackets
    let mut bracket_level = 0;
    for (i, c) in expr.char_indices() {
        if c == '[' {
            bracket_level += 1;
        } else if c == ']' {
            bracket_level -= 1;
        } else if bracket_level == 0 {
            // Check if this character matches any operator
            for (sym, op) in &ops {
                if expr[i..].starts_with(sym) {
                    return Some((expr[..i].trim(), op.clone(), expr[i + sym.len()..].trim()));
                }
            }
        }
    }
    None
}

fn evaluate_operand(
    db: &mut TensorDb,
    expr: &str,
    base_name: &str,
    suffix: &str,
    line_no: usize,
    ctx: &mut ExecutionContext,
) -> Result<String, DslError> {
    let expr = expr.trim();

    // Check if it's indexing: name[indices]
    if expr.contains('[') && expr.ends_with(']') {
        // Use a unique enough name for the temporary variable
        let temp_name = format!("_tmp_{}_{}", base_name, suffix);
        handle_indexing(db, &temp_name, expr, line_no, ctx)?;
        return Ok(temp_name);
    }

    // Check if it's dot notation: name.field
    // But avoid numbers like 1.0
    if expr.contains('.') && !expr.contains('[') && expr.parse::<f32>().is_err() {
        let temp_name = format!("_tmp_{}_{}", base_name, suffix);
        handle_dot_notation(db, &temp_name, expr, line_no, ctx)?;
        return Ok(temp_name);
    }

    // Check if it's a numeric literal
    if let Ok(f) = expr.parse::<f32>() {
        let temp_name = format!("_tmp_lit_{}_{}", base_name, suffix);
        let shape = crate::core::tensor::Shape::new(vec![]); // Scalar
        db.insert_named(&temp_name, shape, vec![f])
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        return Ok(temp_name);
    }

    // Otherwise assume it's a variable name
    Ok(expr.to_string())
}

fn handle_infix_op(
    db: &mut TensorDb,
    output_name: &str,
    left: &str,
    op: crate::engine::BinaryOp,
    right: &str,
    line_no: usize,
    ctx: &mut ExecutionContext,
) -> Result<DslOutput, DslError> {
    // Generate timestamp-based suffix for uniqueness to avoid collision in tight loops
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let left_name = evaluate_operand(
        db,
        left,
        output_name,
        &format!("L_{}", timestamp),
        line_no,
        ctx,
    )?;
    let right_name = evaluate_operand(
        db,
        right,
        output_name,
        &format!("R_{}", timestamp),
        line_no,
        ctx,
    )?;

    db.eval_binary(ctx, output_name, &left_name, &right_name, op)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::Message(format!(
        "Defined variable: {}",
        output_name
    )))
}
