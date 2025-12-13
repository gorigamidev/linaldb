use crate::dsl::{DslError, DslOutput};
use crate::engine::{BinaryOp, TensorDb, UnaryOp};

/// LET c = ADD a b
/// LET score = CORRELATE a WITH b
/// LET sim = SIMILARITY a WITH b
/// LET half = SCALE a BY 0.5
pub fn handle_let(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
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
            db.eval_binary(output_name, left, right, BinaryOp::Add)
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
            db.eval_binary(output_name, left, right, BinaryOp::Subtract)
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
            db.eval_binary(output_name, left, right, BinaryOp::Multiply)
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
            db.eval_binary(output_name, left, right, BinaryOp::Divide)
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
            db.eval_binary(output_name, left, right, BinaryOp::Correlate)
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
            db.eval_binary(output_name, left, right, BinaryOp::Similarity)
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
            db.eval_binary(output_name, left, right, BinaryOp::Distance)
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
            db.eval_unary(output_name, input_name, UnaryOp::Scale(factor))
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
            db.eval_unary(output_name, input_name, UnaryOp::Normalize)
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
            db.eval_matmul(output_name, left, right)
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
                db.eval_reshape(output_name, input_name, shape)
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
            db.eval_unary(output_name, input_name, UnaryOp::Transpose)
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
            db.eval_unary(output_name, input_name, UnaryOp::Flatten)
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
            db.eval_stack(output_name, input_names, 0) // Axis 0 fixed for now
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
