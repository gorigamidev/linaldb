use super::dataset::build_dataset_query_plan;
use super::search::build_search_plan;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use crate::query::planner::Planner;

pub fn handle_explain(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("EXPLAIN").trim();
    let query_line = if rest.to_uppercase().starts_with("PLAN ") {
        rest[5..].trim()
    } else {
        rest
    };

    let logical_plan = if query_line.starts_with("DATASET ") {
        let (_, plan) = build_dataset_query_plan(db, query_line, line_no)?;
        plan
    } else if query_line.starts_with("SEARCH ") {
        // Need to parse SEARCH args carefully again or duplicate parsing logic?
        // Reuse handle_search parsing logic?
        // handle_search does: parse parts -> build_search_plan
        // We need to duplicate parsing or refactor `handle_search` to return `(target, LogicalPlan)` like dataset.
        // It's safer to duplicate parsing for now to avoid breaking handle_search signature too much if complex.
        // But `handle_search` is small. Let's refactor `handle_search` to be `build_search_query` returning plan.

        // Actually, `build_search_plan` takes parsed args.
        // I need to parse the SEARCH line here.
        // Let's create a helper `parse_search_line` in `search.rs`?
        // Or just implement parsing here (duplication).
        // Let's implement parsing here for now, it's not too long.

        // Parse: SEARCH <target> FROM <source> ...
        let rest = query_line.trim_start_matches("SEARCH").trim();
        let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
        if parts.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: SEARCH <target> FROM <source> ...".into(),
            });
        }
        let parts2: Vec<&str> = parts[1].trim().splitn(2, " QUERY ").collect();
        if parts2.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: ... FROM <source> QUERY ...".into(),
            });
        }
        let source_name = parts2[0].trim();
        let after_query = parts2[1].trim();

        let parts3: Vec<&str> = after_query.splitn(2, " ON ").collect();
        if parts3.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: ... QUERY <vector> ON <column> ...".into(),
            });
        }
        let vector_str = parts3[0].trim();
        let after_on = parts3[1].trim();

        let parts4: Vec<&str> = if after_on.contains(" K=") {
            after_on.splitn(2, " K=").collect()
        } else if after_on.contains(" K =") {
            after_on.splitn(2, " K =").collect()
        } else {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: ... ON <column> K=<k>".into(),
            });
        };
        let column_name = parts4[0].trim();
        let k_str = parts4[1].trim();
        let k: usize = k_str.parse().map_err(|_| DslError::Parse {
            line: line_no,
            msg: format!("Invalid K: {}", k_str),
        })?;

        use crate::core::value::Value;
        let query_val = super::dataset::parse_single_value(vector_str, line_no)?;
        let query_tensor = match query_val {
            Value::Vector(data) => {
                use crate::core::tensor::{Shape, Tensor, TensorId};
                Tensor::new(TensorId(0), Shape::new(vec![data.len()]), data).map_err(|e| {
                    DslError::Parse {
                        line: line_no,
                        msg: e,
                    }
                })?
            }
            _ => {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Query must be vector".into(),
                })
            }
        };

        let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

        build_search_plan(
            source_name,
            source_ds.schema.clone(),
            column_name,
            query_tensor,
            k,
        )
    } else {
        return Err(DslError::Parse {
            line: line_no,
            msg: "EXPLAIN only supports DATASET or SEARCH queries".into(),
        });
    };

    let planner = Planner::new(db);
    let physical_plan =
        planner
            .create_physical_plan(&logical_plan)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;

    let output = format!(
        "--- Logical Plan ---\n{:#?}\n\n--- Physical Plan ---\n{:#?}",
        logical_plan, physical_plan
    );
    // PhysicalPlan is a trait object, can't derive Debug easily on Box<dyn ...>.
    // Usually we implement Display or Debug manually.
    // For MVP, showing LogicalPlan is enough to prove planner works (it shows Filter vs Scan etc).
    // Adding Debug to specific PhysicalPlan structs works but Box<dyn PhysicalPlan> needs it in trait bound?
    // Trait `PhysicalPlan` is `Send + Sync`. Adding `Debug` to it?
    // `pub trait PhysicalPlan: Send + Sync + std::fmt::Debug`
    // If I add Debug to PhysicalPlan trait, I can print it.

    Ok(DslOutput::Message(output))
}
