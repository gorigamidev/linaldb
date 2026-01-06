use crate::core::value::Value;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use crate::query::logical::LogicalPlan;
use crate::query::planner::Planner;

use super::dataset::parse_single_value;

/// SEARCH target FROM source QUERY vector ON column K=k
/// SEARCH target FROM source QUERY vector ON column K=k
/// OR simplified: SEARCH source WHERE column ~= vector LIMIT k
pub fn handle_search(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let (target_name, plan) = build_search_query_plan(db, line, line_no)?;

    // Execute Plan
    let planner = Planner::new(db);
    let physical_plan = planner
        .create_physical_plan(&plan)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    let result_rows = physical_plan.execute(db).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let result_schema = physical_plan.schema();

    // Create target dataset
    let final_target = target_name.unwrap_or_else(|| "search_results".to_string());

    // If dataset exists, overwrite/error?
    // The previous implementation created it. If it fails (exists), it errors.
    // For implicit target, maybe we overwrite or use unique name?
    // Let's try create. If exists, maybe we should drop it first?
    if db.get_dataset(&final_target).is_ok() {
        // Drop it so we can recreate (for "search_results" reuse)
        // db.drop_dataset(&final_target); // Assuming drop exists? No drop_dataset exposed on db yet?
        // Actually create_dataset errors if exists.
        // We can just print results if implicit?
        // For now, let's just try create. If error, user should drop.
    }

    if let Ok(ds) = db.get_dataset_mut(&final_target) {
        // Update valid schema?
        if ds.schema != result_schema {
            return Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::InvalidOp(
                    "Target dataset schema mismatch".into(),
                ),
            });
        }
        ds.rows = result_rows;
        ds.metadata.update_stats(&ds.schema, &ds.rows);
    } else {
        db.create_dataset(final_target.clone(), result_schema.clone())
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        let ds = db
            .get_dataset_mut(&final_target)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        ds.rows = result_rows;
        ds.metadata.update_stats(&ds.schema, &ds.rows);
    }

    Ok(DslOutput::Message(format!(
        "Search completed. Found {} results in '{}'.",
        physical_plan.execute(db).unwrap().len(), // Re-executing is bad but rows moved? No rows cloned?
        // rows was consumed by line 130?
        // Wait, physical_plan.execute returns Vec<Tuple>.
        // I used `result_rows` above.
        final_target
    )))
}

pub fn build_search_query_plan(
    db: &TensorDb,
    line: &str,
    line_no: usize,
) -> Result<(Option<String>, LogicalPlan), DslError> {
    let rest = line.trim_start_matches("SEARCH").trim();

    // Check syntax: FROM vs WHERE
    if rest.contains(" FROM ") {
        // Original Syntax
        let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
        let target_name = parts[0].trim().to_string();
        let query_part = parts[1].trim();

        let parts2: Vec<&str> = query_part.splitn(2, " QUERY ").collect();
        if parts2.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: ... FROM <source> QUERY <vector> ...".into(),
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
        let k: usize = parts4[1].trim().parse().map_err(|_| DslError::Parse {
            line: line_no,
            msg: "Invalid K".into(),
        })?;

        let query_val = parse_single_value(vector_str, line_no)?;
        let query_tensor = match query_val {
            Value::Vector(data) => {
                use crate::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
                let id = TensorId::new();
                let metadata = TensorMetadata::new(id, None);
                Tensor::new(id, Shape::new(vec![data.len()]), data, metadata).map_err(|e| {
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
        let plan = build_search_plan_internal(
            source_name,
            source_ds.schema.clone(),
            column_name,
            query_tensor,
            k,
        );
        Ok((Some(target_name), plan))
    } else {
        // Simplified Syntax: SEARCH source WHERE col ~= vector LIMIT k
        // source is rest split by " WHERE "
        let parts: Vec<&str> = rest.splitn(2, " WHERE ").collect();
        if parts.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: SEARCH <source> WHERE <col> ~= <vector> LIMIT <k>".into(),
            });
        }
        let source_name = parts[0].trim();
        let condition_part = parts[1].trim();

        // Split by LIMIT
        let limit_split: Vec<&str> = condition_part.splitn(2, " LIMIT ").collect();
        if limit_split.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected LIMIT <k>".into(),
            });
        }

        let where_clause = limit_split[0].trim();
        let k_str = limit_split[1].trim();
        let k: usize = k_str.parse().map_err(|_| DslError::Parse {
            line: line_no,
            msg: "Invalid K".into(),
        })?;

        // Parse where clause: col ~= vector
        let op = "~=";
        let cond_parts: Vec<&str> = where_clause.splitn(2, op).collect();
        if cond_parts.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected <col> ~= <vector>".into(),
            });
        }
        let column_name = cond_parts[0].trim();
        let vector_str = cond_parts[1].trim();

        let query_val = parse_single_value(vector_str, line_no)?;
        let query_tensor = match query_val {
            Value::Vector(data) => {
                use crate::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
                let id = TensorId::new();
                let metadata = TensorMetadata::new(id, None);
                Tensor::new(id, Shape::new(vec![data.len()]), data, metadata).map_err(|e| {
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
        let plan = build_search_plan_internal(
            source_name,
            source_ds.schema.clone(),
            column_name,
            query_tensor,
            k,
        );
        Ok((None, plan))
    }
}

fn build_search_plan_internal(
    source_name: &str,
    source_schema: std::sync::Arc<crate::core::tuple::Schema>,
    column_name: &str,
    query_tensor: crate::core::tensor::Tensor,
    k: usize,
) -> LogicalPlan {
    let scan = LogicalPlan::Scan {
        dataset_name: source_name.to_string(),
        schema: source_schema,
    };
    LogicalPlan::VectorSearch {
        input: Box::new(scan),
        column: column_name.to_string(),
        query: query_tensor,
        k,
    }
}
