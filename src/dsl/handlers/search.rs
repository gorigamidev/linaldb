use crate::core::value::Value;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use crate::query::logical::LogicalPlan;
use crate::query::planner::Planner;

use super::dataset::parse_single_value;

/// SEARCH target FROM source QUERY vector ON column K=k
pub fn handle_search(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("SEARCH").trim();

    // Parse: <target> FROM <source> ...
    let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: SEARCH <target> FROM <source> ...".into(),
        });
    }

    let target_name = parts[0].trim();
    let query_part = parts[1].trim();

    // Parse: <source> QUERY <vector> ON <column> K=<k>
    // We expect explicit keywords: QUERY, ON, K
    // Split by " QUERY "
    let parts2: Vec<&str> = query_part.splitn(2, " QUERY ").collect();
    if parts2.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: ... FROM <source> QUERY <vector> ...".into(),
        });
    }
    let source_name = parts2[0].trim();
    let after_query = parts2[1].trim();

    // Split by " ON "
    let parts3: Vec<&str> = after_query.splitn(2, " ON ").collect();
    if parts3.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: ... QUERY <vector> ON <column> ...".into(),
        });
    }
    let vector_str = parts3[0].trim();
    let after_on = parts3[1].trim();

    // Split by " K=" (case sensitive? usually DSL is flexible but let's stick to simple first)
    // Or just " K " and parse next
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

    // Parse K
    let k: usize = k_str.parse().map_err(|_| DslError::Parse {
        line: line_no,
        msg: format!("Invalid K: {}", k_str),
    })?;

    // Parse Vector
    let query_val = parse_single_value(vector_str, line_no)?;
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
                msg: format!("Query must be a vector, got {:?}", query_val),
            })
        }
    };

    // Get source dataset schema
    let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let source_schema = source_ds.schema.clone();

    let search_plan = build_search_plan(source_name, source_schema, column_name, query_tensor, k);

    // Execute Plan
    let planner = Planner::new(db);
    let physical_plan =
        planner
            .create_physical_plan(&search_plan)
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
    db.create_dataset(target_name.to_string(), result_schema.clone())
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    let target_ds = db
        .get_dataset_mut(target_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    target_ds.rows = result_rows;
    target_ds
        .metadata
        .update_stats(&target_ds.schema, &target_ds.rows);

    Ok(DslOutput::Message(format!(
        "Search completed. Found {} results.",
        target_ds.len()
    )))
}

pub fn build_search_plan(
    source_name: &str,
    source_schema: std::sync::Arc<crate::core::tuple::Schema>,
    column_name: &str,
    query_tensor: crate::core::tensor::Tensor,
    k: usize,
) -> LogicalPlan {
    // Scan -> VectorSearch
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
