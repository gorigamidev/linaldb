use crate::core::value::Value;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

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

    // Get source dataset
    let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;

    // Get Index
    let index = source_ds
        .get_index(column_name)
        .ok_or_else(|| DslError::Engine {
            line: line_no,
            source: crate::engine::error::EngineError::InvalidOp(format!(
                "No index found on column '{}'. SEARCH currently requires an index.",
                column_name
            )),
        })?;

    if index.index_type() != crate::core::index::IndexType::Vector {
        return Err(DslError::Engine {
            line: line_no,
            source: crate::engine::error::EngineError::InvalidOp(format!(
                "Index on '{}' is not a VECTOR index.",
                column_name
            )),
        });
    }

    // Perform Search
    let results = index
        .search(&query_tensor, k)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: crate::engine::error::EngineError::InvalidOp(e),
        })?;

    // Results are (row_id, score)
    // We want to return a Dataset with the actual rows.
    // Ideally we should preserve order. `get_rows_by_ids` does preserve order of input IDs.
    let row_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
    let matched_rows = source_ds.get_rows_by_ids(&row_ids);

    // Create target dataset
    // We use the same schema as source
    db.create_dataset(target_name.to_string(), source_ds.schema.clone())
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

    target_ds.rows = matched_rows;
    // Update stats?

    Ok(DslOutput::Message(format!(
        "Search completed. Found {} results.",
        target_ds.len()
    )))
}
