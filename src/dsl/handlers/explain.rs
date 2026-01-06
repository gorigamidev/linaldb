use super::dataset::build_dataset_query_plan;
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use crate::query::planner::Planner;

pub fn handle_explain(db: &TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
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
        // } else if query_line.starts_with("SEARCH ") {
        let (_, plan) = super::search::build_search_query_plan(db, query_line, line_no)?;
        plan
    } else if query_line.starts_with("SELECT ") {
        super::dataset::build_select_query_plan(db, query_line, line_no)?
    } else {
        return Err(DslError::Parse {
            line: line_no,
            msg: "EXPLAIN only supports DATASET, SEARCH or SELECT queries".into(),
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
