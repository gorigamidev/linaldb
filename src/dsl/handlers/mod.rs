pub mod dataset;
pub mod explain;
pub mod index;
pub mod instance;
pub mod introspection;
pub mod metadata;
pub mod operations;
pub mod persistence;
pub mod search;
pub mod semantics;
pub mod tensor;

pub use dataset::{handle_dataset, handle_insert};
pub use instance::{handle_create_database, handle_drop_database, handle_use_database};
pub use introspection::handle_show;
pub use operations::handle_let;
pub use semantics::{handle_attach, handle_bind, handle_derive};
pub use tensor::handle_define;
