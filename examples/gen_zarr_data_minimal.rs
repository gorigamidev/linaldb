use zarrs::filesystem::FilesystemStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = FilesystemStore::new("test"); // Returns a Result or a builder?
    Ok(())
}
