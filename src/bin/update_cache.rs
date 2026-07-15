use bramha::storage::answer_cache::DeterministicAnswerCache;

fn main() {
    let cache = DeterministicAnswerCache::load();
    cache
        .insert(
            "Hi",
            "Llama",
            &[],
            "Hello! How can I help you today?".to_string(),
        )
        .unwrap();
    println!("Inserted Hi into cache successfully.");
}
