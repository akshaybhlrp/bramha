use crate::core::collection::SearchResult;
use crate::inference::engine::InferenceEngine;
use crate::storage::Database;
use std::sync::Arc;
use std::collections::HashMap;

/// Implementation of high-precision retrieval strategies (Multi-Query Expansion, HyDE)
pub struct RetrievalStrategies;

impl RetrievalStrategies {
    /// Performs Multi-Query search: generates 3 alternative reformulations,
    /// embeds them, performs parallel vector searches, and fuses using RRF.
    pub async fn multi_query_search(
        db: Arc<Database>,
        collection_name: &str,
        model_name: &str,
        raw_query: &str,
        k: usize,
        use_index: bool,
    ) -> Result<Vec<SearchResult>, String> {
        // 1. Generate 3 alternative search queries using the LLaMA model
        let prompt = format!(
            "<|system|>\nYou are a professional query expansion search assistant. \
            Generate exactly 3 alternative search queries for the user prompt. \
            Return ONLY the queries, one per line, with no bullets, numbering, or intro text.\n\
            <|user|>\n{}\n<|assistant|>\n",
            raw_query
        );

        let gen_result = InferenceEngine::new(None).generate(
            db.clone(),
            model_name,
            &prompt,
            40, // 40 tokens is perfect for 3 queries
            0.3,
            None,
            None,
        ).await?;

        let mut queries = Vec::new();
        queries.push(raw_query.to_string()); // Include the original query

        for line in gen_result.completion.lines() {
            let clean = line.trim()
                .trim_start_matches(|c: char| c.is_numeric() || c == '.' || c == '-' || c == '*' || c == ')')
                .trim()
                .to_string();
            if !clean.is_empty() && queries.len() < 4 {
                queries.push(clean);
            }
        }

        println!("🔍 Multi-Query Expansion Formulations generated: {:?}", queries);

        // 2. Fetch embeddings for all 4 queries using the native Rust WGPU embedder
        let embedder = crate::inference::embedder::Embedder::get_global().await
            .map_err(|e| format!("Failed to initialize native embedder: {}", e))?;
        
        let mut embeddings = Vec::new();
        for query in &queries {
            match embedder.embed(query) {
                Ok(emb) => embeddings.push(emb),
                Err(e) => println!("⚠️ Native query embedding error: {}", e),
            }
        }

        if embeddings.is_empty() {
            return Err("Failed to generate embeddings for query expansion".to_string());
        }

        // 3. Search collection with all queries and fuse via RRF
        let state = db.state.read().await;
        let collection = state.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        let mut all_search_results = Vec::new();
        for emb in &embeddings {
            let results = collection.search(emb, k * 2, None, use_index);
            all_search_results.push(results);
        }

        // Apply RRF merging
        let rrf_k = 60.0;
        let mut rrf_scores: HashMap<String, f64> = HashMap::new();

        for results in &all_search_results {
            for (rank, res) in results.iter().enumerate() {
                let score = 1.0 / (rrf_k + rank as f64);
                *rrf_scores.entry(res.id.clone()).or_insert(0.0) += score;
            }
        }

        let mut fused_results: Vec<SearchResult> = rrf_scores
            .into_iter()
            .map(|(id, rrf_score)| {
                let metadata = collection.vectors.get(&id).and_then(|v| v.metadata.clone());
                SearchResult {
                    id,
                    score: rrf_score as f32,
                    metadata,
                    ..Default::default()
                }
            })
            .collect();

        // Sort descending by RRF score
        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        fused_results.truncate(k);

        Ok(fused_results)
    }

    /// Performs HyDE (Hypothetical Document Embeddings) search: generates a hypothetical
    /// answer first, embeds that answer, and runs vector search using the answer vector.
    pub async fn hyde_search(
        db: Arc<Database>,
        collection_name: &str,
        model_name: &str,
        raw_query: &str,
        k: usize,
        use_index: bool,
    ) -> Result<Vec<SearchResult>, String> {
        // 1. Generate hypothetical response using the LLaMA model
        let prompt = format!(
            "<|system|>\nYou are a factual assistant. Write a short, highly informative paragraph answering the user's question. \
            Do not introduce the response or write anything other than the factual paragraph answer itself.\n\
            <|user|>\n{}\n<|assistant|>\n",
            raw_query
        );

        let gen_result = InferenceEngine::new(None).generate(
            db.clone(),
            model_name,
            &prompt,
            60,
            0.5,
            None,
            None,
        ).await?;

        let hypothetical_doc = gen_result.completion;
        println!("✨ HyDE Hypothetical Document generated:\n{}", hypothetical_doc);

        // 2. Fetch embedding of the hypothetical document using the native Rust WGPU embedder
        let embedder = crate::inference::embedder::Embedder::get_global().await
            .map_err(|e| format!("Failed to initialize native embedder: {}", e))?;
        let embedding = embedder.embed(&hypothetical_doc)
            .map_err(|e| format!("Embedding failed: {}", e))?;

        // 3. Search collection with the hypothetical document embedding
        let state = db.state.read().await;
        let collection = state.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        let results = collection.search(&embedding, k, None, use_index);
        Ok(results)
    }
}
