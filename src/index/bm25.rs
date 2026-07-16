use std::collections::HashMap;

/// A high-performance BM25 Lexical Keyword Search Index
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BM25Index {
    pub doc_count: usize,
    pub avg_doc_len: f64,
    pub doc_lens: Vec<usize>,
    pub doc_ids: Vec<String>,
    pub term_freqs: Vec<HashMap<String, usize>>, // document index -> (term -> count)
    pub doc_freqs: HashMap<String, usize>,       // term -> number of docs containing it
    pub k1: f64,                                 // term frequency saturation parameter
    pub b: f64,                                  // document length normalization parameter
}

impl BM25Index {
    /// Creates a fresh BM25 index with standard defaults
    pub fn new() -> Self {
        BM25Index {
            doc_count: 0,
            avg_doc_len: 0.0,
            doc_lens: Vec::new(),
            doc_ids: Vec::new(),
            term_freqs: Vec::new(),
            doc_freqs: HashMap::new(),
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Cleanly tokenizes text into lowercase alphanumeric terms
    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Indexes a document's lexical terms
    pub fn add_document(&mut self, doc_id: String, text: &str) {
        let tokens = Self::tokenize(text);
        if tokens.is_empty() {
            return;
        }

        let doc_len = tokens.len();
        self.doc_lens.push(doc_len);
        self.doc_ids.push(doc_id);

        let mut local_freqs = HashMap::new();
        for token in tokens {
            *local_freqs.entry(token).or_insert(0) += 1;
        }

        // Record global term document frequencies
        for term in local_freqs.keys() {
            *self.doc_freqs.entry(term.clone()).or_insert(0) += 1;
        }

        self.term_freqs.push(local_freqs);
        self.doc_count += 1;

        // Recalculate average document length across active space
        let total_len: usize = self.doc_lens.iter().sum();
        self.avg_doc_len = total_len as f64 / self.doc_count as f64;
    }

    /// Performs BM25 scoring search for a keyword query
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(String, f64)> {
        let query_tokens = Self::tokenize(query);
        if query_tokens.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }

        let mut scores = Vec::new();

        for doc_idx in 0..self.doc_count {
            let doc_id = &self.doc_ids[doc_idx];
            let doc_len = self.doc_lens[doc_idx] as f64;
            let term_freq_map = &self.term_freqs[doc_idx];
            let mut score = 0.0;

            for term in &query_tokens {
                if let Some(&tf) = term_freq_map.get(term) {
                    let tf = tf as f64;
                    // Retrieve global doc frequency or default to 0
                    let df = *self.doc_freqs.get(term).unwrap_or(&0) as f64;

                    // IDF calculation using standard BM25 logarithmic scaling (smoothed)
                    let idf = ((self.doc_count as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();

                    // Term frequency scaling with saturation k1 and length normalization b
                    let num = tf * (self.k1 + 1.0);
                    let den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len));

                    score += idf * (num / den);
                }
            }

            if score > 0.0 {
                scores.push((doc_id.clone(), score));
            }
        }

        // Sort descending by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Resets the index space completely
    pub fn clear(&mut self) {
        self.doc_count = 0;
        self.avg_doc_len = 0.0;
        self.doc_lens.clear();
        self.doc_ids.clear();
        self.term_freqs.clear();
        self.doc_freqs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_tokenize() {
        let text = "Hello, Bramha! The Cognitive Database: 123.";
        let tokens = BM25Index::tokenize(text);
        assert_eq!(
            tokens,
            vec!["hello", "bramha", "the", "cognitive", "database", "123"]
        );
    }

    #[test]
    fn test_bm25_search_indexing() {
        let mut index = BM25Index::new();
        index.add_document(
            "doc1".to_string(),
            "The quick brown fox jumps over the lazy dog",
        );
        index.add_document(
            "doc2".to_string(),
            "Bramha is a high performance cognitive database",
        );
        index.add_document("doc3".to_string(), "Rust is fast, safe, and concurrent");

        assert_eq!(index.doc_count, 3);
        assert!(index.avg_doc_len > 0.0);

        // Search keyword "Bramha"
        let results = index.search("Bramha cognitive", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "doc2");

        // Search fast
        let results_fast = index.search("fast Rust", 5);
        assert!(!results_fast.is_empty());
        assert_eq!(results_fast[0].0, "doc3");
    }
}
