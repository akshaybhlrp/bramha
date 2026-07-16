use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct EvidenceMap {
    pub sentence: String,
    pub supporting_chunk_ids: Vec<String>,
    pub overlap_score: f32,
}

pub struct EvidenceMapper;

impl EvidenceMapper {
    pub fn new() -> Self {
        EvidenceMapper
    }

    /// Parses a generated completion into sentences and maps them against retrieved chunks.
    pub fn map_evidence(
        &self,
        completion: &str,
        retrieved_chunks: &[(String, String)], // (chunk_id, chunk_text)
    ) -> Vec<EvidenceMap> {
        let sentences = self.split_into_sentences(completion);
        let mut overlap_maps = Vec::new();

        for sentence in sentences {
            let sentence_tokens: HashSet<&str> = sentence.split_whitespace().collect();
            let mut best_score = 0.0;
            let mut supporting_chunks = Vec::new();

            for (chunk_id, chunk_text) in retrieved_chunks {
                let chunk_tokens: HashSet<&str> = chunk_text.split_whitespace().collect();
                let intersection = sentence_tokens.intersection(&chunk_tokens).count();

                if sentence_tokens.is_empty() {
                    continue;
                }

                let score = intersection as f32 / sentence_tokens.len() as f32;
                if score > 0.3 {
                    // Threshold for considering it "supported"
                    supporting_chunks.push(chunk_id.clone());
                    if score > best_score {
                        best_score = score;
                    }
                }
            }

            overlap_maps.push(EvidenceMap {
                sentence: sentence.to_string(),
                supporting_chunk_ids: supporting_chunks,
                overlap_score: best_score,
            });
        }

        overlap_maps
    }

    fn split_into_sentences<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split(|c| c == '.' || c == '?' || c == '!')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_overlap_mapping_generation() {
        let mapper = EvidenceMapper::new();
        let chunks = vec![
            (
                "chunk1".to_string(),
                "The quick brown fox jumps over the lazy dog".to_string(),
            ),
            (
                "chunk2".to_string(),
                "Rust is a systems programming language".to_string(),
            ),
        ];
        let completion = "The quick brown fox jumps. Rust is great.";

        let maps = mapper.map_evidence(completion, &chunks);
        assert_eq!(maps.len(), 2);
        assert!(maps[0].supporting_chunk_ids.contains(&"chunk1".to_string()));
        assert!(maps[1].supporting_chunk_ids.contains(&"chunk2".to_string()));
    }
}
