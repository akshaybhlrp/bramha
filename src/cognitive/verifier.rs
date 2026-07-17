use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Pass,
    Fail,
    Flag,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierPolicy {
    pub min_overlap_ratio: f32,
    pub enable_semantic_verification: bool,
    pub strict_mode: bool,
}

impl Default for VerifierPolicy {
    fn default() -> Self {
        Self {
            min_overlap_ratio: 0.3,
            enable_semantic_verification: false,
            strict_mode: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub status: VerificationStatus,
    pub score: f32,
    pub matched_facts: Vec<String>,
    pub unmatched_claims: Vec<String>,
    pub details: String,
}

pub struct ModelVerifier {
    pub policy: VerifierPolicy,
}

impl ModelVerifier {
    pub fn new(policy: VerifierPolicy) -> Self {
        Self { policy }
    }

    /// Tokenize text into lowercase words, filtering out common stop words
    fn get_content_words(text: &str) -> Vec<String> {
        let stop_words = vec![
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in",
            "on", "at", "for", "with", "by", "about", "as",
        ];
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .filter(|w| w.len() > 1 && !stop_words.contains(w))
            .map(|w| w.to_string())
            .collect()
    }

    /// Verifies the output completion against the retrieved context chunks (ground truth facts)
    pub fn verify(
        &self,
        completion: &str,
        context_chunks: &[(String, String)],
    ) -> VerificationReport {
        println!(
            "🔍 Running Model Verifier on output ({} bytes) against {} context chunks...",
            completion.len(),
            context_chunks.len()
        );

        if completion.trim().is_empty() {
            return VerificationReport {
                status: VerificationStatus::Fail,
                score: 0.0,
                matched_facts: Vec::new(),
                unmatched_claims: Vec::new(),
                details: "Empty completion provided.".to_string(),
            };
        }

        if context_chunks.is_empty() {
            let status = if self.policy.strict_mode {
                VerificationStatus::Fail
            } else {
                VerificationStatus::Flag
            };
            return VerificationReport {
                status,
                score: 0.0,
                matched_facts: Vec::new(),
                unmatched_claims: vec![completion.to_string()],
                details: "No context chunks provided for verification.".to_string(),
            };
        }

        // Split completion into sentences/claims
        let raw_sentences: Vec<&str> = completion
            .split(['.', '!', '?'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut matched_facts = Vec::new();
        let mut unmatched_claims = Vec::new();
        let mut total_score = 0.0;

        // Process each sentence as a distinct claim
        for sentence in &raw_sentences {
            let claim_words = Self::get_content_words(sentence);
            if claim_words.is_empty() {
                continue;
            }

            let mut max_overlap = 0.0;

            // Compare claim against each context chunk
            for (_, chunk_text) in context_chunks {
                let context_words = Self::get_content_words(chunk_text);
                if context_words.is_empty() {
                    continue;
                }

                // Count matching words
                let mut match_count = 0;
                for word in &claim_words {
                    if context_words.contains(word) {
                        match_count += 1;
                    }
                }

                let ratio = match_count as f32 / claim_words.len() as f32;
                if ratio > max_overlap {
                    max_overlap = ratio;
                }
            }

            if max_overlap >= self.policy.min_overlap_ratio {
                matched_facts.push(sentence.to_string());
                total_score += 1.0;
            } else {
                unmatched_claims.push(sentence.to_string());
            }
        }

        let num_claims = raw_sentences.len().max(1);
        let final_score = total_score / num_claims as f32;

        let status = if final_score >= 0.8 {
            VerificationStatus::Pass
        } else if final_score >= 0.5 && !self.policy.strict_mode {
            VerificationStatus::Flag
        } else {
            VerificationStatus::Fail
        };

        let details = format!(
            "Verification completed. Claims validated: {}/{}. Score: {:.2}",
            total_score, num_claims, final_score
        );

        VerificationReport {
            status,
            score: final_score,
            matched_facts,
            unmatched_claims,
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_pass_flow() {
        let policy = VerifierPolicy {
            min_overlap_ratio: 0.25,
            ..Default::default()
        };
        let verifier = ModelVerifier::new(policy);

        let context = vec![
            (
                "doc1".to_string(),
                "Bramha runs local intelligence natively on CPU and GPU.".to_string(),
            ),
            (
                "doc2".to_string(),
                "The system achieves stable performance using zero-copy tensor mapping."
                    .to_string(),
            ),
        ];

        let completion = "Bramha runs local intelligence on CPU. It uses zero-copy tensor mapping.";
        let report = verifier.verify(completion, &context);

        assert_eq!(report.status, VerificationStatus::Pass);
        assert!(report.score >= 0.8);
        assert_eq!(report.unmatched_claims.len(), 0);
    }

    #[test]
    fn test_verifier_fail_hallucination_flow() {
        let policy = VerifierPolicy {
            min_overlap_ratio: 0.3,
            strict_mode: true,
            ..Default::default()
        };
        let verifier = ModelVerifier::new(policy);

        let context = vec![(
            "doc1".to_string(),
            "Bramha runs local intelligence natively on CPU and GPU.".to_string(),
        )];

        // The second sentence is a hallucination not grounded in the context
        let completion =
            "Bramha runs local intelligence. It also connects to external cloud servers.";
        let report = verifier.verify(completion, &context);

        assert_eq!(report.status, VerificationStatus::Fail);
        assert!(report.score < 0.8);
        assert_eq!(report.unmatched_claims.len(), 1);
        assert!(report.unmatched_claims[0].contains("external cloud"));
    }
}
