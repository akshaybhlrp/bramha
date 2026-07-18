//! Query-conditional sparse weight paging (SPANDA v5.2 Phase 1+). RAM-resident
//! `Vec` storage per the design decision recorded for this project — mmap was
//! explicitly rejected, so pages here are plain owned buffers, not memory-mapped
//! file views. Eviction is driven by a per-page confidence score derived from
//! recent hit history, not by pure recency (LRU) — a page that keeps getting
//! predicted-and-hit stays resident even if something else was touched more
//! recently.
//!
//! `PagingEngine` gates itself behind `jaccard::AccessTracker::evaluate_gate()`.
//! Before this module, spanda-engine had no paging, no eviction, and no gate —
//! this ties them together in the order the plan specifies: measure
//! predictability first, only then allow eviction/prefetch policy to build on it.

use crate::jaccard::{AccessTracker, GateResult};
use std::collections::{HashMap, HashSet};

pub type PageId = u64;

/// One RAM-resident page of tensor data. Plain owned `Vec<f32>` — no mmap.
#[derive(Debug, Clone)]
pub struct Page {
    pub id: PageId,
    pub data: Vec<f32>,
}

impl Page {
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Per-page confidence bookkeeping used by the eviction policy. Confidence rises
/// on each step the page is actually touched, and decays on each step it's
/// resident but unused.
#[derive(Debug, Clone, Copy)]
struct PageStats {
    confidence: f32,
    last_access_step: u64,
    hits: u32,
}

const CONFIDENCE_DECAY_PER_STEP: f32 = 0.02;
const CONFIDENCE_BOOST_ON_HIT: f32 = 0.25;
const INITIAL_CONFIDENCE: f32 = 0.5;

pub struct PageStore {
    budget_bytes: usize,
    resident_bytes: usize,
    pages: HashMap<PageId, Page>,
    stats: HashMap<PageId, PageStats>,
    current_step: u64,
}

impl PageStore {
    pub fn new(budget_bytes: usize) -> Self {
        PageStore {
            budget_bytes,
            resident_bytes: 0,
            pages: HashMap::new(),
            stats: HashMap::new(),
            current_step: 0,
        }
    }

    pub fn is_resident(&self, id: PageId) -> bool {
        self.pages.contains_key(&id)
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Called once per inference step with the set of pages actually touched.
    /// Boosts confidence for hits, decays everyone else. Does not evict by
    /// itself — eviction happens lazily in `insert` when budget is exceeded.
    pub fn begin_step(&mut self, accessed_ids: &[PageId]) {
        self.current_step += 1;
        let accessed: HashSet<PageId> = accessed_ids.iter().copied().collect();

        for (id, stat) in self.stats.iter_mut() {
            if accessed.contains(id) {
                stat.confidence = (stat.confidence + CONFIDENCE_BOOST_ON_HIT).min(1.0);
                stat.last_access_step = self.current_step;
                stat.hits += 1;
            } else {
                stat.confidence = (stat.confidence - CONFIDENCE_DECAY_PER_STEP).max(0.0);
            }
        }
    }

    /// Insert a freshly loaded page, evicting lowest-confidence residents first
    /// if this would exceed the byte budget.
    pub fn insert(&mut self, page: Page) {
        let incoming_size = page.size_bytes();

        while self.resident_bytes + incoming_size > self.budget_bytes && !self.pages.is_empty() {
            match self.pick_eviction_victim() {
                Some(victim_id) => self.evict(victim_id),
                None => break,
            }
        }

        self.resident_bytes += incoming_size;
        self.stats.insert(
            page.id,
            PageStats {
                confidence: INITIAL_CONFIDENCE,
                last_access_step: self.current_step,
                hits: 0,
            },
        );
        self.pages.insert(page.id, page);
    }

    pub fn get(&self, id: PageId) -> Option<&Page> {
        self.pages.get(&id)
    }

    fn evict(&mut self, id: PageId) {
        if let Some(page) = self.pages.remove(&id) {
            self.resident_bytes -= page.size_bytes();
        }
        self.stats.remove(&id);
    }

    /// Lowest confidence wins eviction (predictor-confidence policy, not LRU).
    /// Ties broken by oldest last-access-step so fresher pages are still favored.
    fn pick_eviction_victim(&self) -> Option<PageId> {
        self.stats
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.last_access_step.cmp(&b.last_access_step))
            })
            .map(|(&id, _)| id)
    }
}

/// Ties the Phase 0 gate to the page store: paging/eviction only activates once
/// access patterns are measured predictable enough. If the gate hasn't passed,
/// `PagingEngine` refuses to page and callers must fall back to loading
/// everything densely — matching the existing shadow-scan "fall back to Banker
/// Mode" pattern already used elsewhere in bramha-engine's cpu_engine.rs.
pub struct PagingEngine {
    tracker: AccessTracker,
    store: Option<PageStore>,
    budget_bytes: usize,
}

impl PagingEngine {
    pub fn new(budget_bytes: usize) -> Self {
        PagingEngine {
            tracker: AccessTracker::new(),
            store: None,
            budget_bytes,
        }
    }

    /// Record one step's access set for gating purposes, and — if paging is
    /// already active — update eviction confidence for that step too.
    pub fn record_step(&mut self, accessed_ids: &[u64]) {
        self.tracker.record_access(accessed_ids);
        if let Some(store) = self.store.as_mut() {
            store.begin_step(accessed_ids);
        }
    }

    pub fn gate_status(&self) -> GateResult {
        self.tracker.evaluate_gate()
    }

    /// Attempt to activate paging. Returns false and leaves paging off if the
    /// Phase 0 gate hasn't passed yet. This is the enforcement point for "no
    /// eviction/prefetch before the predictability number exists."
    pub fn try_activate_paging(&mut self) -> bool {
        let gate = self.tracker.evaluate_gate();
        if gate.passed && self.store.is_none() {
            self.store = Some(PageStore::new(self.budget_bytes));
        }
        self.store.is_some()
    }

    pub fn is_paging_active(&self) -> bool {
        self.store.is_some()
    }

    pub fn insert_page(&mut self, page: Page) -> Result<(), String> {
        match self.store.as_mut() {
            Some(store) => {
                store.insert(page);
                Ok(())
            }
            None => Err(
                "Paging not active: Phase 0 gate has not passed (insufficient predictability \
                 samples or score below threshold). Call try_activate_paging() after the gate \
                 passes."
                    .to_string(),
            ),
        }
    }

    pub fn get_page(&self, id: PageId) -> Option<&Page> {
        self.store.as_ref().and_then(|s| s.get(id))
    }

    pub fn is_page_resident(&self, id: PageId) -> bool {
        self.store.as_ref().is_some_and(|s| s.is_resident(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jaccard::MIN_SAMPLES;

    fn make_page(id: PageId, floats: usize) -> Page {
        Page {
            id,
            data: vec![0.0; floats],
        }
    }

    #[test]
    fn test_paging_refuses_to_activate_before_gate_passes() {
        let mut engine = PagingEngine::new(1024 * 1024);
        // Only a couple samples, far below MIN_SAMPLES.
        engine.record_step(&[1, 2, 3]);
        engine.record_step(&[1, 2, 3]);
        assert!(!engine.try_activate_paging());
        assert!(!engine.is_paging_active());
        assert!(engine.insert_page(make_page(1, 10)).is_err());
    }

    #[test]
    fn test_paging_activates_after_predictable_pattern_observed() {
        let mut engine = PagingEngine::new(1024 * 1024);
        for _ in 0..(MIN_SAMPLES + 10) {
            engine.record_step(&[1, 2, 3, 4]);
        }
        assert!(engine.try_activate_paging());
        assert!(engine.is_paging_active());
        assert!(engine.insert_page(make_page(1, 10)).is_ok());
        assert!(engine.is_page_resident(1));
    }

    #[test]
    fn test_eviction_prefers_low_confidence_over_recency() {
        // Budget for exactly 2 pages of 4 f32 each (16 bytes each = 32 byte budget).
        let mut store = PageStore::new(32);
        store.insert(make_page(1, 4));
        store.insert(make_page(2, 4)); // store now full at 32 bytes

        // Page 1 gets hit repeatedly (confidence rises); page 2 never touched (decays).
        for _ in 0..5 {
            store.begin_step(&[1]);
        }

        // Inserting a 3rd page forces an eviction; page 2 (low confidence) should go,
        // not page 1, even though nothing here is measuring "recency of use" as the
        // sole signal.
        store.insert(make_page(3, 4));

        assert!(
            store.is_resident(1),
            "high-confidence page should survive eviction"
        );
        assert!(
            !store.is_resident(2),
            "low-confidence page should be evicted first"
        );
        assert!(store.is_resident(3));
    }

    #[test]
    fn test_budget_enforced_after_eviction() {
        let mut store = PageStore::new(32);
        store.insert(make_page(1, 4));
        store.insert(make_page(2, 4));
        assert_eq!(store.resident_bytes(), 32);
        store.insert(make_page(3, 4));
        // Budget of 32 bytes holds at most 2 pages of 16 bytes; after inserting a 3rd,
        // resident bytes must not exceed budget.
        assert!(store.resident_bytes() <= 32);
        assert_eq!(store.page_count(), 2);
    }

    #[test]
    fn test_single_page_larger_than_budget_still_inserts() {
        // Degenerate case: a page bigger than the whole budget. Eviction loop must
        // terminate (via the !pages.is_empty() guard) rather than looping forever.
        let mut store = PageStore::new(8);
        store.insert(make_page(1, 4)); // 16 bytes > 8 byte budget
        assert!(store.is_resident(1));
    }
}
