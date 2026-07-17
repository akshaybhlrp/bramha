// STORAGE_ORCHESTRATION_EXAMPLE.rs
//
// An executable walkthrough demonstrating the end-to-end integration of the
// Bramha database-native storage layer: Model Manifests, Content-Addressed
// Storage (CAS) for deduplication, and Multi-Tier Storage routing.
//
// This file is a runnable integration script demonstrating model ingestion,
// cross-model deduplication, and tiered memory access/promotion flows.

use std::fs;

use bramha::storage::content_addressing::ContentAddressedStorage;
use bramha::storage::multi_tier::{MultiTierStorage, TierConfig};
use bramha::storage::storage_manifest::{LayerMetadata, ModelManifest, StorageTier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Bramha Storage Orchestration Walkthrough...");

    // 1. Setup Temporary Directories for the Simulation
    let base_dir = std::env::current_dir()?.join("storage_simulation");
    fs::create_dir_all(&base_dir)?;

    let cas_dir = base_dir.join("cas");
    let hot_dir = base_dir.join("hot_tier");
    let warm_dir = base_dir.join("warm_tier");
    let cold_dir = base_dir.join("cold_tier");

    // Initialize CAS
    println!("\n📦 Initializing Content-Addressed Storage...");
    let cas = ContentAddressedStorage::new(cas_dir)?;

    // Initialize Multi-Tier Storage with custom small sizes for simulation
    println!("🟡 Initializing Multi-Tier Storage Manager...");
    let config = TierConfig {
        hot_max_bytes: 10 * 1024 * 1024, // 10 MB limit for DRAM simulation
        warm_max_bytes: 50 * 1024 * 1024, // 50 MB limit for SSD simulation
        promotion_threshold: 2, // Promote on second access
        ..TierConfig::default()
    };

    let mut multi_tier =
        MultiTierStorage::new(config, hot_dir.clone(), warm_dir.clone(), cold_dir.clone())?;

    // 2. Simulate Ingesting Model A (Base LLaMA Model)
    println!("\n📥 Step 1: Ingesting Model A (Base Model - 3 layers)...");
    let mut manifest_a = ModelManifest::new(
        "llama-7b-base".to_string(),
        "f32".to_string(),
        3,
        "llama".to_string(),
        4096,
        32,
        32,
        base_dir.join("llama-base"),
    );

    // Mock layer weight data (1 million elements = 4MB)
    let weight_size_elements = 1_000_000;
    let mock_weights_shared = vec![0.5f32; weight_size_elements]; // Shared layer (e.g. embed)
    let mock_weights_layer1 = vec![0.123f32; weight_size_elements];
    let mock_weights_layer2 = vec![0.456f32; weight_size_elements];

    let layers_a = vec![
        (
            "model.embed_tokens.weight",
            &mock_weights_shared,
            StorageTier::Critical,
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            &mock_weights_layer1,
            StorageTier::Important,
        ),
        (
            "model.layers.1.self_attn.q_proj.weight",
            &mock_weights_layer2,
            StorageTier::Robust,
        ),
    ];

    for (layer_name, data, tier) in layers_a {
        // Store layer in CAS (dedup)
        let (stored, savings) = cas.store_tensor("llama-7b-base", layer_name, data)?;
        println!(
            "   Saved layer {} to CAS (stored: {} bytes, savings: {} bytes)",
            layer_name, stored, savings
        );

        // Register in Multi-Tier Storage
        let layer_path = cas.data_dir.join("chunk_store.bin");
        multi_tier.register_layer(
            layer_name.to_string(),
            (data.len() * 4) as u64,
            tier,
            layer_path,
        )?;

        // Add to Manifest
        let mut layer_meta = LayerMetadata::new(layer_name.to_string(), vec![weight_size_elements]);
        layer_meta.storage_tier = tier;
        layer_meta.stored_bytes = stored;
        manifest_a.add_layer(layer_meta);
    }
    cas.save_index()?;

    println!("\n--- Model A Ingestion Complete. Manifest Report: ---");
    manifest_a.report();

    // 3. Simulate Ingesting Model B (Fine-Tuned Variant - shares embed layer)
    println!("\n📥 Step 2: Ingesting Model B (Fine-Tuned Variant - shares embed tokens layer)...");
    let mut manifest_b = ModelManifest::new(
        "llama-7b-tuned".to_string(),
        "f32".to_string(),
        3,
        "llama".to_string(),
        4096,
        32,
        32,
        base_dir.join("llama-tuned"),
    );

    // Fine-tuned model has different layer 1 and 2, but identical embed layer
    let mock_weights_layer1_tuned = vec![0.789f32; weight_size_elements];
    let mock_weights_layer2_tuned = vec![0.987f32; weight_size_elements];

    let layers_b = vec![
        (
            "model.embed_tokens.weight",
            &mock_weights_shared,
            StorageTier::Critical,
        ), // Identical to A!
        (
            "model.layers.0.self_attn.q_proj.weight",
            &mock_weights_layer1_tuned,
            StorageTier::Important,
        ),
        (
            "model.layers.1.self_attn.q_proj.weight",
            &mock_weights_layer2_tuned,
            StorageTier::Robust,
        ),
    ];

    for (layer_name, data, tier) in layers_b {
        // Store layer in CAS (dedup should kick in for model.embed_tokens.weight)
        let (stored, savings) = cas.store_tensor("llama-7b-tuned", layer_name, data)?;
        println!(
            "   Saved layer {} to CAS (stored: {} bytes, savings: {} bytes)",
            layer_name, stored, savings
        );

        // Register in Multi-Tier Storage
        let layer_path = cas.data_dir.join("chunk_store.bin");
        // We register with a slightly different layer name to distinguish models in multi-tier routing
        let unique_layer_id = format!("{}:{}", "llama-7b-tuned", layer_name);
        multi_tier.register_layer(unique_layer_id, (data.len() * 4) as u64, tier, layer_path)?;

        // Add to Manifest
        let mut layer_meta = LayerMetadata::new(layer_name.to_string(), vec![weight_size_elements]);
        layer_meta.storage_tier = tier;
        layer_meta.stored_bytes = stored;
        manifest_b.add_layer(layer_meta);
    }
    cas.save_index()?;

    println!("\n--- Model B Ingestion Complete. Manifest Report: ---");
    manifest_b.report();

    // Show Deduplication Efficiency
    println!("\n📊 Cross-Model Deduplication Check:");
    cas.report();

    // 4. Simulate Inference Access Patterns & Tier Promotions
    println!("\n⚡ Step 3: Simulating Inference Access & Dynamic Tiering...");

    // Layer 1 of Model A starts in SSD (Important). Let's access it.
    let target_layer = "model.layers.0.self_attn.q_proj.weight";
    println!("   Access 1: Loading {}...", target_layer);
    multi_tier.access_layer(target_layer)?;

    // Let's access it a second time. This should trigger a promotion to DRAM (Critical / Hot).
    println!("   Access 2: Loading {}...", target_layer);
    multi_tier.access_layer(target_layer)?;

    // Accessing a Cold layer (Robust)
    let cold_layer = "model.layers.1.self_attn.q_proj.weight";
    println!("   Access 3: Loading cold {}...", cold_layer);
    multi_tier.access_layer(cold_layer)?;

    // Show Multi-Tier Storage report to prove the promotion and access hits
    println!("\n--- Multi-Tier Memory Routing Report: ---");
    multi_tier.report();

    // 5. Cleanup Simulation
    println!("\n🧹 Cleaning up simulation files...");
    fs::remove_dir_all(&base_dir)?;
    println!("✅ Storage simulation cleaned up successfully.");

    Ok(())
}
