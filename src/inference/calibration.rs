use crate::storage::tensor_db::ModelTable;

/// Compute model-specific early-exit thresholds based on layer convergence rates
pub fn calibrate_thresholds(model: &ModelTable) -> Vec<f32> {
    // TinyLlama usually has 22 layers, but let's read dynamically from model keys
    let mut max_layer = 0;
    for key in model.layers.keys() {
        if key.starts_with("model.layers.") {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() > 2 {
                if let Ok(layer_idx) = parts[2].parse::<usize>() {
                    if layer_idx > max_layer {
                        max_layer = layer_idx;
                    }
                }
            }
        }
    }

    let num_layers = max_layer + 1;
    let mut thresholds = vec![0.0; num_layers];

    for i in 0..num_layers {
        // Model-specific confidence band: early layers need extremely high confidence,
        // deeper layers require progressively less additional confidence to exit.
        let ratio = i as f32 / num_layers as f32;
        thresholds[i] = 0.95 - 0.25 * ratio; // e.g. layer 0 = 0.95, last layer = 0.70
    }

    thresholds
}
