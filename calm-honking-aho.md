# Plan: Implement SPANDA Sparse Inference

## Context

The current `spanda-engine` implementation does not perform sparse inference on the GPU path. The planner can select a `SpandaSparse` route, but this path leads to a mocked `generate` function that does no real work, eventually falling back to the dense CPU or WGPU engine. This defeats the purpose of SPANDA.

This plan will implement the necessary components to enable true 2:4 block-sparse inference, providing a significant performance advantage by reducing the computational load.

## Plan

### 1. Correct the Sparsity Conversion in `convert.rs`

The logic in `spanda-engine/src/bin/convert.rs` for creating `BlockSparse2_4` tensors is incorrect. It prunes the 14 smallest values in a 16-element block, which is not 2:4 sparsity.

-   **Modify `spanda-engine/src/bin/convert.rs`**:
    -   Inside the loop that processes tensors, iterate through weight data in chunks of 4 elements.
    -   For each 4-element chunk, find the indices of the 2 elements with the largest absolute magnitude.
    -   Keep those 2 elements and zero out the other 2.
    -   Store the results in a format compatible with a future sparse matrix multiplication kernel. The existing `BlockSparse2_4` enum variant with `masks` and `nonzero_values` is a good starting point, but the mask should be generated from the 4-element chunks.

### 2. Implement Real Sparse Inference in `InferenceSession`

The `InferenceSession::generate` function in `spanda-engine/src/lib.rs` is currently a mock. It needs to be replaced with a real implementation that can handle `BlockSparse2_4` tensors.

-   **Modify `spanda-engine/src/lib.rs`**:
    -   Change `InferenceSession::generate` to perform a simplified transformer forward pass.
    -   For each layer, it should look up the tensor from `self.model.tensors`.
    -   If the tensor is `SpandaTensor::Dense`, perform a standard matrix multiplication.
    -   If the tensor is `SpandaTensor::BlockSparse2_4`, use the CPU-based `sparse_matvec_mul_2_4` function from `spanda-engine/src/sparse.rs` to perform the matrix multiplication.
    -   This implementation will be CPU-only for now, but it will correctly use the sparse weights.

### 3. Wire the Planner to the Real Sparse Implementation

The `generate` function in `src/inference/engine.rs` needs to be updated to correctly call the new `spanda-engine` sparse implementation when the planner chooses the `SpandaSparse` path.

-   **Modify `src/inference/engine.rs`**:
    -   In the `generate` function, find the block where `decision == crate::planner::policy::PlannerDecision::SpandaSparse`.
    -   Ensure this block correctly calls the now-functional `spanda_session.generate`.
    -   Remove any fallback logic that assumes the spanda path will fail. The `spanda_session.generate` should now return a valid result or a meaningful error.

## Verification

1.  **Run the `spanda-convert` tool** on a `.safetensors` model and verify that the output `.spanda` file contains `BlockSparse2_4` tensors for MLP and projection layers.
2.  **Add a new test** to `spanda-engine/src/lib.rs` that loads a converted sparse model and runs a single forward pass using `InferenceSession::generate`, asserting that the output has the correct shape.
3.  **Run an inference query** that is known to trigger the `SpandaSparse` path in the planner. Verify from the logs that the `spanda_session.generate` function is called and that the inference completes successfully without falling back to the dense engine.