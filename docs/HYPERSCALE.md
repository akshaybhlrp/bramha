# HYPERSCALE.md — Bramha Distributed Scale

> **DO NOT OPEN THIS DOCUMENT UNTIL v1.0 ENTRY GATE IS PASSED.**
> For what we are building now, see `EXECUTION_ROADMAP.md`.
> For the long-term vision, see `VISION.md`.

---

## Entry Gate

vNext (distributed scale) begins only when **ALL** of the following are true:

- [ ] v1.0 API is frozen for 30 days with zero breaking changes
- [ ] Single-node throughput validated at ≥10 req/s sustained on 4GB GPU target
- [ ] 7-day stress test: zero memory leaks, zero data corruption, P99 latency stable
- [ ] Distributed design doc approved (not implemented — just designed)
- [ ] Team has 2+ engineers with distributed systems experience, OR single engineer has completed v1.0 solo

**If any item is unchecked:** Extend v1.0 stabilization. Do not read past this point.

---

## Sprint 13 — Hyper-Scale Future (Blocked)

### 13.1 Distributed Control Plane
- [ ] Cluster membership and health checking
- [ ] Coordinator election
- [ ] Configuration propagation

### 13.2 Data Plane Workers
- [ ] Remote execution protocol
- [ ] Activation streaming between nodes
- [ ] Expert placement and migration

### 13.3 Shard Replication
- [ ] Model shard replication across nodes
- [ ] KV cache replication for hot sessions
- [ ] Consistency model (eventual, configurable)

### 13.4 Rebalancing
- [ ] Automatic shard rebalancing based on load
- [ ] Thermal-aware placement
- [ ] Network topology-aware routing

### 13.5 Remote Execution
- [ ] Offload inference to remote workers
- [ ] Cache-aware routing (route to node with prefix in KV cache)
- [ ] Fallback to local execution if remote fails

### 13.6 Multi-Node Routing
- [ ] Cluster-aware planner decisions
- [ ] Cost model includes network latency
- [ ] Cross-node pipeline execution

### 13.7 Mixed-Hardware Node Support
- [ ] CPU-only nodes
- [ ] CPU + iGPU nodes
- [ ] CPU + dGPU nodes
- [ ] Heterogeneous scheduling across cluster

---

## DS4 Distributed Lessons (Reference Only)

From ds4 distributed inference:

- **Prefill pipelining works**: Two MacBooks on Thunderbolt 5 achieve 1.38x–1.85x speedup on long prefills by pipelining chunks.
- **Generation is always slower distributed**: 19.4% loss on two-Mac setup due to autoregressive serialization. Expect this.
- **Layer splitting is the model**: Each node owns a contiguous layer range. Activations flow worker-to-worker. Final worker owns output head.
- **Fast networking is mandatory**: Thunderbolt 5 (0.45ms ping) → 25 t/s. WiFi (77ms) → 10.7 t/s. Internet/VPN (152ms) → 3.6 t/s.
- **8-bit activation transport is experimental**: ds4 found reduction to 8-bit did not improve performance significantly. Start with 16-bit.
- **No encryption or auth yet**: ds4 protocol is plain TCP on trusted networks. Do not expose to untrusted networks.

**Bramha should adopt:**
- Layer-range splitting with worker-to-worker activation flow
- Prefill pipelining for long contexts
- 16-bit activation transport as default
- Plain TCP on trusted networks (with auth added later)

**Bramha should NOT adopt:**
- Expecting distributed generation to be faster than single-node
- Running distributed over slow links for interactive use

---

## Architecture Notes (Pre-Design Only)

- **Node classes**: CPU-only, CPU+iGPU, CPU+dGPU, mixed heterogeneous
- **Communication**: Plain TCP for control and data. gRPC or custom protocol TBD.
- **KV cache**: Distributed prefix cache — route to node that owns the prefix.
- **Model placement**: Expert placement across nodes based on memory and compute capacity.
- **Planner**: Cluster-aware cost model. Local vs remote execution decision per layer.
- **Failure model**: Worker disconnect → remove from route, replay prefix to rebuild KV on replacement worker.

---

## Condition

This sprint exists so the following items are never forgotten or skipped later. They are not scheduled. They are not resourced. They are not started.

**The only way to move this from `HYPERSCALE.md` to `EXECUTION_ROADMAP.md` is to pass the Entry Gate at the top of this document.**
