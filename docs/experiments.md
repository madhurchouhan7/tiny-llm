# Experiments & Profiling Log

## Configuration Matrix

| Experiment | Model Params | Embedding Dim | Heads | Layers | Context Length | Target Tokens | Batch Size | Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Debug** | ~0.15M | 64 | 2 | 2 | 32 | 10k | 4 | CPU |
| **Tiny Validation** | ~35.8M | 512 | 8 | 6 | 256 | 1–5M | 16 (accum 2 = 32) | CPU / GPU |
| **50M Full Run** | ~35.8M | 512 | 8 | 6 | 256 | 100–200M | 32 (accum 4 = 128) | CUDA AMP |

---

## Model Architecture Details (Option A: ~35.8M Params)

- **Vocabulary**: 16,384 BPE tokens
- **Context Window**: 256 tokens
- **Hidden Dim ($C$)**: 512
- **Heads ($H$)**: 8 ($D = C/H = 64$)
- **Layers ($N$)**: 6
- **FFN Hidden Dim**: $4 \times 512 = 2048$
- **Weight Tying**: Disabled (`weight_tying: false`) to reach 35.8M parameters

### Parameter Breakdown
- Token Embeddings: $16,384 \times 512 = 8,388,608$
- Positional Embeddings: $256 \times 512 = 131,072$
- 6 Transformer Blocks: $6 \times 3,152,384 = 18,914,304$
- Final LayerNorm: $512 \times 2 = 1,024$
- LM Head Projection: $512 \times 16,384 = 8,388,608$
- **Total Parameters**: **35,823,616** (~35.8M)

---

## Profiling & Performance Metrics

| Run | Platform | Batch Size | Precision | Throughput (tokens/sec) | GPU Memory (VRAM) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Single Batch Overfit | Local CPU | 2 | FP32 | ~15,000 tok/s | N/A |
| Tiny Pretrain (1M) | Local CPU | 16 | FP32 | Measured during run | N/A |
| Full 50M Pretrain | GPU (CUDA) | 32 | FP16 AMP | Measured on GPU | ~4–6 GB VRAM |
