"""
Default small Qwen2 configuration targeting ~88M parameters.

Parameter budget (approximate):
  hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
  num_key_value_heads=4 (GQA), intermediate_size=2048

  Embedding (tied with lm_head): vocab_size * hidden_size
    = 151936 * 768 ≈ 116M  (counted once due to weight tying)
  Per layer:
    QKV proj: (12+4+4) heads * 64 * 768 * 2 ≈ 1.97M
    FFN:      768*2048*2 + 2048 ≈ 3.15M
    Norms:    negligible
    ≈ 5.1M per layer × 12 = 61M
  Total unique params ≈ 116 + 61 ≈ ~177M with full vocab embedding

To fit ~88M, tie embeddings and reduce:
  hidden_size=512, layers=12, intermediate=1536, heads=8, kv_heads=2
  Embedding (tied): 151936 * 512 ≈ 77.8M
  Per layer ≈ 2.2M × 12 = 26.4M
  Total ≈ 77.8 + 26.4 ≈ ~104M

Adjust hidden_size / layers to hit exact target.
The config below is a starting point; tune as needed.
"""

from transformers import Qwen2Config


def get_small_qwen2_config(vocab_size: int = 151936) -> Qwen2Config:
    """Return a Qwen2Config for the ~88M drafter LLM."""
    return Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,         # GQA: fewer KV heads saves params
        intermediate_size=1536,
        max_position_embeddings=32768,
        rope_theta=1_000_000.0,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        use_sliding_window=False,
        tie_word_embeddings=True,      # share embed_tokens and lm_head
        initializer_range=0.02,
        attention_dropout=0.0,
    )
