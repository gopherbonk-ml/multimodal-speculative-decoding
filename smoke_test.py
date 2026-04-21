"""
Smoke test — no real model weights needed.

Tests:
  [1] MockTarget construction & all property accessors
  [2] All 5 drafter architectures: construction + full forward pass
  [3] DistillationLoss + Eagle3Loss
  [4] Manual training step (forward + backward + optimizer step) for each arch
  [5] SpeculativeDecoder.generate()
  [6] Eagle3SpeculativeDecoder.generate()

Run:
    python smoke_test.py
    python smoke_test.py --device mps   # explicit MPS
    python smoke_test.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# ─────────────────────────────────────────────────────────────────────────────
# Tiny dimensions — keeps the test fast (sub-second per arch)
# ─────────────────────────────────────────────────────────────────────────────
T_HIDDEN   = 64    # target LLM hidden size
T_VIT      = 48    # target raw ViT hidden size  (before merger)
VOCAB      = 512   # shared vocabulary size
D_HIDDEN   = 32    # drafter LLM hidden size
D_INTER    = 64
D_LAYERS   = 2
D_HEADS    = 2
D_KV_HEADS = 1
SEQ        = 10    # sequence length
IMG_TOK    = 4     # image tokens injected into the sequence
IMAGE_TOK_ID = 200 # pseudo image-token id  (<200 is the special slot)
EOS_ID     = 1


def _tiny_qwen2_config(vocab: int = VOCAB) -> Qwen2Config:
    return Qwen2Config(
        vocab_size=vocab,
        hidden_size=D_HIDDEN,
        num_hidden_layers=D_LAYERS,
        num_attention_heads=D_HEADS,
        num_key_value_heads=D_KV_HEADS,
        intermediate_size=D_INTER,
        max_position_embeddings=512,
        rope_theta=10_000.0,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_sliding_window=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MockVisual — mimics Qwen2-VL ViT+merger
# ─────────────────────────────────────────────────────────────────────────────

class _MockNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = nn.LayerNorm(T_VIT)

    def forward(self, x, **kw):
        return x


class MockVisual(nn.Module):
    """Tiny stub matching the ViT interface used by Arch1–5."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_MockNorm()])
        self._merger = nn.Linear(T_VIT, T_HIDDEN, bias=False)
        self._patch_proj = nn.Linear(T_VIT, T_VIT, bias=False)

    def forward(self, pixel_values: torch.Tensor, grid_thw=None) -> torch.Tensor:
        """Returns (N_patches, T_HIDDEN) merged visual features."""
        B = pixel_values.shape[0] if pixel_values.ndim > 1 else 1
        patches = IMG_TOK * B
        raw = torch.zeros(patches, T_VIT, device=pixel_values.device, dtype=pixel_values.dtype)
        return self._merger(raw)  # (N, T_HIDDEN)

    def patch_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0] if pixel_values.ndim > 1 else 1
        patches = IMG_TOK * B
        return torch.zeros(patches, T_VIT, device=pixel_values.device, dtype=pixel_values.dtype)

    def rot_pos_emb(self, grid_thw):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MockTarget — exposes all TargetModel properties / methods
# ─────────────────────────────────────────────────────────────────────────────

class MockTarget(nn.Module):
    """
    Thin mock around a tiny Qwen2 LLM + MockVisual.
    Exposes the same interface as models.target.TargetModel.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self._visual = MockVisual()
        llm_cfg = Qwen2Config(
            vocab_size=VOCAB,
            hidden_size=T_HIDDEN,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=T_HIDDEN * 2,
            max_position_embeddings=512,
            rope_theta=10_000.0,
            tie_word_embeddings=False,
            use_sliding_window=False,
        )
        self._llm = Qwen2ForCausalLM(llm_cfg)
        self._device = device
        self.to(device)

    # ----- TargetModel property interface -----

    @property
    def visual(self):
        return self._visual

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self._llm.model.embed_tokens

    @property
    def lm_hidden_size(self) -> int:
        return T_HIDDEN

    @property
    def vocab_size(self) -> int:
        return VOCAB

    @property
    def image_token_id(self) -> int:
        return IMAGE_TOK_ID

    @property
    def vit_hidden_size(self) -> int:
        return T_VIT

    @torch.no_grad()
    def get_visual_features(self, pixel_values, image_grid_thw=None):
        return self._visual(pixel_values, grid_thw=image_grid_thw)

    @torch.no_grad()
    def get_raw_vit_features(self, pixel_values, image_grid_thw=None):
        B = pixel_values.shape[0] if pixel_values.ndim > 1 else 1
        return torch.zeros(
            IMG_TOK * B, T_VIT,
            device=pixel_values.device, dtype=pixel_values.dtype,
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_hidden_states=False,
        **kwargs,
    ):
        return self._llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fake_batch(device: torch.device):
    """Build a minimal multimodal batch with IMG_TOK image tokens."""
    ids = torch.randint(2, VOCAB - 1, (1, SEQ), device=device)
    # inject image token ids at the first IMG_TOK positions
    ids[0, :IMG_TOK] = IMAGE_TOK_ID
    mask = torch.ones(1, SEQ, dtype=torch.long, device=device)
    labels = ids.clone()
    labels[0, :IMG_TOK] = -100  # ignore image positions in CE loss
    pixel_values = torch.rand(1, 3, 16, 16, device=device)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }


def ok(msg: str):
    print(f"  \033[32m✓\033[0m  {msg}")


def fail(msg: str, exc: Exception):
    print(f"  \033[31m✗\033[0m  {msg}")
    traceback.print_exc()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Test suites
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_target(device):
    print("\n[1] MockTarget")
    target = MockTarget(device)
    assert target.visual is not None
    ok("visual property")
    assert isinstance(target.embed_tokens, nn.Embedding)
    ok("embed_tokens property")
    assert target.lm_hidden_size == T_HIDDEN
    ok(f"lm_hidden_size = {T_HIDDEN}")
    assert target.vocab_size == VOCAB
    ok(f"vocab_size = {VOCAB}")
    assert target.image_token_id == IMAGE_TOK_ID
    ok(f"image_token_id = {IMAGE_TOK_ID}")
    assert target.vit_hidden_size == T_VIT
    ok(f"vit_hidden_size = {T_VIT}")

    pv = torch.rand(1, 3, 16, 16, device=device)
    feats = target.get_visual_features(pv)
    assert feats.shape == (IMG_TOK, T_HIDDEN), feats.shape
    ok(f"get_visual_features → {feats.shape}")

    raw = target.get_raw_vit_features(pv)
    assert raw.shape == (IMG_TOK, T_VIT), raw.shape
    ok(f"get_raw_vit_features → {raw.shape}")

    batch = _fake_batch(device)
    out = target(input_ids=batch["input_ids"], use_cache=False)
    assert out.logits.shape == (1, SEQ, VOCAB)
    ok(f"forward → logits {out.logits.shape}")

    out_h = target(input_ids=batch["input_ids"], use_cache=False, output_hidden_states=True)
    assert out_h.hidden_states is not None
    ok(f"forward w/ hidden_states → {len(out_h.hidden_states)} layers, last={out_h.hidden_states[-1].shape}")

    return target


def test_all_drafters(target, device):
    print("\n[2] Drafter architectures")
    from models.drafters.arch1 import Arch1Drafter
    from models.drafters.arch2 import Arch2Drafter
    from models.drafters.arch3 import Arch3Drafter
    from models.drafters.arch4 import Arch4Drafter
    from models.drafters.arch5_eagle3 import Arch5Eagle3Drafter

    tiny = _tiny_qwen2_config()
    batch = _fake_batch(device)
    results = {}

    # Arch3 uses SmallViT with 14×14 patches and fixed pos_embed.
    # 28×28 image → 2×2 = 4 patches, matching IMG_TOK=4 image token slots in input_ids.
    batch_arch3 = dict(batch)
    batch_arch3["pixel_values"] = torch.rand(1, 3, 28, 28, device=device)

    for name, cls, kwargs, b in [
        ("Arch1",      Arch1Drafter,      {"drafter_config": tiny}, batch),
        ("Arch2",      Arch2Drafter,      {"drafter_config": tiny}, batch),
        # img_size=28, patch_size=14 → 2×2=4 patches, matches IMG_TOK=4
        ("Arch3",      Arch3Drafter,      {"drafter_config": tiny, "img_size": 28,
                                            "patch_size": 14, "vit_embed_dim": 32,
                                            "vit_depth": 2, "vit_num_heads": 2}, batch_arch3),
        ("Arch4",      Arch4Drafter,      {"drafter_config": tiny}, batch),
        ("Arch5/Eagle3", Arch5Eagle3Drafter, {"drafter_config": tiny}, batch),
    ]:
        try:
            drafter = cls(target=target, **kwargs).to(device)
            n = sum(p.numel() for p in drafter.parameters() if p.requires_grad)

            if name == "Arch5/Eagle3":
                out = drafter(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                    pixel_values=b["pixel_values"],
                    use_cache=False,
                    output_hidden_states=True,
                )
            else:
                out = drafter(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                    pixel_values=b["pixel_values"],
                    use_cache=False,
                )

            assert out.logits.shape == (1, SEQ, VOCAB), out.logits.shape
            ok(f"{name}: {n/1e3:.0f}K params, logits {out.logits.shape}")
            results[name] = drafter
        except Exception as e:
            fail(f"{name} failed", e)

    return results


def test_losses(target, drafters, device):
    print("\n[3] Loss functions")
    from distillation.losses import DistillationLoss
    from distillation.eagle3_losses import Eagle3Loss

    batch = _fake_batch(device)

    # target logits
    with torch.no_grad():
        t_out = target(input_ids=batch["input_ids"], use_cache=False, output_hidden_states=True)
    t_logits = t_out.logits.detach()
    t_hidden = t_out.hidden_states[-1].detach()

    # DistillationLoss with Arch1
    if "Arch1" in drafters:
        drafter = drafters["Arch1"]
        d_out = drafter(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            use_cache=False,
        )
        loss_fn = DistillationLoss(temperature=2.0, alpha=0.9)
        ld = loss_fn(
            drafter_logits=d_out.logits,
            target_logits=t_logits,
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        )
        assert ld["loss"].item() > 0
        ok(f"DistillationLoss: total={ld['loss'].item():.4f}  "
           f"distill={ld['distill_loss'].item():.4f}  task={ld['task_loss'].item():.4f}")

    # Eagle3Loss with Arch5
    if "Arch5/Eagle3" in drafters:
        from models.drafters.arch5_eagle3 import Arch5Eagle3Drafter
        drafter: Arch5Eagle3Drafter = drafters["Arch5/Eagle3"]
        proj_feats = drafter.project_target_features(t_hidden)
        d_out = drafter(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            use_cache=False,
            output_hidden_states=True,
        )
        eagle_fn = Eagle3Loss(temperature=2.0, alpha=0.7, beta=0.2)
        le = eagle_fn(
            drafter_logits=d_out.logits,
            target_logits=t_logits,
            drafter_hidden=d_out.hidden_states[-1],
            target_features_projected=proj_feats,
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        )
        assert le["loss"].item() > 0
        ok(f"Eagle3Loss:        total={le['loss'].item():.4f}  "
           f"distill={le['distill_loss'].item():.4f}  "
           f"feat={le['feature_loss'].item():.4f}  "
           f"task={le['task_loss'].item():.4f}")


def test_training_step(target, drafters, device):
    print("\n[4] Training step (forward + backward + optimizer)")
    from distillation.losses import DistillationLoss
    from distillation.eagle3_losses import Eagle3Loss

    batch = _fake_batch(device)
    batch_arch3 = dict(batch)
    batch_arch3["pixel_values"] = torch.rand(1, 3, 28, 28, device=device)

    with torch.no_grad():
        t_out = target(input_ids=batch["input_ids"], use_cache=False, output_hidden_states=True)
    t_logits = t_out.logits.detach()
    t_hidden = t_out.hidden_states[-1].detach()

    for name, drafter in drafters.items():
        try:
            b = batch_arch3 if name == "Arch3" else batch
            opt = torch.optim.AdamW(
                [p for p in drafter.parameters() if p.requires_grad], lr=1e-4
            )
            opt.zero_grad()

            if name == "Arch5/Eagle3":
                from models.drafters.arch5_eagle3 import Arch5Eagle3Drafter
                drafter: Arch5Eagle3Drafter
                B, L, _ = t_hidden.shape
                h_shifted = torch.cat([
                    torch.zeros(B, 1, T_HIDDEN, device=device),
                    t_hidden[:, :-1, :],
                ], dim=1)
                proj_shifted = drafter.project_target_features(h_shifted)
                proj_align = drafter.project_target_features(t_hidden)

                d_out = drafter(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                    pixel_values=b["pixel_values"],
                    projected_features=proj_shifted,
                    use_cache=False,
                    output_hidden_states=True,
                )
                loss_fn = Eagle3Loss(temperature=2.0, alpha=0.7, beta=0.2)
                ld = loss_fn(
                    drafter_logits=d_out.logits,
                    target_logits=t_logits,
                    drafter_hidden=d_out.hidden_states[-1],
                    target_features_projected=proj_align,
                    labels=b["labels"],
                    attention_mask=b["attention_mask"],
                )
            else:
                d_out = drafter(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                    pixel_values=b["pixel_values"],
                    use_cache=False,
                )
                loss_fn = DistillationLoss(temperature=2.0, alpha=0.9)
                ld = loss_fn(
                    drafter_logits=d_out.logits,
                    target_logits=t_logits,
                    labels=b["labels"],
                    attention_mask=b["attention_mask"],
                )

            ld["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in drafter.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            ok(f"{name}: loss={ld['loss'].item():.4f}  grad_norm={grad_norm:.4f}")
        except Exception as e:
            fail(f"{name} backward failed", e)


def test_speculative_decoding(target, drafters, device):
    print("\n[5] SpeculativeDecoder.generate()")
    from inference.speculative_decoding import SpeculativeDecoder, SpeculativeDecodingConfig

    if "Arch1" not in drafters:
        print("  (Arch1 unavailable, skipping)")
        return

    drafter = drafters["Arch1"]
    drafter.eval()
    target.eval()

    cfg = SpeculativeDecodingConfig(
        gamma=3,
        max_new_tokens=6,
        eos_token_id=EOS_ID,
        do_sample=False,  # greedy for reproducibility
    )
    decoder = SpeculativeDecoder(target, drafter, cfg)

    input_ids = torch.randint(2, VOCAB - 1, (1, 4), device=device)
    try:
        with torch.no_grad():
            out_ids, stats = decoder.generate(input_ids=input_ids)
        new_toks = out_ids.shape[1] - input_ids.shape[1]
        ok(f"Generated {new_toks} new tokens | {stats}")
    except Exception as e:
        fail("SpeculativeDecoder.generate() failed", e)


def test_eagle3_speculative_decoding(target, drafters, device):
    print("\n[6] Eagle3SpeculativeDecoder.generate()")
    from inference.eagle3_speculative_decoding import Eagle3SpeculativeDecoder
    from inference.speculative_decoding import SpeculativeDecodingConfig

    if "Arch5/Eagle3" not in drafters:
        print("  (Arch5 unavailable, skipping)")
        return

    drafter = drafters["Arch5/Eagle3"]
    drafter.eval()
    target.eval()

    cfg = SpeculativeDecodingConfig(
        gamma=3,
        max_new_tokens=6,
        eos_token_id=EOS_ID,
        do_sample=False,
    )
    decoder = Eagle3SpeculativeDecoder(target, drafter, cfg)

    input_ids = torch.randint(2, VOCAB - 1, (1, 4), device=device)
    try:
        with torch.no_grad():
            out_ids, stats = decoder.generate(input_ids=input_ids)
        new_toks = out_ids.shape[1] - input_ids.shape[1]
        ok(f"Generated {new_toks} new tokens | {stats}")
    except Exception as e:
        fail("Eagle3SpeculativeDecoder.generate() failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None,
                        help="cpu | mps | cuda (default: mps if available, else cpu)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nSmoke test on device: {device}")
    print("=" * 55)

    target = test_mock_target(device)
    drafters = test_all_drafters(target, device)
    test_losses(target, drafters, device)
    test_training_step(target, drafters, device)
    test_speculative_decoding(target, drafters, device)
    test_eagle3_speculative_decoding(target, drafters, device)

    print("\n" + "=" * 55)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
