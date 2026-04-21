"""
Demo script: run speculative decoding with a trained drafter.

Usage:
    python inference/run_inference.py \
        --target Qwen/Qwen2-VL-2B-Instruct \
        --drafter checkpoints/arch1/final/drafter.pt \
        --arch arch1 \
        --image path/to/image.jpg \
        --prompt "Describe the image in detail." \
        --gamma 5 \
        --max_new_tokens 200
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image
from transformers import Qwen2VLProcessor

from models import TargetModel, Arch1Drafter, Arch2Drafter, Arch3Drafter, Arch4Drafter, Arch5Eagle3Drafter
from inference.speculative_decoding import SpeculativeDecoder, SpeculativeDecodingConfig
from inference.eagle3_speculative_decoding import Eagle3SpeculativeDecoder

ARCH_MAP = {
    "arch1": Arch1Drafter,
    "arch2": Arch2Drafter,
    "arch3": Arch3Drafter,
    "arch4": Arch4Drafter,
    "eagle3": Arch5Eagle3Drafter,
}

EAGLE3_ARCHS = {"eagle3"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--drafter", required=True, help="Path to drafter checkpoint (.pt)")
    parser.add_argument("--arch", required=True, choices=list(ARCH_MAP.keys()))
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # ---- Load target ----
    print(f"Loading target: {args.target}")
    target = TargetModel(args.target, torch_dtype=dtype).to(device)
    target.eval()

    processor = Qwen2VLProcessor.from_pretrained(args.target)

    # ---- Load drafter ----
    print(f"Loading drafter ({args.arch}): {args.drafter}")
    drafter_cls = ARCH_MAP[args.arch]
    drafter = drafter_cls(target=target)
    state = torch.load(args.drafter, map_location=device)
    drafter.load_state_dict(state)
    drafter = drafter.to(device).eval()

    # ---- Prepare input ----
    messages = []
    content = []
    if args.image:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": args.prompt})
    messages.append({"role": "user", "content": content})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image = None
    if args.image:
        image = Image.open(args.image).convert("RGB")

    inputs = processor(
        text=[text],
        images=[image] if image else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # ---- Speculative decoding ----
    cfg = SpeculativeDecodingConfig(
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    if args.arch in EAGLE3_ARCHS:
        decoder = Eagle3SpeculativeDecoder(target, drafter, cfg)
    else:
        decoder = SpeculativeDecoder(target, drafter, cfg)

    print("\n--- Generating with speculative decoding ---")
    with torch.no_grad():
        output_ids, stats = decoder.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )

    # Decode only the new tokens
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.decode(new_ids[0], skip_special_tokens=True)

    print(f"\nResponse:\n{response}")
    print(f"\nStats:\n{stats}")

    # ---- Baseline: target-only autoregressive ----
    print("\n--- Baseline: target-only autoregressive ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        base_out = target.model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
        )
    t1 = time.perf_counter()
    n_new = base_out.shape[1] - inputs["input_ids"].shape[1]
    print(f"Target-only: {n_new} tokens in {t1-t0:.2f}s ({n_new/(t1-t0):.1f} tok/s)")


if __name__ == "__main__":
    main()
