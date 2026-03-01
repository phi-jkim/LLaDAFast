#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import torch

# ---- Fix the Transformers warning_once logging crash + reduce noise ----
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM

hf_logging.set_verbosity_error()

def main():
    model_path = "inclusionAI/LLaDA2.1-mini"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional: higher HF Hub rate limits / faster downloads
    # Export HF_TOKEN in your shell:  export HF_TOKEN=xxxxx
    hf_token = os.environ.get("HF_TOKEN", None)

    print(f"Loading {model_path} on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=hf_token)
    model = LLaDA2MoeModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        device_map=None,   # keep explicit .to(device) below
    ).to(device).eval()

    print("Special tokens:",
          "eos=", tok.eos_token_id,
          "pad=", tok.pad_token_id,
          "mask=", getattr(tok, "mask_token_id", None))

    prompts = [
        "Explain the concept of discrete diffusion in one sentence.",
        "Write a short poem about a cat in space.",
        "How do I optimize a linear attention module for low memory?",
    ]

    # LLaDA2.1-mini "Quality Mode" per model card
    # block_length=32, temperature=0.0, top_p=None, top_k=None
    # threshold=0.7, editing_threshold=0.5, max_post_steps=16
    gen_kwargs = dict(
        eos_early_stop=True,
        gen_length=128,
        block_length=32,
        steps=32,                 # their examples use 32
        threshold=0.7,
        editing_threshold=0.5,
        max_post_steps=16,
        temperature=0.0,
        top_p=None,
        top_k=None,
    )

    for p in prompts:
        print("\nUser Prompt:", p)

        # IMPORTANT: use chat template exactly like the model card
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                inputs=input_ids,
                **gen_kwargs,
                eos_id=tok.eos_token_id,
                mask_id=getattr(tok, "mask_token_id", None),
            )

        # Diagnostics: show token ids + raw decode so you can see special-token issues
        gen = out_ids[0]
        print("Generated ids (first 48):", gen[:48].tolist())

        raw = tok.decode(gen, skip_special_tokens=False)
        clean = tok.decode(gen, skip_special_tokens=True)

        print("Teacher Response (raw):", repr(raw[:300]))
        print("Teacher Response (clean):", repr(clean[:300]))

if __name__ == "__main__":
    main()