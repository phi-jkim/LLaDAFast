# scripts/train_lora_step2.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import load_dataset
from tqdm import tqdm
import warnings
from transformers.utils import logging as hf_logging

# Silence problematic Transformers warning formatting
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

# Fix #4: Enable TF32 + “high” matmul precision (Free speed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from peft import LoraConfig, TaskType, get_peft_model
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM
from llada_fast.modeling.configuration_llada2_moe import LLaDA2MoeConfig


NOISE_LOW, NOISE_HIGH = 0.3, 0.8


def _build_block_causal_mask_proto(seq_len: int, block_size: int, device: torch.device):
    """Build a (1, 1, L, L) float32 0/1 mask prototype for Block-Parallel Decoding.

    Standard causal masks are triangles (token i sees 0..i).
    Block-causal masks are 'staircases' of squares:
    - Tokens are grouped into blocks of `block_size`.
    - Inside a block, tokens can see each other (full attention square on diagonal).
    - Tokens cannot see any future blocks.
    - This allows parallel denoising within a block while maintaining document-level causality.
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    # 1. Start with a block-level causal triangle
    blk = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    # 2. Stretch each 1/0 into a block_size x block_size square
    attend = (
        blk.repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)[:seq_len, :seq_len]
        .float()
    )
    return attend.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)


def train_lora_step2(
    model_checkpoint: str,
    dataset_name: str = "stingning/ultrachat",
    output_dir: str = "./lora_checkpoints",
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_steps: int = 10000,
    seq_len: int = 2048,
    omega_mask: float = 0.5,
    omega_edit: float = 0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = 156895
    pad_id = tokenizer.pad_token_id

    # For T2T noise: exclude special tokens
    special_ids = set(tokenizer.all_special_ids)
    vocab_ids = torch.tensor([i for i in range(tokenizer.vocab_size) if i not in special_ids], dtype=torch.long)

    config = LLaDA2MoeConfig.from_pretrained(model_checkpoint)
    config.use_linear_attention = True
    config.use_qk_norm = True # Stabilization
    block_size = int(getattr(config, "block_size", 32))

    model = LLaDA2MoeModelLM.from_pretrained(
        model_checkpoint, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # LoRA: attention + MLP projections (do NOT target phi_* via LoRA; unfreeze them directly)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "dense", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)

    # Ensure feature-map params train
    for name, param in model.named_parameters():
        if "phi_scale" in name or "phi_bias" in name:
            param.requires_grad = True

    model = model.to(device)
    model.train()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)

    attn_mask_proto = _build_block_causal_mask_proto(seq_len, block_size, device)

    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    def _to_text_and_prompt(ex):
        # UltraChat usually has "data": alternating user/assistant turns
        if "data" in ex:
            turns = ex["data"]
            if has_chat_template:
                messages = [{"role": "user" if idx % 2 == 0 else "assistant", "content": str(t)} for idx, t in enumerate(turns)]
                full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                if len(messages) > 1 and messages[-1]["role"] == "assistant":
                    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                else:
                    prompt = full
                return full, prompt
            # fallback plain join
            full = " ".join(str(u) for u in turns)
            prompt = " ".join(str(u) for u in turns[:-1]) if len(turns) > 1 else ""
            return full, prompt

        # generic dataset fallback
        return ex.get("text", ""), ""

    def collate(examples):
        pairs = [_to_text_and_prompt(ex) for ex in examples]
        texts = [t for t, _ in pairs]
        prompts = [p for _, p in pairs]

        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        B, L = enc.input_ids.shape
        real_lengths = enc.attention_mask.sum(dim=1).tolist()

        # Replace pad tokens with mask_id so pad never appears as a token (matches inference dist.)
        x_clean = enc.input_ids.clone()
        if pad_id is not None:
            x_clean[x_clean == pad_id] = mask_id
        for i in range(B):
            rl = int(real_lengths[i])
            x_clean[i, rl:] = mask_id  # hard stop beyond real length

        # Compute prompt lengths (token count without special tokens)
        prompt_lengths = []
        for p in prompts:
            if p:
                plen = len(tokenizer(p, truncation=True, max_length=seq_len, add_special_tokens=False).input_ids)
            else:
                plen = 0
            prompt_lengths.append(plen)

        x_m2t = x_clean.clone()
        labels_m2t = torch.full((B, L), -100, dtype=torch.long)
        x_t2t = x_clean.clone()
        labels_t2t = torch.full((B, L), -100, dtype=torch.long)

        for i in range(B):
            plen = int(prompt_lengths[i])
            rlen = int(real_lengths[i])

            if rlen <= plen:
                continue

            first_resp_block = plen // block_size
            last_resp_block = max(first_resp_block, (rlen - 1) // block_size)
            num_resp_blocks = last_resp_block - first_resp_block + 1
            if num_resp_blocks <= 0:
                continue

            current_block = first_resp_block + torch.randint(0, num_resp_blocks, (1,)).item()
            block_start = current_block * block_size
            block_end = min((current_block + 1) * block_size, rlen)
            resp_start = max(block_start, plen)

            if resp_start >= block_end:
                continue

            n_pos = block_end - resp_start

            # M2T (Mask-to-Token): Mask Drafting
            # This is the "standard" diffusion task. We replace tokens with [MASK] at
            # a random timestep `t`. The model learns to predict the original token,
            # which teaches it how to "draft" text from scratch in a block.
            t = NOISE_LOW + (NOISE_HIGH - NOISE_LOW) * torch.rand(1).item()
            m2t_mask = torch.rand(n_pos) < t
            x_m2t[i, resp_start:block_end][m2t_mask] = mask_id
            labels_m2t[i, resp_start:block_end][m2t_mask] = x_clean[i, resp_start:block_end][m2t_mask]

            # T2T (Token-to-Token): Corruption Editing
            # This is "self-correction" training. Instead of masking, we replace tokens
            # with RANDOM tokens from the vocab. The model must recognize these are
            # wrong and "edit" them back to the original clean token. This builds
            # robustness against drafting errors during inference.
            t_edit = NOISE_LOW + (NOISE_HIGH - NOISE_LOW) * torch.rand(1).item()
            t2t_mask = torch.rand(n_pos) < t_edit
            k = int(t2t_mask.sum().item())
            if k > 0:
                noise_tokens = vocab_ids[torch.randint(0, vocab_ids.numel(), (k,))]
                x_t2t[i, resp_start:block_end][t2t_mask] = noise_tokens
                labels_t2t[i, resp_start:block_end][t2t_mask] = x_clean[i, resp_start:block_end][t2t_mask]

        attn_mask = attn_mask_proto[:, :, :L, :L].expand(B, -1, -1, -1).to(dtype=torch.float32)
        # NOTE: forward() will convert this 0/1 4D into additive mask for SDPA path.

        return x_m2t, labels_m2t, x_t2t, labels_t2t, attn_mask

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    pbar = tqdm(total=max_steps)
    for step, (x_m2t, labels_m2t, x_t2t, labels_t2t, attn_mask) in enumerate(loader):
        if step >= max_steps:
            break

        x_m2t = x_m2t.to(device)
        labels_m2t = labels_m2t.to(device)
        x_t2t = x_t2t.to(device)
        labels_t2t = labels_t2t.to(device)
        attn_mask = attn_mask.to(device, dtype=model.dtype)

        out_m2t = model(x_m2t, attention_mask=attn_mask, labels=labels_m2t)
        loss_m2t = out_m2t.loss

        out_t2t = model(x_t2t, attention_mask=attn_mask, labels=labels_t2t)
        loss_t2t = out_t2t.loss

        loss = omega_mask * loss_m2t + omega_edit * loss_t2t
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        pbar.set_description(f"loss={loss.item():.4f} m2t={loss_m2t.item():.4f} t2t={loss_t2t.item():.4f}")
        pbar.update(1)

        if step % 500 == 0 and step > 0:
            model.save_pretrained(f"{output_dir}/step_{step}")

    pbar.close()
    model.save_pretrained(output_dir)
    print("LoRA training complete.")


if __name__ == "__main__":
    train_lora_step2(model_checkpoint="inclusionAI/LLaDA2.1-mini")