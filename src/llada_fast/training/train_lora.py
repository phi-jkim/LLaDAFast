import os
import torch
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModel
from llada_fast.modeling.configuration_llada2_moe import LLaDA2MoeConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def train_lora_step2(
    model_checkpoint,
    dataset_name="stingning/ultrachat",
    output_dir="./lora_checkpoints",
    batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    seq_len=2048
):
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Load Linearized Model (Student)
    print(f"Loading linearized model from {model_checkpoint}...")
    config = LLaDA2MoeConfig.from_pretrained(model_checkpoint)
    config.rope_scaling = None
    config.use_linear_attention = True
    
    model = LLaDA2MoeModel.from_pretrained(
        model_checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Configure LoRA
    # Targeting the qkv projection, dense projection, and linear attention params
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value", "dense", "phi_scale", "phi_bias"], 
        lora_dropout=0.05,
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load Dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # --- LLaDA Denoising Setup ---
    def llada_masking_collator(examples):
        """
        Custom collator for LLaDA denoising.
        Applies a random masking ratio t ~ Uniform(0, 1) to each sequence.
        """
        texts = [ex["text"] for ex in examples]
        inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
        
        input_ids = inputs.input_ids
        labels = input_ids.clone()
        
        # Masking schedule: Pick a random ratio for each sequence in batch
        # LLaDA uses a unified masking strategy during training.
        batch_size, length = input_ids.shape
        mask_ratio = torch.rand(batch_size, 1)
        
        # Create mask: 1 for mask, 0 for keep
        rand_mask = torch.rand(batch_size, length)
        
        # Never hardcode masks! Fetch mask_id from tokenizer or config.
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            mask_token_id = tokenizer.mask_token_id
        else:
            mask_token_id = getattr(config, "mask_token_id", 156895)
        
        is_masked = rand_mask < mask_ratio
        input_ids[is_masked] = mask_token_id
        
        # Labels should be -100 for non-masked tokens (don't train on them)
        labels[~is_masked] = -100

        # LLaDA 2.1 requires a 4D block diffusion attention mask
        block_size = config.block_size if hasattr(config, "block_size") else 32
        num_blocks = (length + block_size - 1) // block_size
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_size, dim=0)
            .repeat_interleave(block_size, dim=1)[:length, :length]
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
        ).to(torch.bfloat16)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": block_diffusion_attention_mask
        }

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        bf16=True,
        save_steps=100,
        streaming=True,
    )

    # Simplified Trainer (Assuming standard Trainer handles the custom forward in LLaDA2MoeModelLM)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=llada_masking_collator,
    )

    print("Starting Step 2: LoRA Training...")
    trainer.train()
    print("LoRA Training complete.")

if __name__ == "__main__":
    train_lora_step2(
        model_checkpoint="inclusionAI/LLaDA2.1-mini"
    )
