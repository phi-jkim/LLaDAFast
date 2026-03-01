import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModel
from llada_fast.modeling.configuration_llada2_moe import LLaDA2MoeConfig
from datasets import load_dataset
from tqdm import tqdm

def distill_step1(
    teacher_model_path,
    student_config_path,
    dataset_name="HuggingFaceFW/fineweb-edu",
    batch_size=1,
    seq_len=1024,
    learning_rate=1e-4,
    num_steps=1000,
    device0="cuda:0",
    device1="cuda:1"
):
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    
    # Load Teacher (on GPU0)
    print(f"Loading teacher from {teacher_model_path} on {device0}...")
    teacher = LLaDA2MoeModel.from_pretrained(
        teacher_model_path, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device0)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Load Student (on GPU1)
    print(f"Initializing student from {teacher_model_path} on {device1}...")
    student_config = LLaDA2MoeConfig.from_pretrained(teacher_model_path)
    student_config.use_linear_attention = True
    student_config.block_size = 512 # Default
    
    student = LLaDA2MoeModel(student_config).to(torch.bfloat16).to(device1)
    
    # Initialize student weights from teacher
    student.load_state_dict(teacher.state_dict())
    
    # Freeze everything except linear attention parameters
    for name, param in student.named_parameters():
        if "linear_attention" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=learning_rate)
    
    # Load Dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, name="sample-10BT", split="train", streaming=True)
    
    def tokenize_and_pack(examples):
        tokens = tokenizer(examples["text"], truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
        return tokens

    # Simple distillation loop
    step = 0
    pbar = tqdm(total=num_steps)
    
    for batch in dataset:
        if step >= num_steps:
            break
            
        input_ids = tokenizer(batch["text"], return_tensors="pt", max_length=seq_len, truncation=True).input_ids
        input_ids = input_ids.to(device0)
        
        # Teacher Forward
        with torch.no_grad():
            teacher_outputs = teacher(input_ids, output_hidden_states=True)
            # Collect internal attention outputs? 
            # For MSE distillation, we might need to hook into the attention layers.
            # Simplified: match final hidden states for now or add hooks.
            teacher_targets = teacher_outputs.hidden_states
            # Transfer targets to GPU1
            teacher_targets = [t.to(device1) for t in teacher_targets]

        # Student Forward
        student_input_ids = input_ids.to(device1)
        student_outputs = student(student_input_ids, output_hidden_states=True)
        student_hidden_states = student_outputs.hidden_states
        
        # Loss: MSE between teacher and student hidden states
        loss = 0
        for t, s in zip(teacher_targets, student_hidden_states):
            loss += torch.nn.functional.mse_loss(s, t)
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pbar.set_description(f"Loss: {loss.item():.4f}")
        pbar.update(1)
        step += 1
        
    pbar.close()
    print("Distillation Step 1 completed.")

if __name__ == "__main__":
    # Placeholder paths
    distill_step1(
        teacher_model_path="inclusionAI/LLaDA2.1-mini",
        student_config_path="src/llada_fast/modeling/config.json"
    )
