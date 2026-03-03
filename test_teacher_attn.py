import torch
from transformers import AutoTokenizer
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM
from llada_fast.training.distill.attn_viz import _capture_teacher_attn, _mask_to_additive

device = torch.device('cuda:0')
print("Loading model...")
teacher = LLaDA2MoeModelLM.from_pretrained("inclusionAI/LLaDA2.1-mini", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.1-mini", trust_remote_code=True)

input_ids = tokenizer("Test sentence", return_tensors="pt").input_ids
B, L = input_ids.shape
pad_mask = torch.ones((B, L), dtype=torch.bool)
mask_0_1 = torch.ones((B, 1, L, L), dtype=torch.float32)
attn_mask_additive = _mask_to_additive(mask_0_1)

# Hack to print the out tuple 
def _debug_hook(module, inp, out):
    shapes = []
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            shapes.append(f"item {i}: {type(o)} {getattr(o, 'shape', 'no shape')}")
    else:
        shapes.append(f"{type(out)} {getattr(out, 'shape', 'no shape')}")
    print(f"DEBUG LAYER HOOK: {shapes}")
    
teacher.model.layers[0].attention.register_forward_hook(_debug_hook)

print("Calling _capture_teacher_attn...")
res = _capture_teacher_attn(teacher, input_ids, attn_mask_additive, pad_mask, [0], device)
print(f"Captured: {list(res.keys())}")
