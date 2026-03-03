import sys
import torch
from transformers import AutoTokenizer
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM, LLaDA2MoeAttention
from llada_fast.training.distill.attn_viz import _capture_teacher_attn, _mask_to_additive

with open('debug_out.txt', 'w') as f:
    sys.stdout = f
    sys.stderr = f
    
    device = torch.device('cuda:0')
    print("Loading model...")
    try:
        teacher = LLaDA2MoeModelLM.from_pretrained("inclusionAI/LLaDA2.1-mini", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.1-mini", trust_remote_code=True)
        
        attn_mod = teacher.model.layers[0].attention
        print(f"Attention Module Type: {type(attn_mod)}")
        print(f"Initial _attn_implementation on teacher: {getattr(teacher, '_attn_implementation', 'N/A')}")
        print(f"Initial _attn_implementation on config: {getattr(teacher.config, '_attn_implementation', 'N/A')}")
        
        input_ids = tokenizer("Test sentence", return_tensors="pt").input_ids
        B, L = input_ids.shape
        pad_mask = torch.ones((B, L), dtype=torch.bool)
        mask_0_1 = torch.ones((B, 1, L, L), dtype=torch.float32)
        attn_mask_additive = _mask_to_additive(mask_0_1)

        # Hook to inspect out
        def _debug_hook(module, inp, out):
            print(f"HOOK CALLED. out type: {type(out)}")
            if isinstance(out, tuple):
                print(f"out len: {len(out)}")
                for i, x in enumerate(out):
                    print(f"  item {i}: {type(x)} {getattr(x, 'shape', 'NO SHAPE')}")
            else:
                print(f"  out shape: {getattr(out, 'shape', 'NO SHAPE')}")

        handle = attn_mod.register_forward_hook(_debug_hook)

        print("Calling _capture_teacher_attn...")
        res = _capture_teacher_attn(teacher, input_ids, attn_mask_additive, pad_mask, [0], device)
        print(f"Captured results keys: {list(res.keys())}")
        
        handle.remove()
        print("Done.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
