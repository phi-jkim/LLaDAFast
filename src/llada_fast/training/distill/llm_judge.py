"""
LLM-based generation quality judge for progressive curriculum evaluation.

Returns 'PASS', 'FAIL', or 'SKIP' based on an OpenAI API call.
Requires OPENAI_API_KEY in the environment (or a .env file).
Falls back to 'FAIL' when the API is unavailable.
"""

import os
from typing import Dict, List


def evaluate_with_llm(
    history: List[Dict],
    prompt: str,
    step: int,
    consecutive_failures: int,
    layer_id: int,
) -> str:
    """
    Ask an LLM judge whether the student model is ready to progress.

    Args:
        history:              list of {"layer": int, "text": str} dicts, oldest first.
        prompt:               the prompt used for generation.
        step:                 current training step (for context).
        consecutive_failures: how many times this layer has failed so far.
        layer_id:             the layer being evaluated.

    Returns:
        'PASS'  — latest generation is readable / coherent; move to next layer.
        'FAIL'  — still learning but not ready; keep training.
        'SKIP'  — no sign of improvement across entire history; revert this layer.
    """
    try:
        from dotenv import load_dotenv
        import openai
    except ImportError:
        print("[LLM Judge] openai/dotenv not installed; returning FAIL.")
        return "FAIL"

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[LLM Judge] OPENAI_API_KEY not found; returning FAIL.")
        return "FAIL"

    try:
        client = openai.OpenAI(api_key=api_key)

        history_str = "".join(
            f"Eval {i + 1} (Layer {e.get('layer', layer_id)}):\n{e.get('text', '')}\n\n"
            for i, e in enumerate(history)
        )

        sys_prompt = (
            f"You are evaluating training progress of Layer {layer_id} at Step {step}.\n\n"
            "History shown oldest-to-newest. Choose ONE verdict:\n"
            "  PASS  — latest generation is readable/coherent. Move to next layer.\n"
            "  FAIL  — still learning but not ready. Keep training this layer.\n"
            "  SKIP  — every generation across the ENTIRE history is pure gibberish "
            "with zero sign of improvement. Use ONLY when certain; this permanently "
            "reverts the layer to teacher softmax weights.\n\n"
            "Default to FAIL when uncertain. Reply with ONLY 'PASS', 'FAIL', or 'SKIP'."
        )
        user_prompt = (
            f"Prompt: {prompt}\n\n"
            f"Generation History:\n{history_str}"
            f"Verdict:"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        if "PASS" in verdict:
            return "PASS"
        if "SKIP" in verdict:
            return "SKIP"
        return "FAIL"

    except Exception as e:
        print(f"[LLM Judge] API call failed: {e}; returning FAIL.")
        return "FAIL"
