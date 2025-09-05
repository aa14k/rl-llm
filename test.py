import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- config ---
BASE_TOKENIZER_ID = "microsoft/phi-4"
MODEL_PATH = "/home/ubuntu/alex/verifiers/outputs/phi-4-math-gamma0.999-seed42-beta0.01-big-capacityblock11/checkpoint-1166"
SYSTEM_PROMPT_STRICT = """You must reply in EXACTLY this XML:

<reasoning>
...
</reasoning>
<answer>
...
</answer>

Rules:
- All text must be wrapped inside a <reasoning> </reasoning> or <answer> </answer> tag. 
"""

# --- helpers ---
def preview_chat_template(tokenizer):
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT_STRICT},
        {"role": "user", "content": "Compute 2+3."},
    ]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print("\n[CHAT TEMPLATE PREVIEW] (first 300 chars)\n", formatted[:300].replace("\n", "\\n"))
    # crude checks that template markers are present
    assert "<|im_start|>system" in formatted and "<|im_start|>assistant" in formatted, \
        "Expected phi-4 chat markers not found — wrong tokenizer or template."

def make_llm():
    llm = LLM(
        model=MODEL_PATH,                 # your **weights**
        tokenizer=BASE_TOKENIZER_ID,      # critical: use the **base tokenizer** with chat template
        trust_remote_code=True,
        max_model_len=4608,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
    )
    return llm

def sampling(no_stop=True):
    # top_k=None disables top-k in vLLM. Avoid passing -1.
    return SamplingParams(
        temperature=0.2,          # low temp for a stable test
        top_p=1.0,
        top_k=-1,
        max_tokens=256,
        seed=42,
        stop=None if no_stop else ["</answer>"],   # we’ll test both modes
    )

def extract_xml_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback if you stopped on </answer> and it got stripped
    m2 = re.search(r"<answer>\s*(.*)$", text, flags=re.DOTALL)
    return m2.group(1).strip() if m2 else ""

def debug_extraction(text: str):
    has_reasoning = "<reasoning>" in text and "</reasoning>" in text
    has_answer_open = "<answer>" in text
    has_answer_close = "</answer>" in text
    ans = extract_xml_answer(text)
    print(f"[TAGS] reasoning:{has_reasoning} answer_open:{has_answer_open} answer_close:{has_answer_close}")
    print(f"[EXTRACTED ANSWER]\n{ans}\n")

# --- run tests ---
print("[TOKENIZER LOAD] using:", BASE_TOKENIZER_ID)
tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID, trust_remote_code=True)
print("name_or_path:", tok.name_or_path)
print("pad_token:", tok.pad_token, "eos_token:", tok.eos_token)

preview_chat_template(tok)

# build a single prompt we can reuse
chat = [
    {"role": "system", "content": SYSTEM_PROMPT_STRICT},
    {"role": "user", "content": "Compute 2+3."},
]
prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print("\n[LLM INIT]")
llm = make_llm()

# --- A) Generate WITHOUT any stop strings ---
print("\n=== GENERATION A: no stop strings ===")
outA = llm.generate([prompt], sampling(no_stop=True))[0].outputs[0].text
print("[RAW OUTPUT A]\n", outA)
debug_extraction(outA)

# --- B) Generate WITH stop on </answer> (to show the tag stripping effect) ---
print("\n=== GENERATION B: stop=['</answer>'] ===")
outB = llm.generate([prompt], sampling(no_stop=False))[0].outputs[0].text
print("[RAW OUTPUT B]\n", outB)
debug_extraction(outB)

print("\n[CHECKLIST]")
print("1) Did the chat preview show <|im_start|>system / user / assistant and <|im_sep|>, <|im_end|>?")
print("2) In Generation A, do you see BOTH <reasoning>...</reasoning> and <answer>...</answer>?")
print("3) In Generation B, does answer_close=False while answer_open=True, and does the fallback still extract the value?")
