# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments


import re
import torch
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
# NEW import for the custom trainer logic
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
from typing import Dict, Any, List
# NEW import for manually creating the dataloader
from torch.utils.data import DataLoader


# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    # Handle cases where the model might not generate the answer tag
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main', split=split) # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    # Filter out examples where the answer could not be extracted
    return data.filter(lambda x: x['answer'] is not None) # type: ignore

# Reward functions (no changes here)
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1: count += 0.125
    if text.count("\n</reasoning>\n") == 1: count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

# --- NEW: Custom Data Collator for Evaluation ---
@dataclass
class EvalDataCollator:
    """
    Custom data collator for evaluation that handles string-based 'answer' columns.
    This collator separates the 'answer' column before padding and adds it back after.
    """
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate answers, which are strings and should not be padded or converted to tensors.
        answers = [feature.pop("answer") for feature in features]
        
        # Use the tokenizer's default padding for the rest of the features (input_ids, attention_mask).
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        # Add the list of string answers back to the batch.
        batch["answer"] = answers
        return batch

# ----------------------------------------------------------------
# Custom Trainer for Reference Model Updating
# ----------------------------------------------------------------
class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, eval_dataset: Dataset, eval_steps: int, per_device_eval_batch_size: int, **kwargs):
        processing_class = kwargs.get("processing_class", None)
        if processing_class is None:
            raise ValueError("CustomGRPOTrainer requires a processing_class (tokenizer).")
        self.tokenizer = processing_class

        self.eval_data_collator = EvalDataCollator(tokenizer=self.tokenizer)

        super().__init__(*args, **kwargs)

        self.eval_dataset = eval_dataset
        self.custom_eval_steps = eval_steps
        self.custom_per_device_eval_batch_size = per_device_eval_batch_size
        self.best_accuracy = -1.0
        self.eval_generation_kwargs = {
            "max_new_tokens": self.args.max_completion_length,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": False,
        }
        self._forward_pass_counter = 0

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any], *args, **kwargs) -> torch.Tensor:
        self._forward_pass_counter += 1
        
        loss = super().training_step(model, inputs)
        
        global_step = self._forward_pass_counter // self.args.gradient_accumulation_steps

        if global_step > 0 and global_step % self.custom_eval_steps == 0:
            if self._forward_pass_counter % self.args.gradient_accumulation_steps == 0:
                 self._evaluate_and_update_ref_model(global_step)
        return loss

    def _evaluate_and_update_ref_model(self, current_global_step: int):
        if not self.accelerator.is_main_process:
            return

        self.accelerator.print(f"\n--- Global Step {current_global_step}: Running evaluation to potentially update reference model ---")
        
        policy_model = self.accelerator.unwrap_model(self.model)
        policy_model.eval()

        correct_predictions = 0
        total_predictions = 0
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.custom_per_device_eval_batch_size,
            collate_fn=self.eval_data_collator,
            shuffle=False,
            drop_last=False
        )
        eval_dataloader = self.accelerator.prepare(eval_dataloader)

        for batch in tqdm(eval_dataloader, desc="Evaluating on validation set", disable=not self.accelerator.is_local_main_process):
            ground_truth_answers = batch.pop("answer")

            with torch.no_grad():
                generated_ids = policy_model.generate(
                    **batch,
                    **self.eval_generation_kwargs,
                )
            
            response_ids = generated_ids[:, batch["input_ids"].shape[1]:]
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            for response, gt_answer in zip(responses, ground_truth_answers):
                extracted_answer = extract_xml_answer(response)
                if extracted_answer == gt_answer:
                    correct_predictions += 1
            total_predictions += len(ground_truth_answers)

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        self.accelerator.print(f"Validation Correctness: {accuracy:.4f}")

        self.log({"eval/correctness_accuracy": accuracy})

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.accelerator.print(f"🚀 New best accuracy: {accuracy:.4f}. Updating reference model.")
            policy_model_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            self.ref_model.load_state_dict(policy_model_state_dict)
        else:
            self.accelerator.print(f"Accuracy did not improve. Best so far: {self.best_accuracy:.4f}. Keeping old reference model.")
        
        policy_model.train()
        self.accelerator.print("--- Evaluation finished ---")

# ----------------------------------------------------------------
# Main script logic
# ----------------------------------------------------------------

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
seed = 42

run_name = f"{model_name.split('/')[-1]}-gsm8k-custom-ref-seed{seed}"
output_dir = f"outputs/{run_name}"

full_dataset = get_gsm8k_questions()
split_dataset = full_dataset.train_test_split(test_size=0.05, seed=seed)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    beta=0.4,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    seed=seed,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=2,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    overwrite_output_dir=True,
    disable_dropout=True,
    # We are handling syncing manually, so no need to set sync_ref_model
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    #device_map="auto",
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# Pre-tokenize the evaluation dataset so the data collator receives the correct format
def preprocess_for_eval(examples):
    # Apply the chat template and tokenize the prompts.
    prompts_str = [tokenizer.apply_chat_template(
        p, 
        tokenize=False,
        add_generation_prompt=True # Ensures the model is prompted to generate a response
    ) for p in examples["prompt"]]
    tokenized = tokenizer(prompts_str, truncation=True, max_length=training_args.max_prompt_length)
    # Keep the answer for evaluation
    tokenized["answer"] = examples["answer"]
    return tokenized

eval_dataset = eval_dataset.map(
    preprocess_for_eval,
    batched=True,
    remove_columns=eval_dataset.column_names
)


trainer = CustomGRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    eval_steps=1,
    per_device_eval_batch_size=512,
)

trainer.train()

trainer.accelerator.wait_for_everyone()
if trainer.accelerator.is_main_process:
    trainer.save_model(output_dir + "/final")
    tokenizer.save_pretrained(output_dir + "/final")
trainer.accelerator.wait_for_everyone()
