"""Dataset for CRIMSON score fine-tuning.

Each sample builds the CRIMSON evaluation prompt from prompt_parts.py
(excluding CONTEXT_GUIDELINES) with ground-truth and candidate reports,
and uses the raw_evaluation JSON as the target output.
"""

import json
import sys
import os

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from CRIMSON.prompt_parts import build_prompt


class CRIMSONDataset(Dataset):
    """Dataset for fine-tuning a model to produce CRIMSON evaluations."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 4096,
    ):
        """
        Args:
            data: List of dicts from the JSONL, each with keys:
                  ground_truth, candidate, patient_context,
                  context_included, raw_evaluation, ...
            tokenizer: HuggingFace tokenizer / processor.
            max_length: Maximum sequence length (input + output).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples: list[dict] = []
        self._skip_reasons: dict[str, int] = {}
        skipped = 0
        for entry in data:
            sample = self._build_sample(entry)
            if sample is not None:
                self.samples.append(sample)
            else:
                skipped += 1
        if skipped:
            print(f"[CRIMSONDataset] Skipped {skipped}/{len(data)} samples:")
            for reason, count in sorted(
                self._skip_reasons.items(), key=lambda x: -x[1]
            ):
                print(f"  - {reason}: {count}")

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_entry(entry: dict) -> str | None:
        """Validate a JSONL entry has correct structure.

        Returns None if valid, or an error message string if invalid.
        """
        if not isinstance(entry, dict):
            return "entry is not a dict"

        for key in ("ground_truth", "candidate", "raw_evaluation"):
            if key not in entry:
                return f"missing required key '{key}'"

        gt = entry["ground_truth"]
        cand = entry["candidate"]
        if not isinstance(gt, str) or not gt.strip():
            return "ground_truth is empty or not a string"
        if not isinstance(cand, str) or not cand.strip():
            return "candidate is empty or not a string"

        re = entry["raw_evaluation"]
        if not isinstance(re, dict):
            return "raw_evaluation is not a dict"

        required_re_keys = {
            "reference_findings",
            "predicted_findings",
            "matched_findings",
            "errors",
        }
        missing = required_re_keys - set(re.keys())
        if missing:
            return f"raw_evaluation missing keys: {missing}"

        for list_key in ("reference_findings", "predicted_findings", "matched_findings"):
            if not isinstance(re[list_key], list):
                return f"raw_evaluation['{list_key}'] is not a list"

        errors = re["errors"]
        if not isinstance(errors, dict):
            return "raw_evaluation['errors'] is not a dict"
        for err_key in ("false_findings", "missing_findings", "attribute_errors"):
            if err_key not in errors:
                return f"raw_evaluation['errors'] missing key '{err_key}'"
            if not isinstance(errors[err_key], list):
                return f"raw_evaluation['errors']['{err_key}'] is not a list"

        return None  # valid

    # ------------------------------------------------------------------
    def _build_sample(self, entry: dict) -> dict | None:
        """Build a single (prompt, target) pair."""
        # Validate structure first
        err = self._validate_entry(entry)
        if err is not None:
            self._skip_reasons[err] = self._skip_reasons.get(err, 0) + 1
            return None

        try:
            ground_truth = entry["ground_truth"]
            candidate = entry["candidate"]
            patient_context = entry.get("patient_context")
            raw_evaluation = entry["raw_evaluation"]

            prompt_text = build_prompt(
                reference_findings=ground_truth,
                predicted_findings=candidate,
                patient_context=patient_context if patient_context else None,
                include_significance_examples=False,
                include_attribute_guidelines=False,
                include_context_guidelines=False,  # exclude CONTEXT_GUIDELINES
            )


            target_text = json.dumps(raw_evaluation, separators=(",", ":"))

            return {"prompt": prompt_text, "target": target_text}

        except Exception as e:
            reason = f"build error: {type(e).__name__}: {e}"
            self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1
            return None

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        target = sample["target"]

        messages = [
            {"role": "user", "content": prompt},
        ]

        # Tokenise prompt (with generation prompt appended so the model
        # sees the assistant turn-start tokens)
        prompt_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_ids = prompt_enc["input_ids"].squeeze(0)

        # Tokenise target (no special tokens – raw content only)
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        target_ids = torch.cat([target_ids, eos])

        input_ids = torch.cat([prompt_ids, target_ids])
        prompt_len = prompt_ids.shape[0]

        # Truncate if necessary (keep prompt, truncate from target end)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            input_ids[-1] = self.tokenizer.eos_token_id

        # Labels: mask prompt tokens with -100
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        attention_mask = torch.ones_like(input_ids)

        # Gemma 3 requires token_type_ids during training.
        # 0 = text token, 1 = image token.  We have no images -> all zeros.
        token_type_ids = torch.zeros_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_type_ids": token_type_ids,
        }


# ------------------------------------------------------------------
# Collate function — left-padding for causal LMs
# ------------------------------------------------------------------
def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences to the same length within a batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("Empty batch – all samples were None.")

    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list, attn_list, labels_list, ttype_list = [], [], [], []

    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]

        if pad_len > 0:
            pad = torch.zeros(pad_len, dtype=torch.long)
            input_ids = torch.cat([pad, item["input_ids"]])
            attention_mask = torch.cat([pad, item["attention_mask"]])
            labels = torch.cat(
                [torch.full((pad_len,), -100, dtype=torch.long), item["labels"]]
            )
            token_type_ids = torch.cat([pad, item["token_type_ids"]])
        else:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            token_type_ids = item["token_type_ids"]

        input_ids_list.append(input_ids)
        attn_list.append(attention_mask)
        labels_list.append(labels)
        ttype_list.append(token_type_ids)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "labels": torch.stack(labels_list),
        "token_type_ids": torch.stack(ttype_list),
    }
