"""
QualitativeGenerationCallback
------------------------------
After each training epoch, randomly sample a few questions from the
forget/train dataset, generate the model's answers, and log them to
stdout (and WandB if active) so we can qualitatively judge how
unlearning is progressing.
"""

import logging
import random
import textwrap

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


def _get_question_dataset(train_dataset):
    """Unwrap ForgetRetainDataset to the underlying question dataset."""
    if hasattr(train_dataset, "forget") and train_dataset.forget is not None:
        return train_dataset.forget
    return train_dataset


def _get_raw_question(dataset, idx, tokenizer):
    """
    Fetch the clean question string for dataset item at `idx`.
    Priority:
      1. dataset.data[idx][question_key]  — raw text, no template boilerplate
      2. Decode input_ids tokens where labels == -100 (fallback)
    """
    # (1) raw text from underlying HF dataset
    try:
        question_key = getattr(dataset, "question_key", "question")
        raw = dataset.data[idx].get(question_key)
        if raw and isinstance(raw, str):
            return raw
    except Exception:
        pass

    # (2) fallback: decode from tokenized item
    try:
        item = dataset[idx]
        if isinstance(item, dict) and "input_ids" not in item:
            item = next(iter(item.values()))
        input_ids = item.get("input_ids")
        labels = item.get("labels")
        if input_ids is None:
            return None
        if labels is not None:
            labels_t = (
                torch.tensor(labels)
                if not isinstance(labels, torch.Tensor)
                else labels
            )
            input_ids_t = (
                torch.tensor(input_ids)
                if not isinstance(input_ids, torch.Tensor)
                else input_ids
            )
            question_ids = input_ids_t[labels_t == -100]
        else:
            question_ids = torch.tensor(input_ids)
        if len(question_ids) == 0:
            return None
        return tokenizer.decode(question_ids, skip_special_tokens=True)
    except Exception:
        return None


def generate_answer_text(
    model, tokenizer, question_text: str, max_new_tokens: int = 200
) -> str:
    """Format the question with the chat template and run model.generate()."""
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = question_text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_qualitative_generation(
    model,
    tokenizer,
    dataset,
    num_samples: int = 3,
    max_new_tokens: int = 200,
    header: str = "Qualitative Samples",
):
    """
    Randomly samples questions from the dataset, generates answers,
    and prints them to stdout.
    """
    source_dataset = _get_question_dataset(dataset)
    n = min(num_samples, len(source_dataset))
    indices = random.sample(range(len(source_dataset)), n)

    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  {header}")
    print(separator)

    model.eval()
    rows = []
    for rank, idx in enumerate(indices, start=1):
        question = _get_raw_question(source_dataset, idx, tokenizer)
        if question is None:
            logger.warning(f"Could not extract question for index {idx}, skipping.")
            continue

        answer = generate_answer_text(
            model, tokenizer, question, max_new_tokens=max_new_tokens
        )

        q_wrapped = textwrap.fill(question, width=66, subsequent_indent="    ")
        a_wrapped = textwrap.fill(
            answer or "(empty)", width=66, subsequent_indent="    "
        )
        print(f"\n  [{rank}/{n}] Q: {q_wrapped}")
        print(f"        A: {a_wrapped}")

        rows.append({"question": question, "model_output": answer})

    print(f"\n{separator}\n")
    return rows


class QualitativeGenerationCallback(TrainerCallback):
    """
    Generates and logs model responses for a random sample of questions
    from the training dataset after every epoch.
    """

    def __init__(
        self,
        train_dataset,
        tokenizer,
        template_args=None,
        num_samples: int = 3,
        max_new_tokens: int = 200,
    ):
        self.source_dataset = _get_question_dataset(train_dataset)
        self.tokenizer = tokenizer
        self.template_args = template_args
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def _sample_indices(self):
        n = min(self.num_samples, len(self.source_dataset))
        return random.sample(range(len(self.source_dataset)), n)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if args.local_process_index != 0:
            return  # only run on the main process

        epoch = int(state.epoch) if state.epoch is not None else "?"
        rows = run_qualitative_generation(
            model=model,
            tokenizer=self.tokenizer,
            dataset=self.source_dataset,
            num_samples=self.num_samples,
            max_new_tokens=self.max_new_tokens,
            header=f"Qualitative Samples — Epoch {epoch}",
        )
        model.train()

        # ---- WandB logging ----
        try:
            import wandb

            if wandb.run is not None and rows:
                table = wandb.Table(columns=["epoch", "question", "model_output"])
                for row in rows:
                    table.add_data(epoch, row["question"], row["model_output"])
                wandb.log({"qualitative_samples": table}, step=state.global_step)
        except ImportError:
            pass
