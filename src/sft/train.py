"""Train a single Qwen LoRA adapter on one query-grouped training partition."""
import math

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from common import MODELS, experiment_parser, load_model, new_run_dir, partition, permission_prompt, write_json


def encode(rows, tokenizer, max_length):
    """Use Qwen's assistant prefix and supervise the answer plus the end-of-turn token."""
    encoded = []
    for row in rows:
        prompt_ids = permission_prompt(tokenizer, row["prompt"])
        complete = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
        if complete[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Chat template prefix mismatch for query {row['query_id']}.")
        suffix = complete[len(prompt_ids):]
        end = suffix.index(tokenizer.eos_token_id)
        if tokenizer.decode(suffix[:end]) != row["completion"][0]["content"]:
            raise ValueError("Tokenized answer differs from the source label.")
        if tokenizer.decode(suffix[end+1:]).strip():
            raise ValueError("Unexpected content after the assistant end-of-turn token.")
        answer = suffix[:end+1]
        ids = prompt_ids + answer
        if len(ids) > max_length:
            raise ValueError(f"Example exceeds {max_length} tokens; refusing to truncate its answer.")
        encoded.append({"input_ids": ids, "completion_mask": [0]*len(prompt_ids) + [1]*len(answer)})
    return Dataset.from_list(encoded)


def verify_supervision(dataset, collate, rows, tokenizer, batch_size):
    """Check the actual labels fed to the trainer, including padded positions."""
    for start in range(0, len(dataset), batch_size):
        examples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = collate(examples)
        for i, example in enumerate(examples):
            labels = batch["labels"][i]
            input_ids = batch["input_ids"][i]
            supervised = labels != -100
            expected = example["completion_mask"] + [0] * (len(labels) - len(example["completion_mask"]))
            if supervised.tolist() != list(map(bool, expected)):
                raise RuntimeError("Prompt or padding was included in the loss.")
            tokens = labels[supervised].tolist()
            if (not torch.equal(labels[supervised], input_ids[supervised])
                    or tokens[-1] != tokenizer.eos_token_id
                    or tokenizer.decode(tokens[:-1]) != rows[start+i]["completion"][0]["content"]):
                raise RuntimeError("Supervised tokens do not match the gold answer and end-of-turn token.")



def main():
    parser = experiment_parser(__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive override for a smoke test")
    args = parser.parse_args()
    if args.epochs < 1 or (args.max_steps != -1 and args.max_steps < 1):
        parser.error("Use positive epochs and positive max steps, or -1 for complete epochs")
    set_seed(args.seed)
    train_rows, _, info = partition(args)
    model, tokenizer = load_model(args.model)
    model.config.use_cache = False
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
    model = get_peft_model(model, lora)
    model.peft_config["default"].base_model_name_or_path = info["model_id"]
    model.peft_config["default"].revision = info["model_revision"]
    train_ds = encode(train_rows, tokenizer, 512)
    out = new_run_dir(args.output_dir)
    _, _, batch, accumulation = MODELS[args.model]
    config = SFTConfig(
        output_dir=str(out / "checkpoints"), seed=args.seed, data_seed=args.seed,
        max_length=512, dataset_kwargs={"skip_prepare_dataset": True}, packing=False,
        completion_only_loss=True, bf16=True, fp16=False, gradient_checkpointing=False,
        per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation,
        learning_rate=5e-5, weight_decay=0.0, lr_scheduler_type="linear", warmup_ratio=0.05,
        optim="adamw_torch", num_train_epochs=args.epochs, max_steps=args.max_steps,
        logging_steps=1, logging_nan_inf_filter=False, disable_tqdm=True, report_to="none",
        eval_strategy="no", save_strategy="no", dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer, train_dataset=train_ds)
    verify_supervision(train_ds, trainer.data_collator, train_rows, tokenizer, batch)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    steps = trainer.state.global_step
    expected = args.max_steps if args.max_steps > 0 else math.ceil(len(train_rows) / (batch * accumulation)) * args.epochs
    if steps != expected:
        raise RuntimeError(f"Expected {expected} optimizer steps, observed {steps}")
    if not math.isfinite(result.training_loss) or any(
        not math.isfinite(log[key]) for log in trainer.state.log_history
        for key in ("loss", "grad_norm") if key in log
    ):
        raise FloatingPointError("Non-finite training metric")
    if not any(param.detach().count_nonzero().item() for name, param in model.named_parameters()
               if "lora_B." in name):
        raise RuntimeError("LoRA weights did not update")
    model.save_pretrained(out / "adapter", save_embedding_layers=False)
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    write_json(out / "training.json", {**info, "train_rows": len(train_rows), "optimizer_steps": steps,
               "effective_batch_size": batch * accumulation, "epochs_completed": trainer.state.epoch,
               "mean_train_loss": result.training_loss, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30})


if __name__ == "__main__":
    main()
