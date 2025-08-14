# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
From TRL vlm example:

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""

from dataclasses import dataclass, field
from functools import partial
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from PIL import Image

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class ExtraConfig:
    """Enhanced training configuration with additional parameters."""

    # Torch compilation settings
    torch_compile: bool = field(
        default=False,
        metadata={
            "help": "Whether to compile the model with torch.compile for faster training"
        },
    )
    torch_compile_backend: str = field(
        default="inductor",
        metadata={
            "help": "Backend to use for torch.compile (inductor, aot_eager, cudagraphs)"
        },
    )
    torch_compile_mode: str = field(
        default="default",
        metadata={
            "help": "Compilation mode (default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs)"
        },
    )

    # Vision encoder settings
    freeze_vision_encoder: bool = field(
        default=True,
        metadata={
            "help": "Whether to freeze the vision encoder parameters to save memory"
        },
    )


def image_valid(image: Image.Image, max_aspect_ratio: float = 195.0) -> bool:
    """
    The Qwen2.5-VL model has a hardcoded limit of 200 in smart_resize
    """
    width, height = image.size
    aspect_ratio = max(width / height, height / width)
    return aspect_ratio <= max_aspect_ratio


# hack, filter images with invalid aspect ratios. should probably do this in the dataset instead
def _all_images_valid(images, max_aspect_ratio=195.0):
    if isinstance(images, list):
        return all(
            isinstance(img, Image.Image) and image_valid(img, max_aspect_ratio)
            for img in images
        )
    elif isinstance(images, Image.Image):
        return image_valid(images, max_aspect_ratio)
    else:
        return False


def make_tokenized_batch(
    processor: Qwen2_5_VLProcessor, examples: list[dict], max_length: int
):
    # Build conversations with explicit image placeholders so the chat template inserts image tokens
    image_column = "image"

    n = len(examples)
    examples = [e for e in examples if _all_images_valid(e[image_column])]
    if wandb and wandb.run:
        wandb.log({"data/invalid_examples": n - len(examples)})

    if len(examples) == 0:
        print(
            "ERROR: All examples in batch were filtered out due to invalid aspect ratios. Probably going to crash.",
            examples,
        )

    conversations = [
        format_example(
            example,
            prompt_column="prompt",
            completion_column="completion",
            image_column=image_column,
        )
        for example in examples
    ]
    images = [example[image_column] for example in examples]
    texts = [processor.apply_chat_template(c, tokenize=False) for c in conversations]

    # Ensure truncation happens on the right to preserve initial image tokens
    processor.tokenizer.truncation_side = "right"
    processor.tokenizer.padding_side = "right"

    # Tokenize texts and process images together
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Log padding stats to wandb
    if wandb and wandb.run:
        pad_mask = batch["input_ids"] == processor.tokenizer.pad_token_id
        wandb.log({"data/padding_count": pad_mask.int().sum().item()})

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # Ignore the image token index in the loss computation (model specific)
    labels[labels == processor.image_token_id] = -100
    batch["labels"] = labels

    return batch


def image_tag_to_multipart_message(prompt: str, image_count: int) -> list[dict]:
    """
    Convert
    ```
    "a prompt with <image> tag"
    ```
    to
    ```
    [
        {"type": "text", "text": "a prompt with"},
        {"type": "image"},
        {"type": "text", "text": " tag"},
    ]
    ```
    """
    parts = prompt.split("<image>")
    content = []
    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < image_count:
            content.append({"type": "image"})
        elif i > image_count + 1:
            raise ValueError(
                f"Too many <image> tags in prompt: {prompt} (expected {image_count})"
            )
    return content


def format_example(
    example,
    prompt_column="prompt",
    completion_column="completion",
    image_column="image",
) -> list[dict]:
    """Format as multimodal messages; insert image placeholder where '<image>' appears."""
    prompt_raw = example[prompt_column]
    prompt = prompt_raw if isinstance(prompt_raw, str) else ""
    completion = example[completion_column]
    images = example[image_column]
    if isinstance(images, Image.Image):
        images = [images]
    elif isinstance(images, str):
        images = [Image.open(images)]

    content = image_tag_to_multipart_message(prompt, image_count=len(images))

    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": completion},
    ]

    return messages


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, ExtraConfig))
    script_args, training_args, model_args, our_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Append wandb run name to output directory so multiple runs don't overwrite each other
    if wandb and wandb.run:
        training_args.output_dir = f"{training_args.output_dir}/{wandb.run.name}"

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    # Apply torch compilation if enabled (this is not working with deepspeed zero-3)
    if our_args.torch_compile:
        model = torch.compile(
            model,
            backend=our_args.torch_compile_backend,
            mode=our_args.torch_compile_mode,
        )

    # Freeze vision encoder to save memory - only train the language model
    if our_args.freeze_vision_encoder:
        for param in model.visual.parameters():
            param.requires_grad = False
        print("🔒 Froze vision encoder parameters")

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=partial(
            make_tokenized_batch,
            processor,
            max_length=training_args.max_length,
        ),
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
