# Train LoRA (Low-Rank Adaptation) fine-tuning on T5 model for NLP to Creo Trail conversion
# This script implements efficient fine-tuning using LoRA to avoid full model training

import os
import sys
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from peft import LoraConfig, TaskType, get_peft_model
import transformers

# Check transformers version for compatibility
print(f"Transformers version: {transformers.__version__}")
if transformers.__version__ < "4.20.0":
    print("Warning: This script is optimized for transformers >= 4.20.0")

# Environment variables for flexible configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "t5-small")  # Base T5 model (small/base/large)
DATA_DIR   = os.environ.get("DATA_DIR", "./ds_creo")   # Processed dataset directory
OUT_DIR    = os.environ.get("OUT_DIR", "./t5_creo_lora")  # Output directory for LoRA adapter

print(f"Using model: {MODEL_NAME}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUT_DIR}")

# Verify data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory '{DATA_DIR}' not found!")
    print("Please run prepare_dataset.py first to create the processed dataset.")
    sys.exit(1)

# Load the preprocessed dataset from disk
try:
    ds = load_from_disk(DATA_DIR)
    print(f"Loaded dataset with {len(ds['train'])} training and {len(ds['test'])} test examples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure prepare_dataset.py was run successfully.")
    sys.exit(1)

# Initialize tokenizer for the base T5 model
# T5 is a text-to-text transformer that treats all NLP tasks as text generation
try:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loaded tokenizer for {MODEL_NAME}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Please check your internet connection and model name.")
    sys.exit(1)

def preprocess(batch):
    """
    Preprocess the dataset batch for T5 training
    - Tokenizes input natural language descriptions
    - Tokenizes target Creo trail file commands
    - Applies truncation to manage sequence lengths
    """
    # Tokenize input natural language with max length 128 tokens
    model_in = tok(batch["input"], max_length=128, truncation=True)
    
    # Tokenize target trail files as decoder targets with max length 512 tokens
    with tok.as_target_tokenizer():
        labels = tok(batch["target"], max_length=512, truncation=True)
    
    # Add labels (target token IDs) to model inputs
    model_in["labels"] = labels["input_ids"]
    return model_in

# Apply preprocessing to entire dataset, removing original columns
tok_ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

# Load the base T5 model for sequence-to-sequence learning
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
# LoRA adds small trainable matrices to existing model weights
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # Sequence-to-sequence task type
    r=8,                              # Rank of adaptation matrices (lower = fewer parameters)
    lora_alpha=32,                    # Scaling factor for LoRA weights
    lora_dropout=0.1,                 # Dropout rate for LoRA layers
    inference_mode=False              # Training mode (not inference)
)

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_cfg)
# Print how many parameters are trainable (should be much less than full model)
model.print_trainable_parameters()

# Configure training arguments for the Seq2Seq trainer
# Handle different parameter names across transformers versions
training_args = {
    "output_dir": OUT_DIR,                    # Directory to save model checkpoints
    "learning_rate": 2e-4,                    # Learning rate (higher than full fine-tuning)
    "per_device_train_batch_size": 8,         # Batch size per GPU for training
    "per_device_eval_batch_size": 8,          # Batch size per GPU for evaluation
    "num_train_epochs": 5,                    # Number of training epochs
    "logging_steps": 50,                      # Log training metrics every 50 steps
    "save_total_limit": 2,                    # Keep only 2 latest checkpoints
    "predict_with_generate": True,            # Use generation for evaluation metrics
    "fp16": True,                             # Use mixed precision for faster training
    "warmup_steps": 100,                      # Learning rate warmup
    "save_steps": 500,                        # Save checkpoint every 500 steps
    "remove_unused_columns": False,           # Keep all columns for custom processing
}

# Use the correct parameter name based on transformers version
if hasattr(Seq2SeqTrainingArguments, 'eval_strategy'):
    training_args["eval_strategy"] = "epoch"  # New parameter name
else:
    training_args["evaluation_strategy"] = "epoch"  # Old parameter name

try:
    args = Seq2SeqTrainingArguments(**training_args)
    print("Training arguments configured successfully")
except Exception as e:
    print(f"Error configuring training arguments: {e}")
    # Fallback with minimal arguments
    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=2e-4
    )
    print("Using fallback training configuration")

# Data collator handles padding and batching for variable-length sequences
collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

# Initialize the trainer with model, arguments, datasets, and collator
trainer = Seq2SeqTrainer(
    model=model, 
    args=args,
    train_dataset=tok_ds["train"],         # Training split
    eval_dataset=tok_ds["test"],           # Evaluation split
    tokenizer=tok, 
    data_collator=collator
)

# Start the training process
print("Starting training...")
print(f"Training on {len(tok_ds['train'])} examples")
print(f"Evaluating on {len(tok_ds['test'])} examples")

try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    print("This might be due to:")
    print("1. Insufficient GPU memory - try reducing batch size")
    print("2. Dataset format issues - check prepare_dataset.py output")
    print("3. Model compatibility - try a different T5 variant")
    sys.exit(1)

# Save the trained LoRA adapter (not the full model)
try:
    trainer.save_model(OUT_DIR)
    print(f"Model saved to {OUT_DIR}")
except Exception as e:
    print(f"Error saving model: {e}")

# Save the tokenizer configuration for later use
try:
    tok.save_pretrained(OUT_DIR)
    print(f"Tokenizer saved to {OUT_DIR}")
except Exception as e:
    print(f"Error saving tokenizer: {e}")

print("Training complete. LoRA adapter saved to:", OUT_DIR)
print("\nNext steps:")
print("1. Run merge_lora.py to merge the adapter with the base model")
print("2. Run export_onnx.py to convert to ONNX format for C++ inference")
