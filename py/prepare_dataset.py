#!/usr/bin/env python3
"""
Dataset preparation script for NLP to Creo Trail conversion
Converts JSONL format dataset to HuggingFace datasets format
"""

import json
import random
import os
import argparse
from datasets import Dataset, DatasetDict

# Command line argument parser
parser = argparse.ArgumentParser(description="Prepare dataset for NLP to Creo trail training")
parser.add_argument("--input", default="../data/creo_dataset.jsonl", 
                   help="Path to input JSONL file containing NL-trail pairs")
parser.add_argument("--out_dir", default="./ds_creo", 
                   help="Output directory for processed dataset")
args = parser.parse_args()

print(f"Loading data from: {args.input}")
print(f"Output directory: {args.out_dir}")

# List to store all training examples
rows = []

# Read JSONL file line by line
try:
    with open(args.input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                # Parse JSON object from each line
                obj = json.loads(line)
                # Create training example with input/target format expected by model
                rows.append({
                    "input": obj["nl"],      # Natural language description
                    "target": obj["trail"]   # Corresponding Creo trail file commands
                })
                print(f"Loaded example {line_num}: {obj['nl'][:50]}...")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
except FileNotFoundError:
    print(f"Error: Could not find input file: {args.input}")
    exit(1)

if not rows:
    print("Error: No valid examples found in dataset")
    exit(1)

print(f"Total examples loaded: {len(rows)}")

# Shuffle and split into train/test (90% train, 10% test)
random.seed(42)  # For reproducibility
random.shuffle(rows)

split_idx = int(0.9 * len(rows))
train_data = rows[:split_idx]
test_data = rows[split_idx:]

print(f"Train examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")

# Convert to HuggingFace Dataset format and save
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# Create DatasetDict and save to disk
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

dataset_dict.save_to_disk(args.out_dir)

print(f"✓ Saved HuggingFace dataset to: {args.out_dir}")
print("Dataset preparation complete!")
