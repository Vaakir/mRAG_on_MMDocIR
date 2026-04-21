# src/data/data_loader.py

import logging
import json
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of records."""
    records = []
    
    # Read the JSONL file line by line, parsing each line as a JSON object and adding it to the records list 
    # (ensures everything isnt loaded into memory at once)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def load_train_data(train_path: Path, pure_text=False) -> List[Dict[str, Any]]:
    """Load training queries."""
    train_data = load_jsonl(train_path)
    if pure_text:
        pure_text_data = [r for r in train_data if r.get("types") == ["Pure-text (Plain-text)"]]
        logger.info(f"Loaded {len(train_data)} qs, filtered to {len(pure_text_data)} pure-text")
        return pure_text_data
    else:
        logger.info(f"Loaded {len(train_data)} questions (all types)")
        return train_data

def load_test_data(test_path: Path) -> List[Dict[str, Any]]:
    """Load test queries."""
    return load_jsonl(test_path)

# -------------------------------------------------------------------
# Example usage:
# train_data = load_train_data(TRAIN_JSONL)
# Each record has: question_id, question, answer, page_ids, types, evidence_images, pdf_path