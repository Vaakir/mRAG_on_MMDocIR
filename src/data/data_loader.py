# src/data/data_loader.py

import json
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def load_train_data(train_path: Path) -> List[Dict[str, Any]]:
    """Load training queries."""
    return load_jsonl(train_path)

def load_test_data(test_path: Path) -> List[Dict[str, Any]]:
    """Load test queries."""
    return load_jsonl(test_path)

# Example usage:
# train_data = load_train_data(TRAIN_JSONL)
# Each record has: question_id, question, answer, page_ids, types, evidence_images, pdf_path