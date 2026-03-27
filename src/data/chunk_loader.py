# src/data/chunk_loader.py
# Loader for pre-processed chunks from chunks_fixed_size.json

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_preprocessed_chunks(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load pre-processed chunks from chunks_fixed_size.json.
    
    This bypasses PDF extraction and chunking, loading chunks directly
    from the JSON file provided by team members.
    
    Args:
        json_path: Path to chunks_fixed_size.json
        
    Returns:
        List of chunk dictionaries with:
        - text: chunk text
        - char_len: character length
        - pdf_name: source PDF filename
        - pdf_path: source PDF path
        - chunk_id: sequential ID within document
        - page_numbers: list of page numbers (optional)
    """
    logger.info(f"Loading pre-processed chunks from {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {json_path}\n"
            f"Expected location: {json_path}"
        )
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Loaded {len(documents)} documents from JSON")
        
        # Flatten into list of chunks with metadata
        all_chunks = []
        total_chunks = 0
        
        for doc in documents:
            pdf_name = doc['pdf_name']
            pdf_path = doc['pdf_path']
            chunks = doc.get('chunks', [])
            
            for chunk_id, chunk in enumerate(chunks):
                # Create standardized chunk format
                chunk_dict = {
                    'text': chunk['text'],
                    'char_len': chunk['char_len'],
                    'pdf_name': pdf_name,
                    'pdf_path': pdf_path,
                    'chunk_id': chunk_id
                }
                
                # Add page numbers if available
                if 'page_numbers' in chunk:
                    chunk_dict['page_numbers'] = chunk['page_numbers']
                
                all_chunks.append(chunk_dict)
                total_chunks += 1
        
        logger.info(f"✓ Extracted {total_chunks} chunks from {len(documents)} documents")
        
        # Show statistics
        chunks_per_doc = total_chunks / len(documents) if len(documents) > 0 else 0
        logger.info(f"  Average chunks per document: {chunks_per_doc:.1f}")
        
        return all_chunks
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        raise


def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about loaded chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {}
    
    total_chunks = len(chunks)
    total_chars = sum(chunk['char_len'] for chunk in chunks)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
    
    # Get unique PDFs
    unique_pdfs = set(chunk['pdf_name'] for chunk in chunks)
    
    # Find min/max chunk sizes
    chunk_sizes = [chunk['char_len'] for chunk in chunks]
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    
    return {
        'total_chunks': total_chunks,
        'total_characters': total_chars,
        'average_chunk_size': avg_chunk_size,
        'min_chunk_size': min_size,
        'max_chunk_size': max_size,
        'unique_pdfs': len(unique_pdfs),
        'pdf_names': sorted(unique_pdfs)
    }


def print_chunk_statistics(chunks: List[Dict[str, Any]]) -> None:
    """Print formatted chunk statistics."""
    stats = get_chunk_statistics(chunks)
    
    print("\n" + "="*80)
    print("CHUNK STATISTICS")
    print("="*80)
    print(f"Total chunks: {stats['total_chunks']:,}")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Average chunk size: {stats['average_chunk_size']:.0f} characters")
    print(f"Chunk size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} characters")
    print(f"Unique PDFs: {stats['unique_pdfs']}")
    print("="*80 + "\n")
