# src/data/chunk_loader.py
# Loader for pre-processed chunks from chunks_fixed_size.json

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
# -------------------------------------------------------------------
def load_preprocessed_chunks(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load pre-processed chunks from chunks_fixed_size.json.
    
    This bypasses PDF extraction and chunking, loading chunks directly
    from the JSON file provided.
    
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
        # Load the JSON file containing pre-processed chunks (list of documents with their chunks)
        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Loaded {len(documents)} documents from JSON")
        
        # Flatten the list of documents into a list of chunks with metadata
        all_chunks = []  # List to hold all chunk dictionaries
        total_chunks = 0 # Counter for total number of chunks loaded
        
        # Process each document and its chunks to create a standardized chunk format
        for doc in documents:
            pdf_name = doc['pdf_name'] # Get the PDF name from the document metadata (e.g., "doc1.pdf") to use in chunk metadata and text enrichment
            pdf_path = doc['pdf_path'] # Get the PDF path from the document metadata (e.g., "pdf_train/doc1.pdf") to include in chunk metadata for better traceability
            chunks = doc.get('chunks', []) # Get the list of chunks for the document (each chunk has text and optional page numbers)
            
            # Process each chunk in the current document to create a standardized format with enriched text (including PDF name) and metadata
            for chunk_id, chunk in enumerate(chunks):
                # Create standardized chunk format
                # Prepend PDF name so embeddings capture document identity
                doc_prefix = f"[Document: {pdf_name}]\n"  # Prefix to add to each chunk's text to include the PDF name for better context in the retrieval and generation
                enriched_text = doc_prefix + chunk['text'] # Enrich the chunk text by adding the PDF name as a prefix (to help the model learn associations between content and source document during embedding and retrieval)
                chunk_dict = { # Create a dictionary for the chunk with enriched text and metadata (including PDF name and path for better traceability in retrieval and generation)
                    'text': enriched_text,
                    'char_len': len(enriched_text),
                    'pdf_name': pdf_name,
                    'pdf_path': pdf_path,
                    'chunk_id': chunk_id
                }
                
                # Add page numbers if available (some chunks may not have page numbers, so we check before adding to metadata)
                if 'page_numbers' in chunk:
                    chunk_dict['page_numbers'] = chunk['page_numbers']
                
                all_chunks.append(chunk_dict) # Add the chunk dictionary to the list of all chunks
                total_chunks += 1 # Increment the total chunk counter for statistics
        
        logger.info(f"✓ Extracted {total_chunks} chunks from {len(documents)} documents")
        
        # Show statistics for the loaded chunks (average chunks per document)
        chunks_per_doc = total_chunks / len(documents) if len(documents) > 0 else 0
        logger.info(f"  Average chunks per document: {chunks_per_doc:.1f}")
        
        return all_chunks
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        raise

# -------------------------------------------------------------------
def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about loaded chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not chunks: # If the list of chunks is empty, return default statistics to avoid errors in calculations
        return {}
    
    total_chunks = len(chunks) # Total number of chunks loaded (used for calculating average chunk size and overall statistics)
    total_chars = sum(chunk['char_len'] for chunk in chunks) # Total number of characters across all chunks (used for calculating average chunk size and understanding the overall size of the chunked data)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0 # Keep track of the average chunk size
    
    # Get unique PDFs (based on pdf_name) to understand how many distinct documents are represented in the chunks and to provide insights into the diversity of the chunked data (important for retrieval and generation performance)
    unique_pdfs = set(chunk['pdf_name'] for chunk in chunks)
    
    # Find min/max chunk sizes
    chunk_sizes = [chunk['char_len'] for chunk in chunks]
    min_size = min(chunk_sizes) if chunk_sizes else 0 # Find the minimum chunk size (in characters) to understand the smallest chunk in the dataset (important for understanding the granularity of the chunks and potential issues with very small chunks)
    max_size = max(chunk_sizes) if chunk_sizes else 0 # Find the maximum chunk size (in characters) to understand the largest chunk in the dataset (important for understanding the granularity of the chunks and potential issues with very large chunks that may exceed model input limits)
    
    return { # Return a dictionary with all the calculated statistics about the chunks for reporting and analysis purposes 
        'total_chunks': total_chunks,
        'total_characters': total_chars,
        'average_chunk_size': avg_chunk_size,
        'min_chunk_size': min_size,
        'max_chunk_size': max_size,
        'unique_pdfs': len(unique_pdfs),
        'pdf_names': sorted(unique_pdfs)
    }

# -------------------------------------------------------------------
def print_chunk_statistics(chunks: List[Dict[str, Any]]) -> None:
    """Print formatted chunk statistics."""
    stats = get_chunk_statistics(chunks) # Use the helper function to calculate statistics about the loaded chunks (total number, average size, unique PDFs, etc.) for reporting and analysis purposes
    
    print("\n" + "="*80)
    print("CHUNK STATISTICS")
    print("="*80)
    print(f"Total chunks: {stats['total_chunks']:,}")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Average chunk size: {stats['average_chunk_size']:.0f} characters")
    print(f"Chunk size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} characters")
    print(f"Unique PDFs: {stats['unique_pdfs']}")
    print("="*80 + "\n")
