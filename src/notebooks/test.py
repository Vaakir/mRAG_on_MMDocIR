# src/data/pdf_processor_unstructured.py
# Alternative PDF processor using unstructured library
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import json
import logging
import os

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

NOTEBOOKS_DIR = Path(__file__).resolve().parent
DATA_DIR = NOTEBOOKS_DIR.parent / "data"


def _normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()


def _get_page_number(block: Any) -> Optional[int]:
    if isinstance(block, dict):
        return block.get("page_number")

    page_number = getattr(block, "page_number", None)
    if page_number is not None:
        return page_number

    metadata = getattr(block, "metadata", None)
    if metadata is None:
        return None

    if hasattr(metadata, "to_dict"):
        return metadata.to_dict().get("page_number")

    return getattr(metadata, "page_number", None)


def _serialize_blocks(blocks: Iterable[Any]) -> List[Dict[str, Any]]:
    serialized_blocks: List[Dict[str, Any]] = []
    for block in blocks or []:
        if isinstance(block, dict):
            category = block.get("category", "Text")
            text = _normalize_text(block.get("text", ""))
            page_number = block.get("page_number")
        else:
            category = getattr(block, "category", "Text")
            text = _normalize_text(getattr(block, "text", ""))
            page_number = _get_page_number(block)

        if text:
            serialized_blocks.append(
                {
                    "category": category,
                    "text": text,
                    "page_number": page_number,
                }
            )

    return serialized_blocks


def _extract_with_unstructured_fast(pdf_path: Path) -> List[Dict[str, Any]]:
    blocks = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",
        languages=["eng"],
    )
    return _serialize_blocks(blocks)


def _extract_with_unstructured_auto(pdf_path: Path) -> List[Dict[str, Any]]:
    blocks = partition(filename=str(pdf_path), languages=["eng"])
    return _serialize_blocks(blocks)


def _extract_with_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    import fitz

    document = fitz.open(str(pdf_path))
    try:
        return [
            {
                "category": "Text",
                "text": text,
                "page_number": page_number,
            }
            for page_number, page in enumerate(document, start=1)
            for text in [_normalize_text(page.get_text("text"))]
            if text
        ]
    finally:
        document.close()


def _extract_with_pypdf(pdf_path: Path) -> List[Dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [
        {
            "category": "Text",
            "text": text,
            "page_number": page_number,
        }
        for page_number, page in enumerate(reader.pages, start=1)
        for text in [_normalize_text(page.extract_text() or "")]
        if text
    ]


EXTRACTION_STEPS: Tuple[Tuple[str, Callable[[Path], List[Dict[str, Any]]]], ...] = (
    ("unstructured-fast", _extract_with_unstructured_fast),
    ("unstructured-auto", _extract_with_unstructured_auto),
    ("pymupdf", _extract_with_pymupdf),
    ("pypdf", _extract_with_pypdf),
)


def _should_use_process_pool(use_processes: bool) -> bool:
    if not use_processes:
        return False

    module_name = extract_text_from_pdf.__module__
    if module_name == "__main__":
        return True

    return importlib.util.find_spec(module_name) is not None


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    resolved_path = str(pdf_path.resolve())
    errors: List[str] = []

    logger.info("Loading %s", resolved_path)

    for method_name, extractor in EXTRACTION_STEPS:
        try:
            blocks = extractor(pdf_path)
            if blocks:
                return {
                    "pdf_name": pdf_path.name,
                    "pdf_path": resolved_path,
                    "blocks": blocks,
                    "extraction_method": method_name,
                }

            message = f"{method_name}: no text extracted"
            errors.append(message)
            logger.warning("%s for %s", message, resolved_path)
        except Exception as exc:
            message = f"{method_name}: {exc}"
            errors.append(message)
            logger.warning("%s failed for %s | %s", method_name, resolved_path, exc)

    return {
        "pdf_name": pdf_path.name,
        "pdf_path": resolved_path,
        "blocks": [],
        "extraction_method": None,
        "errors": errors,
    }


def process_all_pdfs(
    pdf_dir: Path,
    max_workers: Optional[int] = None,
    use_processes: bool = True,
) -> List[Dict[str, Any]]:
    """Process every PDF under a directory concurrently and keep the full file set."""
    pdf_dir = Path(pdf_dir).expanduser().resolve()
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    logger.info("Found %s PDF files under %s", len(pdf_files), pdf_dir)

    if not pdf_files:
        return []

    worker_count = max_workers or max(1, (os.cpu_count() or 2) - 1)
    use_process_pool = _should_use_process_pool(use_processes)
    executor_class = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor

    if use_processes and not use_process_pool:
        logger.warning(
            "Process pool is not available from module context '%s'; falling back to threads.",
            extract_text_from_pdf.__module__,
        )

    documents_by_path: Dict[str, Dict[str, Any]] = {}
    failed_paths: List[str] = []

    with executor_class(max_workers=worker_count) as executor:
        future_to_pdf = {
            executor.submit(extract_text_from_pdf, pdf_path): pdf_path for pdf_path in pdf_files
        }

        for index, future in enumerate(as_completed(future_to_pdf), start=1):
            pdf_path = future_to_pdf[future]
            resolved_path = str(pdf_path.resolve())

            try:
                document = future.result()
            except Exception as exc:
                logger.error("Worker failed for %s | %s", resolved_path, exc)
                document = {
                    "pdf_name": pdf_path.name,
                    "pdf_path": resolved_path,
                    "blocks": [],
                    "extraction_method": None,
                    "errors": [f"worker: {exc}"],
                }

            documents_by_path[document["pdf_path"]] = document
            if not document["blocks"]:
                failed_paths.append(document["pdf_path"])

            if index % 10 == 0 or index == len(pdf_files):
                logger.info("Processed %s/%s PDFs", index, len(pdf_files))

    ordered_documents = [documents_by_path[str(pdf_path.resolve())] for pdf_path in pdf_files]

    if failed_paths:
        failed_names = ", ".join(Path(path).name for path in failed_paths[:5])
        logger.warning(
            "Text extraction failed for %s PDFs. First failures: %s",
            len(failed_paths),
            failed_names,
        )
    else:
        logger.info("Successfully extracted text from all %s PDFs", len(pdf_files))

    return ordered_documents


def save_read_pdf_data(
    all_documents: List[Dict[str, Any]],
    output_path: Path = DATA_DIR / "all_documents.json",
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(all_documents, file_handle, ensure_ascii=False, indent=2)


def load_read_documents(path: Path) -> List[Dict[str, Any]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file_handle:
        docs = json.load(file_handle)

    for doc in docs:
        doc["blocks"] = [
            SimpleNamespace(
                category=block["category"],
                text=block["text"],
                page_number=block.get("page_number"),
            )
            for block in doc.get("blocks", [])
        ]

    return docs



if __name__ == "__main__":
    path = DATA_DIR / "train" / "pdfs_train"
    docs = process_all_pdfs(path, use_processes=True)
    # save_read_pdf_data(docs)