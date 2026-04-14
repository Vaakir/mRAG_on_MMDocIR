# src/preprocessing/extract_figures.py
# Extract images, figures, charts, and tables from PDFs as separate image files
# using docling's built-in detection. Outputs one PNG per detected element.
#
# Usage:
#   python src/preprocessing/extract_figures.py
#   python src/preprocessing/extract_figures.py --pdf-dir src/data/train/pdf_train --output-dir src/data/train/figures_train --scale 2.0

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.datamodel.base_models import InputFormat, DocItemLabel
from docling_core.types.doc import PictureItem, TableItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Labels whose bounding boxes we crop and save as images
VISUAL_LABELS = {DocItemLabel.PICTURE, DocItemLabel.CHART, DocItemLabel.TABLE}


def _build_converter(scale: float = 2.0) -> DocumentConverter:
    """Build a docling converter with picture + page image generation enabled."""
    try:
        import torch
        device = (
            AcceleratorDevice.CUDA
            if torch.cuda.is_available()
            else AcceleratorDevice.CPU
        )
    except ImportError:
        device = AcceleratorDevice.CPU

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = scale
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=device,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def extract_figures_from_pdf(
    converter: DocumentConverter,
    pdf_path: Path,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Extract all figures, charts, and tables from a single PDF.

    Saves each as:  output_dir/{pdf_stem}_{label}_p{page}_{idx}.png

    Returns a list of metadata dicts, one per saved image.
    """
    pdf_stem = pdf_path.stem
    result = converter.convert(str(pdf_path))
    doc = result.document
    total_pages = len(result.pages)
    records = []
    idx = 0

    for item, _level in doc.iterate_items():
        if item.label not in VISUAL_LABELS:
            continue

        page_no = item.prov[0].page_no if item.prov else 0
        label_name = item.label.value  # "picture", "chart", or "table"

        image = item.get_image(doc)

        # Fallback: crop from the rendered page image when get_image() returns None
        if image is None and item.prov:
            page_ix = item.prov[0].page_no - 1
            if page_ix < len(result.pages):
                page_obj = result.pages[page_ix]
                page_image = page_obj.image
                if page_image is not None:
                    page_h = page_obj.size.height
                    tl_bbox = item.prov[0].bbox.to_top_left_origin(page_h)
                    sx = page_image.width / page_obj.size.width
                    sy = page_image.height / page_h
                    crop_box = (
                        int(tl_bbox.l * sx),
                        int(tl_bbox.t * sy),
                        int(tl_bbox.r * sx),
                        int(tl_bbox.b * sy),
                    )
                    image = page_image.crop(crop_box)

        if image is None:
            logger.debug(
                f"  {pdf_stem}: no image for {label_name} on page {page_no}"
            )
            continue

        filename = f"{pdf_stem}_{label_name}_p{page_no}_{idx}.png"
        out_path = output_dir / filename
        image.save(out_path)
        idx += 1
        logger.debug(f"  Saved {out_path.name}")

        records.append({
            "filename": filename,
            "doc_name": pdf_stem,
            "pdf_name": pdf_path.name,
            "label": label_name,
            "page_num": page_no,
            "total_pages": total_pages,
            "width": image.width,
            "height": image.height,
        })

    return records


def extract_all(pdf_dir: Path, output_dir: Path, scale: float = 2.0):
    """Process every PDF in pdf_dir and write extracted images + metadata to output_dir."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    converter = _build_converter(scale)

    all_records = []
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            records = extract_figures_from_pdf(converter, pdf_path, output_dir)
            all_records.extend(records)
            logger.info(f"  -> {len(records)} images extracted")
        except Exception as e:
            logger.error(f"  Error processing {pdf_path.name}: {e}")

    # Save metadata index
    metadata_path = output_dir / "figures_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. {len(all_records)} images saved to {output_dir}")
    logger.info(f"Metadata written to {metadata_path}")


def main():
    # Default paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    default_pdf_dir = project_root / "src" / "data" / "train" / "pdf_train"
    default_output_dir = project_root / "src" / "data" / "train" / "figures_train"

    parser = argparse.ArgumentParser(
        description="Extract figures, charts, and tables from PDFs as images using docling.",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=default_pdf_dir,
        help="Directory containing input PDFs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_output_dir,
        help="Directory to save extracted images",
    )
    parser.add_argument(
        "--scale", type=float, default=2.0,
        help="Image scale factor (higher = better resolution, default 2.0)",
    )
    args = parser.parse_args()

    extract_all(args.pdf_dir, args.output_dir, args.scale)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
