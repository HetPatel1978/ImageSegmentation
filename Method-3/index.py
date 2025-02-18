import os
import argparse
import json
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from pdf2image import convert_from_path

# --- Performance Optimization: Set Environment Variables ---
os.environ["TORCH_DEVICE"] = "cuda"  
os.environ["COMPILE_RECOGNITION"] = "true"
os.environ["COMPILE_DETECTOR"] = "true"
os.environ["RECOGNITION_BATCH_SIZE"] = "512"
os.environ["DETECTOR_BATCH_SIZE"] = "36"

class SuryaOCR:
    """
    Wrapper class for Surya OCR that pre-loads the detection and recognition models.
    """
    def __init__(self, languages=["en"]):
        self.languages = languages
        print("Pre-loading detection model...")
        self.detection_predictor = DetectionPredictor()
        print("Pre-loading recognition model...")
        self.recognition_predictor = RecognitionPredictor()

    def extract_text_from_image(self, image):
        """
        Extract text from a single PIL image.
        """
        predictions = self.recognition_predictor([image], [self.languages], self.detection_predictor)
        return predictions

    def extract_text_from_pdf(self, pdf_path, dpi=200, resize_width=2048, poppler_path=None):
        """
        Convert PDF pages to images and extract text from each page.
        
        Args:
            pdf_path (str): Path to the PDF file.
            dpi (int): DPI for converting PDF pages.
            resize_width (int): Maximum width for PDF page images.
            poppler_path (str): Path to Poppler's bin directory (if not in PATH).
        
        Returns:
            dict: OCR results per page.
        """
        print(f"Converting PDF pages to images (dpi={dpi})...")
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        results = {}
        for i, page in enumerate(pages):
            if page.width > resize_width:
                new_height = int((resize_width / page.width) * page.height)
                # Use Image.Resampling.LANCZOS instead of the deprecated Image.ANTIALIAS
                page = page.resize((resize_width, new_height), Image.Resampling.LANCZOS)
            print(f"Processing page {i+1}...")
            page_result = self.extract_text_from_image(page)
            results[f"page_{i+1}"] = page_result
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Fast OCR extraction from images and PDFs using Surya with performance optimizations."
    )
    parser.add_argument("file_path", type=str, help="Path to the input image or PDF file")
    parser.add_argument("--langs", type=str, default="en", help="Comma-separated list of languages (default 'en')")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for converting PDF pages (default 200)")
    parser.add_argument("--resize_width", type=int, default=2048, help="Maximum width for PDF page images (default 2048)")
    parser.add_argument("--poppler_path", type=str, default=None, help="Path to the Poppler bin directory (e.g., 'C:\\poppler\\bin')")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.langs.split(",")]
    surya_ocr = SuryaOCR(languages=languages)

    file_ext = os.path.splitext(args.file_path)[1].lower()
    if file_ext == ".pdf":
        results = surya_ocr.extract_text_from_pdf(args.file_path, dpi=args.dpi, resize_width=args.resize_width, poppler_path=args.poppler_path)
    else:
        try:
            image = Image.open(args.file_path)
        except Exception as e:
            print(f"Error: Unable to open file {args.file_path}: {e}")
            return
        results = surya_ocr.extract_text_from_image(image)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
