import os
import argparse
import re
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
    Wrapper class for Surya OCR that pre-loads detection and recognition models.
    """
    def __init__(self, languages=["en"]):
        self.languages = languages
        print("Pre-loading detection model...")
        self.detection_predictor = DetectionPredictor()
        print("Pre-loading recognition model...")
        self.recognition_predictor = RecognitionPredictor()

    def extract_text_from_image(self, image):
        predictions = self.recognition_predictor([image], [self.languages], self.detection_predictor)
        return predictions

    def extract_text_from_pdf(self, pdf_path, dpi=200, resize_width=2048, poppler_path=None):
        print(f"Converting PDF pages to images (dpi={dpi})...")
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        processed_pages = []
        for page in pages:
            if page.width > resize_width:
                new_height = int((resize_width / page.width) * page.height)
                page = page.resize((resize_width, new_height), Image.Resampling.LANCZOS)
            processed_pages.append(page)
        if not processed_pages:
            return {}
        print(f"Processing {len(processed_pages)} pages in a single batch...")
        predictions = self.recognition_predictor(
            processed_pages, [self.languages] * len(processed_pages), self.detection_predictor
        )
        results = {}
        for i, pred in enumerate(predictions):
            results[f"page_{i+1}"] = pred
        return results

def combine_text(ocr_result):
    """
    Combine text from OCR results into a single string.
    """
    combined_text = ""
    if hasattr(ocr_result, "text_lines"):
        for line in ocr_result.text_lines:
            combined_text += (getattr(line, "text", "") + " ")
    elif isinstance(ocr_result, dict):
        for line in ocr_result.get("text_lines", []):
            combined_text += (line.get("text", "") + " ")
    return combined_text.strip()

def pre_process_text(text):
    """
    Pre-process text by lowercasing, removing punctuation, and extra whitespace.
    """
    text = text.lower()                               # Lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)            # Remove punctuation (keep alphanumerics and whitespace)
    text = re.sub(r'\s+', ' ', text)                   # Replace multiple spaces with a single space
    return text.strip()

def main():
    parser = argparse.ArgumentParser(
        description="Extract and pre-process text using Surya OCR."
    )
    parser.add_argument("file_path", type=str, help="Path to the input image or PDF file")
    parser.add_argument("--langs", type=str, default="en", help="Comma-separated languages (default 'en')")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF conversion (default 200)")
    parser.add_argument("--resize_width", type=int, default=2048, help="Maximum width for PDF images (default 2048)")
    parser.add_argument("--poppler_path", type=str, default=None, help="Path to Poppler bin directory")
    parser.add_argument("--output_file", type=str, default="preprocessed_text.txt", help="Output file for pre-processed text")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.langs.split(",")]
    surya_ocr = SuryaOCR(languages=languages)
    file_ext = os.path.splitext(args.file_path)[1].lower()
    extracted_text = ""

    if file_ext == ".pdf":
        ocr_results = surya_ocr.extract_text_from_pdf(args.file_path, dpi=args.dpi, resize_width=args.resize_width, poppler_path=args.poppler_path)
        for i in range(1, len(ocr_results) + 1):
            page_result = ocr_results.get(f"page_{i}", {})
            extracted_text += combine_text(page_result) + "\n"
    else:
        try:
            image = Image.open(args.file_path)
        except Exception as e:
            print(f"Error: Unable to open file {args.file_path}: {e}")
            return
        ocr_results = surya_ocr.extract_text_from_image(image)
        extracted_text = combine_text(ocr_results)

    print("Extracted Text:")
    print(extracted_text)
    
    processed_text = pre_process_text(extracted_text)
    print("\nPre-Processed Text:")
    print(processed_text)
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(processed_text)
    print(f"Pre-processed text saved to: {args.output_file}")

if __name__ == "__main__":
    main()