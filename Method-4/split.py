# import os
# import argparse
# import json
# from PIL import Image, ImageDraw, ImageFont
# from surya.recognition import RecognitionPredictor
# from surya.detection import DetectionPredictor
# from pdf2image import convert_from_path

# # --- Performance Optimization: Set Environment Variables ---
# os.environ["TORCH_DEVICE"] = "cuda"  
# os.environ["COMPILE_RECOGNITION"] = "true"
# os.environ["COMPILE_DETECTOR"] = "true"
# os.environ["RECOGNITION_BATCH_SIZE"] = "512"
# os.environ["DETECTOR_BATCH_SIZE"] = "36"

# # Custom JSON encoder to handle non-serializable OCR result objects
# class SuryaJSONEncoder(json.JSONEncoder):
#     def default(self, o):
#         if hasattr(o, "to_dict"):
#             return o.to_dict()
#         try:
#             return o.__dict__
#         except Exception:
#             return str(o)

# def draw_ocr_results(image, ocr_result):
#     """
#     Draw bounding boxes and recognized text on an image.
    
#     Args:
#         image (PIL.Image.Image): The image to annotate.
#         ocr_result (dict or object): OCR result for the page. Expected to contain a "text_lines" attribute.
        
#     Returns:
#         PIL.Image.Image: The annotated image.
#     """
#     draw = ImageDraw.Draw(image)
#     try:
#         font = ImageFont.truetype("arial.ttf", 16)
#     except IOError:
#         font = ImageFont.load_default()

#     # Try to retrieve text_lines from a Pydantic model attribute or from a dict.
#     if hasattr(ocr_result, "text_lines"):
#         text_lines = ocr_result.text_lines
#     elif isinstance(ocr_result, dict):
#         text_lines = ocr_result.get("text_lines", [])
#     else:
#         text_lines = []

#     for line in text_lines:
#         # Use getattr to access attributes from the Pydantic model
#         bbox = getattr(line, "bbox", None)
#         text = getattr(line, "text", "")
#         if bbox:
#             draw.rectangle(bbox, outline="red", width=2)
#             draw.text((bbox[0], max(bbox[1] - 15, 0)), text, fill="blue", font=font)
#     return image

# class SuryaOCR:
#     """
#     Wrapper class for Surya OCR that pre-loads the detection and recognition models.
#     """
#     def __init__(self, languages=["en"]):
#         self.languages = languages
#         print("Pre-loading detection model...")
#         self.detection_predictor = DetectionPredictor()
#         print("Pre-loading recognition model...")
#         self.recognition_predictor = RecognitionPredictor()

#     def extract_text_from_image(self, image):
#         """
#         Extract text from a single PIL image.
#         """
#         predictions = self.recognition_predictor([image], [self.languages], self.detection_predictor)
#         return predictions

#     def extract_text_from_pdf(self, pdf_path, dpi=200, resize_width=2048, poppler_path=None):
#         """
#         Convert PDF pages to images and extract text from each page in a single batch.
        
#         Args:
#             pdf_path (str): Path to the PDF file.
#             dpi (int): DPI for converting PDF pages.
#             resize_width (int): Maximum width for PDF page images.
#             poppler_path (str): Path to Poppler's bin directory (if not in PATH).
        
#         Returns:
#             dict: OCR results per page.
#         """
#         print(f"Converting PDF pages to images (dpi={dpi})...")
#         pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
#         processed_pages = []
#         for i, page in enumerate(pages):
#             if page.width > resize_width:
#                 new_height = int((resize_width / page.width) * page.height)
#                 page = page.resize((resize_width, new_height), Image.Resampling.LANCZOS)
#             processed_pages.append(page)
        
#         if not processed_pages:
#             return {}
        
#         print(f"Processing {len(processed_pages)} pages in a single batch...")
#         predictions = self.recognition_predictor(
#             processed_pages, [self.languages] * len(processed_pages), self.detection_predictor
#         )
        
#         results = {}
#         for i, pred in enumerate(predictions):
#             results[f"page_{i+1}"] = pred
#         return results

# def main():
#     parser = argparse.ArgumentParser(
#         description="Fast OCR extraction from images and PDFs using Surya with performance optimizations, saving annotated images and JSON output."
#     )
#     parser.add_argument("file_path", type=str, help="Path to the input image or PDF file")
#     parser.add_argument("--langs", type=str, default="en", help="Comma-separated list of languages (default 'en')")
#     parser.add_argument("--dpi", type=int, default=200, help="DPI for converting PDF pages (default 200)")
#     parser.add_argument("--resize_width", type=int, default=2048, help="Maximum width for PDF page images (default 2048)")
#     parser.add_argument("--poppler_path", type=str, default=None, help="Path to the Poppler bin directory (e.g., 'D:\\poppler-24.08.0\\Library\\bin')")
#     parser.add_argument("--output_dir", type=str, default=None, help="Directory to save annotated images and JSON output")
#     args = parser.parse_args()

#     languages = [lang.strip() for lang in args.langs.split(",")]
#     surya_ocr = SuryaOCR(languages=languages)

#     file_ext = os.path.splitext(args.file_path)[1].lower()
#     if file_ext == ".pdf":
#         ocr_results = surya_ocr.extract_text_from_pdf(
#             args.file_path, dpi=args.dpi, resize_width=args.resize_width, poppler_path=args.poppler_path
#         )
#     else:
#         try:
#             image = Image.open(args.file_path)
#         except Exception as e:
#             print(f"Error: Unable to open file {args.file_path}: {e}")
#             return
#         ocr_results = surya_ocr.extract_text_from_image(image)

#     # Prepare output directory
#     output_dir = args.output_dir if args.output_dir else "."
#     os.makedirs(output_dir, exist_ok=True)

#     # Save annotated images
#     if file_ext == ".pdf":
#         print("Annotating and saving PDF pages...")
#         pages = convert_from_path(args.file_path, dpi=args.dpi, poppler_path=args.poppler_path)
#         for i, page in enumerate(pages):
#             if page.width > args.resize_width:
#                 new_height = int((args.resize_width / page.width) * page.height)
#                 page = page.resize((args.resize_width, new_height), Image.Resampling.LANCZOS)
#             page_result = ocr_results.get(f"page_{i+1}", {})
#             annotated_page = draw_ocr_results(page, page_result)
#             out_image_path = os.path.join(output_dir, f"page_{i+1}_annotated.png")
#             annotated_page.save(out_image_path)
#             print(f"Saved annotated image: {out_image_path}")
#     else:
#         annotated_image = draw_ocr_results(image, ocr_results)
#         out_image_path = os.path.join(output_dir, "annotated_image.png")
#         annotated_image.save(out_image_path)
#         annotated_image.show()
#         print(f"Saved annotated image: {out_image_path}")

#     # Save JSON results
#     json_out_path = os.path.join(output_dir, "ocr_results.json")
#     with open(json_out_path, "w", encoding="utf-8") as f:
#         json.dump(ocr_results, f, indent=2, cls=SuryaJSONEncoder)
#     print(f"Saved OCR results JSON: {json_out_path}")

# if __name__ == "__main__":
#     main()







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
        Convert PDF pages to images and extract text from each page in a single batch.
        
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
    
    Args:
        ocr_result (object or dict): The OCR result which contains a "text_lines" attribute.
        
    Returns:
        str: The concatenated text.
    """
    combined_text = ""
    if hasattr(ocr_result, "text_lines"):
        for line in ocr_result.text_lines:
            # Using getattr for safe attribute access
            combined_text += (getattr(line, "text", "") + " ")
    elif isinstance(ocr_result, dict):
        for line in ocr_result.get("text_lines", []):
            combined_text += (line.get("text", "") + " ")
    return combined_text.strip()

def pre_process_text(text):
    """
    Pre-process text by lowercasing, removing punctuation, and extra whitespace.
    
    Args:
        text (str): The raw extracted text.
        
    Returns:
        str: The cleaned text.
    """
    # Lowercase the text
    text = text.lower()
    # Remove punctuation using regex (keep alphanumerics and whitespace)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    parser = argparse.ArgumentParser(
        description="Extract text using Surya OCR and pre-process it."
    )
    parser.add_argument("file_path", type=str, help="Path to the input image or PDF file")
    parser.add_argument("--langs", type=str, default="en", help="Comma-separated list of languages (default 'en')")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for converting PDF pages (default 200)")
    parser.add_argument("--resize_width", type=int, default=2048, help="Maximum width for PDF page images (default 2048)")
    parser.add_argument("--poppler_path", type=str, default=None, help="Path to the Poppler bin directory (e.g., 'D:\\poppler-24.08.0\\Library\\bin')")
    parser.add_argument("--output_file", type=str, default="preprocessed_text.txt", help="File to save the pre-processed text")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.langs.split(",")]
    surya_ocr = SuryaOCR(languages=languages)

    file_ext = os.path.splitext(args.file_path)[1].lower()
    extracted_text = ""

    if file_ext == ".pdf":
        ocr_results = surya_ocr.extract_text_from_pdf(
            args.file_path, dpi=args.dpi, resize_width=args.resize_width, poppler_path=args.poppler_path
        )
        # Combine text from all pages
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
    
    # Save the pre-processed text to a file
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(processed_text)
    print(f"Pre-processed text saved to: {args.output_file}")

if __name__ == "__main__":
    main()
