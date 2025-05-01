import os
import time
import logging
import concurrent.futures
from typing import Tuple, List, Dict, Union, Optional
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import pdfplumber
import docx
import re
import cv2
import numpy as np
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
from collections import defaultdict
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Application Configuration
app.config.update(
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'pdf', 'docx'},
    UPLOAD_FOLDER='uploads',
    MAX_CONTENT_LENGTH=32 * 1024 * 1024,  # 32MB
    PROCESSING_TIMEOUT=30,  # seconds
    MODEL_NAME="microsoft/layoutlmv3-base",
    OCR_CONFIGS=[
        {'preprocess': 'advanced', 'config': r'--oem 3 --psm 6'},
        {'preprocess': 'sharpen', 'config': r'--oem 1 --psm 11'},
        {'preprocess': 'autocontrast', 'config': r'--oem 3 --psm 4'}
    ]
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DocumentProcessor:
    def __init__(self, poppler_path: str = None):
        self.poppler_path = poppler_path

    def _preprocess_image(self, img: Image.Image, method: str) -> Image.Image:
        """Image preprocessing pipeline"""
        try:
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            
            if method == 'advanced':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                return Image.fromarray(thresh)
            elif method == 'sharpen':
                return Image.fromarray(gray).filter(ImageFilter.SHARPEN)
            elif method == 'autocontrast':
                return ImageOps.autocontrast(Image.fromarray(gray))
            else:
                return Image.fromarray(gray)
        except Exception as e:
            logger.warning(f"Preprocessing failed: {str(e)}")
            return img.convert('L')

    def _merge_text_elements(self, words: List[str], boxes: List[List[int]]) -> Tuple[List[str], List[List[int]]]:
        """Merge broken text elements while preserving layout"""
        if not words:
            return [], []
            
        merged_words = [words[0]]
        merged_boxes = [boxes[0].copy()]
        line_height = boxes[0][3] - boxes[0][1]
        
        for i in range(1, len(words)):
            prev_box = merged_boxes[-1]
            curr_box = boxes[i]
            
            # Check if same line and horizontally close
            same_line = abs(curr_box[1] - prev_box[1]) < line_height * 0.5
            close_horizontal = (curr_box[0] - prev_box[2]) < line_height * 0.8
            
            if same_line and close_horizontal:
                merged_words[-1] += " " + words[i]
                merged_boxes[-1][2] = curr_box[2]  # Extend right boundary
            else:
                merged_words.append(words[i])
                merged_boxes.append(curr_box.copy())
                line_height = curr_box[3] - curr_box[1]  # Update line height
                
        return merged_words, merged_boxes

    def extract_from_image(self, image_path: str) -> Dict[str, Union[List[str], List[List[int]]]]:
        """Robust text extraction with layout preservation"""
        result = {'text': [], 'boxes': [], 'confidences': []}
        
        try:
            with Image.open(image_path) as img:
                for strategy in app.config['OCR_CONFIGS']:
                    try:
                        processed = self._preprocess_image(img, strategy['preprocess'])
                        data = pytesseract.image_to_data(
                            processed,
                            config=strategy['config'],
                            output_type=pytesseract.Output.DICT
                        )
                        
                        temp_result = {'text': [], 'boxes': [], 'confidences': []}
                        for i in range(len(data['text'])):
                            if int(data['conf'][i]) > 60:  # Confidence threshold
                                word = data['text'][i].strip()
                                if word:
                                    temp_result['text'].append(word)
                                    temp_result['boxes'].append([
                                        data['left'][i],
                                        data['top'][i],
                                        data['left'][i] + data['width'][i],
                                        data['top'][i] + data['height'][i]
                                    ])
                                    temp_result['confidences'].append(float(data['conf'][i]))
                        
                        # Keep the result with most high-confidence elements
                        if len(temp_result['text']) > len(result['text']):
                            result = temp_result
                            
                    except Exception as e:
                        logger.warning(f"OCR strategy failed: {str(e)}")
                        continue
                
                # Post-process merged results
                result['text'], result['boxes'] = self._merge_text_elements(result['text'], result['boxes'])
                
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            
        return result

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Union[List[str], List[List[int]]]]:
        """Hybrid PDF extraction with layout analysis"""
        result = {'text': [], 'boxes': []}
        
        try:
            # First attempt: Native PDF extraction
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        words = page.extract_words(
                            extra_attrs=["size", "fontname"],
                            keep_blank_chars=False,
                            use_text_flow=True,
                            x_tolerance=3,
                            y_tolerance=3
                        )
                        
                        for word in words:
                            if word['text'].strip():
                                result['text'].append(word['text'].strip())
                                result['boxes'].append([
                                    word['x0'],
                                    word['top'],
                                    word['x1'],
                                    word['bottom']
                                ])
                                
                if not result['text']:
                    raise ValueError("No text extracted via PDFPlumber")
                    
            except Exception as e:
                logger.info(f"Falling back to OCR: {str(e)}")
                images = convert_from_path(
                    pdf_path,
                    dpi=400,
                    poppler_path=self.poppler_path,
                    first_page=1,
                    last_page=min(5, len(convert_from_path(pdf_path))),  # Process max 5 pages
                    grayscale=True,
                    thread_count=4
                )
                
                for img in images:
                    ocr_result = self.extract_from_image(img)
                    result['text'].extend(ocr_result['text'])
                    result['boxes'].extend(ocr_result['boxes'])
                
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            
        return result

    def extract_from_docx(self, docx_path: str) -> Dict[str, Union[List[str], List[List[int]]]]:
        """DOCX extraction with structural awareness"""
        result = {'text': [], 'boxes': []}
        
        try:
            doc = docx.Document(docx_path)
            y_offset = 0
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    words = re.findall(r'\S+', text)
                    result['text'].extend(words)
                    
                    # Simulate layout boxes for downstream processing
                    for i, word in enumerate(words):
                        result['boxes'].append([
                            i * 100,  # x0
                            y_offset,  # y0
                            (i + 1) * 100,  # x1
                            y_offset + 20  # y1
                        ])
                    
                    y_offset += 24  # Line spacing
            
        except Exception as e:
            logger.error(f"DOCX processing failed: {str(e)}")
            
        return result


class DocumentQA:
    def __init__(self):
        try:
            self.processor = LayoutLMv3Processor.from_pretrained(app.config['MODEL_NAME'])
            self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(app.config['MODEL_NAME'])
            self.doc_processor = DocumentProcessor()
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def is_counting_question(self, question: str) -> bool:
        """Detect if question requires counting"""
        question_lower = question.lower()
        counting_phrases = [
            "how many", "number of", "count of", 
            "total number", "how much", "quantity of"
        ]
        return any(phrase in question_lower for phrase in counting_phrases)

    def extract_search_terms(self, question: str) -> List[str]:
        """Extract key terms from question"""
        stop_words = {
            "what", "which", "how", "many", "much", "number",
            "of", "are", "there", "the", "a", "an", "is", "in"
        }
        
        stem_map = {
            "projects": "project", "items": "item",
            "elements": "element", "sections": "section"
        }
        
        words = re.findall(r'\b\w+\b', question.lower())
        terms = []
        for word in words:
            if word not in stop_words and len(word) > 2:
                terms.append(stem_map.get(word, word))
        return list(set(terms))

    def find_relevant_context(self, text: List[str], terms: List[str], window_size: int = 5) -> str:
        """Locate most relevant text portion"""
        if not terms:
            return " ".join(text[:200]) + ("..." if len(text) > 200 else "")
        
        scores = np.zeros(len(text))
        for i, word in enumerate(text):
            word_lower = word.lower()
            for term in terms:
                if term in word_lower:
                    start = max(0, i - window_size)
                    end = min(len(text), i + window_size + 1)
                    scores[start:end] += 1 / (abs(np.arange(start, end) - i) + 1)
        
        if np.max(scores) > 0:
            best_pos = np.argmax(scores)
            start = max(0, best_pos - window_size)
            end = min(len(text), best_pos + window_size + 1)
            return " ".join(text[start:end])
        
        return " ".join(text[:200]) + ("..." if len(text) > 200 else "")

    def normalize_boxes(self, boxes: List[List[int]], image_size: Tuple[int, int]) -> List[List[int]]:
        """Convert boxes to LayoutLMv3 expected format"""
        width, height = image_size
        normalized = []
        for box in boxes:
            try:
                if len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
                    normalized.append([
                        int(1000 * box[0] / max(width, 1)),
                        int(1000 * box[1] / max(height, 1)),
                        int(1000 * box[2] / max(width, 1)),
                        int(1000 * box[3] / max(height, 1))
                    ])
                else:
                    normalized.append([0, 0, 1000, 1000])
            except (TypeError, ValueError):
                normalized.append([0, 0, 1000, 1000])
        return normalized

    def answer_question(self, file_path: str, question: str, image: Optional[Image.Image] = None) -> str:
        """
        Complete QA pipeline with enhanced error handling
        
        Args:
            file_path: Path to document
            question: User question
            image: Optional pre-loaded PIL Image
            
        Returns:
            Answer string with confidence indication
        """
        try:
            # Determine file type and process
            ext = file_path.rsplit('.', 1)[-1].lower()
            
            if ext in {'png', 'jpg', 'jpeg'}:
                result = self.doc_processor.extract_from_image(file_path)
                if image is None:
                    with Image.open(file_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        image = img.copy()
            elif ext == 'pdf':
                result = self.doc_processor.extract_from_pdf(file_path)
                if image is None:
                    images = convert_from_path(file_path, dpi=300, first_page=1, last_page=1)
                    image = images[0] if images else Image.new('RGB', (1000, 1000), (255, 255, 255))
            elif ext == 'docx':
                result = self.doc_processor.extract_from_docx(file_path)
                image = Image.new('RGB', (1000, 1000), (255, 255, 255))
            else:
                raise ValueError("Unsupported file format")
            
            words = result['text']
            boxes = result['boxes']
            
            if not words or len(words) < 3:
                return "Could not extract sufficient text from document"
            
            # Handle counting questions
            if self.is_counting_question(question):
                terms = self.extract_search_terms(question)
                if terms:
                    term_counts = defaultdict(int)
                    for word in words:
                        word_lower = word.lower()
                        for term in terms:
                            if term in word_lower:
                                term_counts[term] += 1
                    
                    if term_counts:
                        count_report = ", ".join(
                            f"{count} {term}(s)" 
                            for term, count in term_counts.items()
                        )
                        return f"Found: {count_report}"
                    return "No matching items found"
            
            # Prepare model inputs
            try:
                inputs = self.processor(
                    image=image,
                    text=question,
                    word_labels=words,
                    boxes=self.normalize_boxes(boxes, image.size if image else (1000, 1000)),
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    verbose=False
                )
            except Exception as e:
                logger.warning(f"Input prep failed: {str(e)}")
                return "Relevant content: " + self.find_relevant_context(words, self.extract_search_terms(question))
            
            # Run inference
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    start_scores = outputs.start_logits
                    end_scores = outputs.end_logits
                    
                    # Apply confidence threshold
                    confidence_threshold = 0.5
                    start_probs = torch.softmax(start_scores, dim=-1)[0]
                    end_probs = torch.softmax(end_scores, dim=-1)[0]
                    
                    if torch.max(start_probs) > confidence_threshold and torch.max(end_probs) > confidence_threshold:
                        answer_start = torch.argmax(start_scores)
                        answer_end = torch.argmax(end_scores)
                        answer = self.processor.decode(
                            inputs.input_ids[0][answer_start:answer_end+1],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        if answer:
                            return f"Answer: {answer.strip()} (confidence: {max(torch.max(start_probs).item(), torch.max(end_probs).item()):.2f})"
            except Exception as e:
                logger.warning(f"Inference failed: {str(e)}")
            
            # Fallback to relevant context
            return "Most relevant content: " + self.find_relevant_context(words, self.extract_search_terms(question))
                
        except Exception as e:
            logger.error(f"QA processing error: {str(e)}")
            if 'words' in locals() and words:
                return "Analysis failed. Partial content: " + self.find_relevant_context(words, [])
            return "Could not process the document"


# Initialize QA system
qa_system = DocumentQA()

def is_allowed_file(filename: str) -> bool:
    """Check if the file extension is permitted"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_remove(filepath: str, max_retries: int = 3, base_delay: float = 0.5) -> bool:
    """Safely remove a file with retries"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            logger.warning(f"File removal attempt {attempt + 1} failed: {str(e)}")
            time.sleep(delay)
    logger.error(f"Failed to remove file after {max_retries} attempts: {filepath}")
    return False

@app.route('/', methods=['GET', 'POST'])
def handle_document_qa():
    """Document QA endpoint with comprehensive error handling"""
    answer = None
    filepath = None
    
    try:
        if request.method == 'POST':
            # Validate request
            if 'file' not in request.files:
                return render_template('index.html', answer="Error: No file uploaded")
            
            file = request.files['file']
            if not file or file.filename == '':
                return render_template('index.html', answer="Error: No file selected")
            
            question = request.form.get('question', '').strip()
            if not question:
                return render_template('index.html', answer="Error: Please enter a question")
            
            if not is_allowed_file(file.filename):
                allowed_types = ", ".join(sorted(app.config['ALLOWED_EXTENSIONS']))
                return render_template('index.html', 
                                    answer=f"Error: Unsupported file type. Allowed: {allowed_types}")
            
            # Secure file handling
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                # Process with timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(qa_system.answer_question, filepath, question)
                    answer = future.result(timeout=app.config['PROCESSING_TIMEOUT'])
                
            except concurrent.futures.TimeoutError:
                answer = "Error: Processing took too long. Try a smaller document."
            except MemoryError:
                answer = "Error: Document too large. Try a smaller file."
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                answer = f"Error: Could not process document - {str(e)}"
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        answer = "Error: An unexpected error occurred"
    finally:
        if filepath and os.path.exists(filepath):
            safe_remove(filepath)
    
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        raise