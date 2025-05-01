import torch
import pytesseract
from PIL import Image, ImageFilter, ImageOps
import pdfplumber
import docx
import re
import cv2
import numpy as np
from typing import Tuple, List, Dict, Union, Optional
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, poppler_path: str = None):
        self.poppler_path = poppler_path
        self.ocr_configs = [
            {'preprocess': self._preprocess_image, 'config': r'--oem 3 --psm 6'},
            {'preprocess': lambda x: x.convert('L').filter(ImageFilter.SHARPEN), 'config': r'--oem 1 --psm 11'},
            {'preprocess': lambda x: ImageOps.autocontrast(x.convert('L')), 'config': r'--oem 3 --psm 4'}
        ]
        
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Advanced image preprocessing pipeline"""
        try:
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            return Image.fromarray(thresh)
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
                for strategy in self.ocr_configs:
                    try:
                        processed = strategy['preprocess'](img)
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
    def __init__(self, poppler_path: str = None, model_name: str = "microsoft/layoutlmv3-base"):
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
        self.doc_processor = DocumentProcessor(poppler_path)
        
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

    def process_document(self, file_path: str) -> Dict[str, Union[List[str], List[List[int]]]]:
        """Unified document processing with automatic format detection"""
        if file_path.lower().endswith('.pdf'):
            return self.doc_processor.extract_from_pdf(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self.doc_processor.extract_from_image(file_path)
        elif file_path.lower().endswith('.docx'):
            return self.doc_processor.extract_from_docx(file_path)
        else:
            raise ValueError("Unsupported file format")

    def answer_question(self, file_path: str, question: str, image: Optional[Image.Image] = None) -> str:
        """
        Complete QA pipeline with enhanced error handling
        
        Args:
            file_path: Path to document (PDF/image/DOCX)
            question: User question
            image: Optional pre-loaded PIL Image (for PDFs)
            
        Returns:
            Answer string with confidence indication
        """
        try:
            # Extract text and boxes
            result = self.process_document(file_path)
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
            
            # Load image if not provided (for PDFs)
            if image is None and file_path.lower().endswith('.pdf'):
                images = convert_from_path(file_path, dpi=300, first_page=1, last_page=1)
                image = images[0] if images else None
            
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
            if words:
                return "Analysis failed. Partial content: " + self.find_relevant_context(words, [])
            return "Could not process the document"


# Example Usage
if __name__ == "__main__":
    qa_system = DocumentQA()
    
    # For PDF/Image
    answer = qa_system.answer_question(
        file_path="Helpful Projects.pdf",
        question="How many ML projects are listed?"
    )
    print(answer)
    
    # For DOCX
    answer = qa_system.answer_question(
        file_path="document.docx",
        question="What is the main topic?"
    )
    print(answer)