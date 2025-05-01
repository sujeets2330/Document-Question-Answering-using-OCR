import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pdfplumber
import docx
import re
import cv2
import numpy as np
from typing import Tuple, List, Dict, Union
from pdf2image import convert_from_path
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