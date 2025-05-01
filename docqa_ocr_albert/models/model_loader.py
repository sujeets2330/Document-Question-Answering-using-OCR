from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
import logging

logger = logging.getLogger(__name__)

def load_qa_model(model_name="microsoft/layoutlmv3-base"):
    try:
        processor = LayoutLMv3Processor.from_pretrained(model_name)
        model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
        logger.info("Model and processor loaded successfully")
        return processor, model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e  # Corrected this line
