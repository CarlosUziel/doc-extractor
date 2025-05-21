from pathlib import Path
from typing import Any, Dict, List, Literal, Union

from langchain_core.messages import HumanMessage

from config.logger import log  # Import the logger
from config.schema import EPCCertificateDetails, PropertyExposeDetails
from pipelines.utils import get_llm, get_prompt_and_parser, load_pdf_images


def run_extraction_pipeline(
    file_path: Path,
    document_type: Literal["property_expose", "epc_certificate"],
    llm_provider: Literal["openai", "gemini"],
    llm_model: str,
    page_numbers: Union[int, List[int], None] = None,
) -> Union[PropertyExposeDetails, EPCCertificateDetails, None]:
    """
    Loads a PDF as image(s), selects an LLM based on provider and model,
    and extracts information using multimodal capabilities.

    Args:
        file_path: The path to the PDF file.
        document_type: The type of document to extract information from.
                       Can be "property_expose" or "epc_certificate".
        llm_provider: The LLM provider to use. Can be "openai" or "gemini".
        llm_model: The specific LLM model to use.
        page_numbers: Optional. An integer or list of integers specifying the page(s)
                      to process. If None, all pages are processed.

    Returns:
        An instance of PropertyExposeDetails or EPCCertificateDetails containing the
        extracted information, or None if an error occurs.
    """
    # Load PDF pages as base64 encoded images
    base64_images = load_pdf_images(file_path, page_numbers=page_numbers)
    if not base64_images:
        print(f"Could not load images from {file_path}")
        log.error(f"Could not load images from {file_path}")  # Log error
        return None

    # Ensure base64_images is a list (it should be if load_pdf_images works correctly)
    if isinstance(base64_images, str):
        base64_images = [base64_images]

    try:
        llm = get_llm(provider=llm_provider, model_name=llm_model)
    except ValueError as e:
        print(f"Error initializing LLM: {e}")
        log.error(f"Error initializing LLM: {e}")  # Log error
        return None

    prompt, parser, _ = get_prompt_and_parser(document_type)

    message_content_parts: List[Union[str, Dict[str, Any]]] = []

    text_part: Dict[str, Any] = {"type": "text", "text": prompt.format()}
    message_content_parts.append(text_part)

    for img_data in base64_images:
        image_part: Dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_data}"},
        }
        message_content_parts.append(image_part)

    message = HumanMessage(content=message_content_parts)

    try:
        log.info(
            f"Sending {len(base64_images)} image(s) for document type: {document_type} "
            f"using {llm_provider} model: {llm_model} to LLM for file: {file_path}"
        )  # Log before sending to LLM
        response = llm.invoke([message])

        if hasattr(response, "content") and isinstance(response.content, str):
            extracted_data: Union[PropertyExposeDetails, EPCCertificateDetails] = (
                parser.parse(response.content)
            )
            log.info(
                f"Successfully extracted data for document type: {document_type} from file: {file_path}"
            )  # Log success
            return extracted_data
        else:
            print(f"LLM response content is not in the expected format: {response}")
            log.warning(
                f"LLM response content is not in the expected format for file: {file_path}. Response: {response}"
            )  # Log warning
            return None

    except Exception as e:
        print(f"Error during extraction or parsing: {e}")
        log.error(
            f"Error during extraction or parsing for file: {file_path}: {e}"
        )  # Log error
        return None
