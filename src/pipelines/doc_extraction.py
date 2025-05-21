from pathlib import Path
from typing import Any, Dict, List, Literal, Union

from config.logger import log
from config.schema import EPCCertificateDetails, PropertyExposeDetails
from langchain_core.messages import HumanMessage
from pipelines.utils import get_llm, get_prompt_and_parser, load_pdf_images
from pydantic import ValidationError


class ExtractionError(Exception):
    """Custom exception for errors during data extraction or parsing.

    This exception is raised when the extraction pipeline encounters issues such as
    failure to load images, LLM initialization errors, unexpected LLM response
    formats, or Pydantic validation errors during the parsing of LLM output.
    """

    pass


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

    The pipeline involves these main steps:
    1. Load specified pages from the PDF as base64 encoded images.
    2. Initialize the appropriate Large Language Model (LLM) based on the
       provider and model name.
    3. Prepare the prompt and parser specific to the document type.
    4. Construct a multimodal message containing the instructional prompt and
       the document images.
    5. Invoke the LLM with the message.
    6. Parse the LLM's response using the Pydantic parser to get structured data.
    7. Handle potential errors at each step, including image loading issues,
       LLM errors, and parsing/validation errors.

    Args:
        file_path (Path): The path to the PDF file.
        document_type (Literal["property_expose", "epc_certificate"]):
            The type of document to extract information from.
        llm_provider (Literal["openai", "gemini"]): The LLM provider to use.
        llm_model (str): The specific LLM model to use.
        page_numbers (Union[int, List[int], None], optional): An integer or list
            of integers specifying the page(s) to process. If None, all pages
            are processed. Defaults to None.

    Returns:
        Union[PropertyExposeDetails, EPCCertificateDetails, None]: An instance of
        PropertyExposeDetails or EPCCertificateDetails containing the extracted
        information, or None if a critical error occurs that prevents extraction.

    Raises:
        ExtractionError: If there are issues during the extraction or parsing
                         process, such as LLM response errors or validation
                         failures.
    """
    # 1. Load PDF pages as base64 images.
    base64_images = load_pdf_images(file_path, page_numbers=page_numbers)
    if not base64_images:
        log.error(f"Could not load images from {file_path}")
        return None

    if isinstance(base64_images, str):
        base64_images = [base64_images]  # Ensure it's a list for iteration

    # 2. Initialize the LLM.
    try:
        llm = get_llm(provider=llm_provider, model_name=llm_model)
    except ValueError as e:
        log.error(f"Error initializing LLM: {e}")
        return None

    # 3. Get the appropriate prompt and parser for the document type.
    prompt, parser, _ = get_prompt_and_parser(document_type)

    # 4. Construct the multimodal message for the LLM.
    message_content_parts: List[Union[str, Dict[str, Any]]] = []

    # 4a. Add the text prompt part.
    text_part: Dict[str, Any] = {"type": "text", "text": prompt.format()}
    message_content_parts.append(text_part)

    # 4b. Add image parts for each loaded page.
    for img_data in base64_images:
        image_part: Dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_data}"},
        }
        message_content_parts.append(image_part)

    message = HumanMessage(content=message_content_parts)

    # 5. Invoke the LLM and parse the response.
    try:
        log.info(
            f"Sending {len(base64_images)} image(s) for document type: {document_type} "
            f"using {llm_provider} model: {llm_model} to LLM for file: {file_path}"
        )
        response = llm.invoke([message])

        # 5a. Process the LLM response.
        if hasattr(response, "content") and isinstance(response.content, str):
            extracted_data: Union[PropertyExposeDetails, EPCCertificateDetails] = (
                parser.parse(response.content)
            )
            log.info(
                f"Successfully extracted data for document type: {document_type} from file: {file_path}"
            )
            return extracted_data
        else:
            log.warning(
                f"LLM response content is not in the expected format for file: {file_path}. Response: {response}"
            )
            raise ExtractionError(
                f"LLM response content not in expected format for {file_path.name}"
            )

    # 6. Handle known exceptions during LLM invocation or parsing.
    except ValidationError as ve:
        error_details = ve.errors()
        log.error(
            f"Pydantic ValidationError for file: {file_path.name}: {error_details}"
        )
        raise ExtractionError(
            f"Failed to parse {parser.pydantic_object.__name__} from completion for {file_path.name}. Got: {ve.error_count()} validation errors. First error: {error_details[0] if error_details else 'N/A'}"
        ) from ve
    except Exception as e:
        log.error(
            f"Unhandled error during extraction or parsing for file: {file_path.name}: {e}",
            exc_info=True,
        )
        raise ExtractionError(
            f"An unexpected error occurred during extraction for {file_path.name}: {e}"
        ) from e
