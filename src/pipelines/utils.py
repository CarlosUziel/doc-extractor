import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pymupdf as fitz
from config.logger import log
from config.prompts import EXTRACT_INFO_PROMPT
from config.schema import EPCCertificateDetails, PropertyExposeDetails
from config.settings import settings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, SecretStr


def load_pdf_images(
    file_path: Path,
    page_numbers: Union[int, List[int], None] = None,
) -> Union[List[str], None]:
    """Loads a PDF document and returns a list of base64 encoded images for specified pages.

    This function opens a PDF document at the given file path and converts the
    specified pages into base64-encoded PNG images. If no page numbers are
    specified, all pages in the PDF will be processed.

    Args:
        file_path (Path): Path object pointing to the PDF file to process.
        page_numbers (Union[int, List[int], None], optional): Specification of
            which pages to process. Can be:
            - None: Process all pages in the PDF.
            - int: Process only the specified page number (1-indexed).
            - List[int]: Process only the specified page numbers (1-indexed).
            Defaults to None.

    Returns:
        Union[List[str], None]: A list of base64-encoded PNG images as strings,
        or None if an error occurred or no valid images could be generated.
    """
    # 1. Determine the list of page numbers to process.
    actual_page_numbers: List[int]
    pdf_document_for_page_count: Optional[fitz.Document] = None
    try:
        if page_numbers is None:
            # 1a. If no page numbers are specified, get all pages from the PDF.
            pdf_document_for_page_count = fitz.open(file_path)
            num_pages = len(pdf_document_for_page_count)
            actual_page_numbers = list(range(1, num_pages + 1))
        elif isinstance(page_numbers, int):
            # 1b. If a single page number is given, convert it to a list.
            actual_page_numbers = [page_numbers]
        else:
            # 1c. If a list of page numbers is given, use it directly.
            actual_page_numbers = page_numbers
    except Exception as e:
        log.error(f"Error opening PDF to get page count for {file_path}: {e}")
        return None
    finally:
        if pdf_document_for_page_count:
            pdf_document_for_page_count.close()

    base64_images: List[str] = []
    pdf_document_for_processing: Optional[fitz.Document] = None
    try:
        # 2. Open the PDF document.
        pdf_document_for_processing = fitz.open(file_path)
        # 3. Iterate through the specified page numbers and convert each to base64.
        for page_num in actual_page_numbers:
            # 3a. Validate page number range.
            if 1 <= page_num <= len(pdf_document_for_processing):
                img_data = pdf_page_to_base64(pdf_document_for_processing, page_num)
                if img_data:
                    base64_images.append(img_data)
            else:
                log.warning(
                    f"Warning: Page number {page_num} is out of range for {file_path} "
                    f"(Total pages: {len(pdf_document_for_processing)}). Skipping this page."
                )
        # 4. Return the list of base64 images, or None if empty.
        return base64_images if base64_images else None
    except Exception as e:
        log.error(f"Error processing PDF {file_path} to images: {e}")
        return None
    finally:
        if pdf_document_for_processing:
            pdf_document_for_processing.close()


def pdf_page_to_base64(
    pdf_document: fitz.Document, page_number: int
) -> Union[str, None]:
    """Converts a specific page of a PDF document to a base64 encoded PNG image.

    This function takes an open PDF document and a 1-indexed page number,
    renders the page as a pixmap, converts it to a PIL Image, and then encodes
    it as a base64 string.

    Args:
        pdf_document (fitz.Document): An open PyMuPDF (fitz) Document object.
        page_number (int): The 1-indexed page number to convert.

    Returns:
        Union[str, None]: A base64-encoded PNG image as a string, or None if
        conversion failed.
    """
    try:
        # 1. Load the specified page (0-indexed for fitz).
        page = pdf_document.load_page(page_number - 1)
        # 2. Render the page to a pixmap.
        pix = page.get_pixmap()  # type: ignore
        # 3. Validate pixmap samples.
        if not isinstance(pix.samples, bytes):
            log.error(
                f"Error: pix.samples is not bytes for page {page_number} "
                f"in {pdf_document.name}. Type: {type(pix.samples)}"
            )
            return None
        # 4. Create a PIL Image from the pixmap.
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        # 5. Save the image to a BytesIO buffer as PNG.
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        # 6. Validate image bytes.
        if not isinstance(img_bytes, bytes):
            log.error(
                f"Error: buffer.getvalue() did not return bytes for page {page_number} "
                f"in {pdf_document.name}. Type: {type(img_bytes)}"
            )
            return None
        # 7. Encode the image bytes to base64 and decode to UTF-8 string.
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        log.error(
            f"Error converting page {page_number} of {pdf_document.name} to base64: {e}"
        )
        return None


def get_llm(provider: Literal["openai", "gemini"], model_name: str) -> BaseChatModel:
    """Selects and returns an LLM instance based on the specified provider and model name.

    This function initializes a language model from either OpenAI or Google Gemini
    based on the specified provider. It validates that the required API keys are
    configured in the application settings and handles potential initialization errors.

    Args:
        provider (Literal["openai", "gemini"]): The LLM provider to use.
        model_name (str): The specific model name/identifier to use for the
                          selected provider (e.g., "gpt-4o", "gemini-1.5-pro").

    Returns:
        BaseChatModel: An initialized instance of BaseChatModel from the specified
                       provider.

    Raises:
        ValueError: If the API key for the provider is not configured, if the
                    provider is unsupported, or if model initialization fails.
    """
    # 1. Handle OpenAI provider.
    if provider == "openai":
        # 1a. Check for OpenAI API key in settings.
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI API key not configured in settings. "
                "Please set OPENAI_API_KEY in your .env file."
            )
        try:
            # 1b. Initialize and return ChatOpenAI model.
            return ChatOpenAI(
                api_key=SecretStr(settings.openai_api_key), model=model_name
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize OpenAI model '{model_name}'. "
                "Ensure the model name is correct and accessible with "
                f"your API key. Error: {e}"
            )
    # 2. Handle Gemini provider.
    elif provider == "gemini":
        # 2a. Check for Gemini API key in settings.
        if not settings.gemini_api_key:
            raise ValueError(
                "Gemini API key not configured in settings. "
                "Please set GEMINI_API_KEY in your .env file."
            )
        try:
            # 2b. Initialize and return ChatGoogleGenerativeAI model.
            return ChatGoogleGenerativeAI(
                model=model_name, google_api_key=SecretStr(settings.gemini_api_key)
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Gemini model '{model_name}'. "
                "Ensure the model name is correct and accessible "
                f"with your API key. Error: {e}"
            )
    # 3. Handle unsupported provider.
    else:
        # This path should ideally not be reached if Literal types are enforced,
        # but as a safeguard:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Must be 'openai' or 'gemini'."
        )


DOCUMENT_TYPE_SCHEMAS: Dict[
    Literal["property_expose", "epc_certificate"], Type[BaseModel]
] = {
    "property_expose": PropertyExposeDetails,
    "epc_certificate": EPCCertificateDetails,
}


def get_prompt_and_parser(
    document_type: Literal["property_expose", "epc_certificate"],
) -> tuple[PromptTemplate, PydanticOutputParser[Any], Type[BaseModel]]:
    """Returns the appropriate prompt template and parser for the given document type.

    This function selects the appropriate Pydantic model schema based on the
    document type, creates a PydanticOutputParser for that schema, and constructs
    a PromptTemplate that instructs the LLM to analyze images according to the schema.

    Args:
        document_type (Literal["property_expose", "epc_certificate"]):
            The type of document to create a prompt and parser for.

    Returns:
        tuple[PromptTemplate, PydanticOutputParser[Any], Type[BaseModel]]: A tuple containing:
            - PromptTemplate: A template with instructions for the LLM.
            - PydanticOutputParser: A parser configured with the appropriate schema.
            - Type[BaseModel]: The Pydantic model class used for the schema.

    Raises:
        ValueError: If an unsupported document type is provided.
    """
    # 1. Get the schema class based on the document type.
    schema_class = DOCUMENT_TYPE_SCHEMAS.get(document_type)
    if not schema_class:
        # This case should ideally be prevented by Literal type hinting on document_type
        # but is kept as a safeguard.
        raise ValueError(f"Unsupported document type: {document_type}")

    # 2. Create a PydanticOutputParser for the schema.
    parser: PydanticOutputParser[Any] = PydanticOutputParser(
        pydantic_object=schema_class
    )

    # 3. Create a PromptTemplate with format instructions from the parser.
    prompt = PromptTemplate(
        template=EXTRACT_INFO_PROMPT,
        input_variables=[],  # Images are passed directly in the message content.
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    # 4. Return the prompt, parser, and schema class.
    return prompt, parser, schema_class
