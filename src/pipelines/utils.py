import base64
import io
from pathlib import Path
from typing import List, Literal, Type, Union

import pymupdf as fitz
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, SecretStr

from config.prompts import EXTRACT_INFO_PROMPT
from config.schema import EPCCertificateDetails, PropertyExposeDetails
from config.settings import settings


def load_pdf_images(
    file_path: Path,
    page_numbers: Union[int, List[int], None] = None,
) -> Union[List[str], None]:
    """Loads a PDF document and returns a list of base64 encoded images for specified pages.

    This function opens a PDF document at the given file path and converts the specified
    pages into base64-encoded PNG images. If no page numbers are specified, all pages
    in the PDF will be processed.

    Args:
        file_path: Path object pointing to the PDF file to process.
        page_numbers: Optional specification of which pages to process. Can be:
            - None: Process all pages in the PDF
            - int: Process only the specified page number
            - List[int]: Process only the specified page numbers
            Note that page numbers are 1-indexed (first page is page 1).

    Returns:
        A list of base64-encoded PNG images as strings, or None if an error occurred
        or no valid images could be generated.

    Raises:
        No exceptions are raised directly; errors are caught and logged with print().
    """
    if page_numbers is None:
        try:
            pdf_document = fitz.open(file_path)
            num_pages = len(pdf_document)
            pdf_document.close()
            page_numbers = list(range(1, num_pages + 1))
        except Exception as e:
            print(f"Error opening PDF to get page count: {e}")
            return None
    elif isinstance(page_numbers, int):
        page_numbers = [page_numbers]

    base64_images = []
    try:
        pdf_document = fitz.open(file_path)
        for page_num in page_numbers:
            if 1 <= page_num <= len(pdf_document):
                img_data = pdf_page_to_base64(pdf_document, page_num)
                if img_data:
                    base64_images.append(img_data)
            else:
                print(
                    f"Warning: Page number {page_num} is out of range for {file_path}"
                )
        pdf_document.close()
        return base64_images if base64_images else None
    except Exception as e:
        print(f"Error processing PDF as images: {e}")
        return None


def pdf_page_to_base64(
    pdf_document: fitz.Document, page_number: int
) -> Union[str, None]:
    """Converts a specific page of a PDF document to a base64 encoded PNG image.

    This function takes an open PDF document and a page number, renders the page
    as a pixmap, converts it to a PIL Image, and then encodes it as a base64 string.

    Args:
        pdf_document: An open PyMuPDF (fitz) Document object.
        page_number: The page number to convert (1-indexed, first page is page 1).

    Returns:
        A base64-encoded PNG image as a string, or None if conversion failed.

    Raises:
        No exceptions are raised directly; errors are caught and logged with print().
    """
    try:
        page = pdf_document.load_page(page_number - 1)
        pix = page.get_pixmap()  # type: ignore
        if not isinstance(pix.samples, bytes):
            print(
                f"Error: pix.samples is not bytes for page {page_number} in {pdf_document.name}. Type: {type(pix.samples)}"
            )
            return None
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        if not isinstance(img_bytes, bytes):
            print(
                f"Error: buffer.getvalue() did not return bytes for page {page_number} in {pdf_document.name}. Type: {type(img_bytes)}"
            )
            return None
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"Error converting page {page_number} to base64: {e}")
        return None


def get_llm(provider: Literal["openai", "gemini"], model_name: str) -> BaseChatModel:
    """Selects and returns an LLM instance based on the specified provider and model name.

    This function initializes a language model from either OpenAI or Google Gemini based on
    the specified provider. It validates that the required API keys are configured in the
    application settings and handles potential initialization errors.

    Args:
        provider: The LLM provider to use, must be either "openai" or "gemini".
        model_name: The specific model name/identifier to use for the selected provider.
            For OpenAI, examples include "gpt-4o", "gpt-4o-mini", etc.
            For Gemini, examples include "gemini-pro", "gemini-1.5-pro", etc.

    Returns:
        An initialized instance of BaseChatModel from the specified provider.

    Raises:
        ValueError: If the API key for the provider is not configured in settings,
                   if the provider is unsupported (not "openai" or "gemini"),
                   or if the model initialization fails for any reason.
    """
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key not configured in settings. "
                "Please set OPENAI_API_KEY in your .env file."
            )
        try:
            return ChatOpenAI(
                api_key=SecretStr(settings.OPENAI_API_KEY), model=model_name
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize OpenAI model '{model_name}'. "
                "Ensure the model name is correct and accessible with "
                f"your API key. Error: {e}"
            )
    elif provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "Gemini API key not configured in settings. "
                "Please set GEMINI_API_KEY in your .env file."
            )
        try:
            return ChatGoogleGenerativeAI(
                model=model_name, google_api_key=SecretStr(settings.GEMINI_API_KEY)
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Gemini model '{model_name}'. "
                "Ensure the model name is correct and accessible "
                f"with your API key. Error: {e}"
            )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Must be 'openai' or 'gemini'."
        )


DOCUMENT_TYPE_SCHEMAS = {
    "property_expose": PropertyExposeDetails,
    "epc_certificate": EPCCertificateDetails,
}


def get_prompt_and_parser(
    document_type: Literal["property_expose", "epc_certificate"],
) -> tuple[PromptTemplate, PydanticOutputParser, Type[BaseModel]]:
    """Returns the appropriate prompt template and parser for the given document type.

    This function selects the appropriate Pydantic model schema based on the document type,
    creates a PydanticOutputParser for that schema, and constructs a PromptTemplate that
    instructs the LLM to analyze images according to the schema.

    Args:
        document_type: The type of document to create a prompt and parser for. Must be one of:
            - "property_expose": For property exposition documents
            - "epc_certificate": For Energy Performance Certificate documents

    Returns:
        A tuple containing:
            - PromptTemplate: A template with instructions for the LLM
            - PydanticOutputParser: A parser configured with the appropriate schema
            - Type[BaseModel]: The Pydantic model class used for the schema

    Raises:
        ValueError: If an unsupported document type is provided
    """
    schema = DOCUMENT_TYPE_SCHEMAS.get(document_type)
    if not schema:
        raise ValueError(f"Unsupported document type: {document_type}")

    parser = PydanticOutputParser(pydantic_object=schema)

    prompt = PromptTemplate(
        template=EXTRACT_INFO_PROMPT,
        input_variables=[],  # Images are passed via HumanMessage
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser, schema
