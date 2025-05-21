import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

import fitz
import streamlit as st  # type: ignore
from config.logger import log
from PIL import Image
from pipelines.doc_extraction import run_extraction_pipeline

st.title("Document Extractor")

# 1. A button to upload a PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# 2. A dropdown menu to select the type of document
document_type_options = {
    "epc_certificate": "EPC Certificate",
    "property_expose": "Property Exposé",
}
# Determine the default index for "EPC Certificate"
default_document_type_index = list(document_type_options.values()).index(
    "EPC Certificate"
)

selected_document_type_display = st.selectbox(
    "Select document type",
    options=list(document_type_options.values()),
    index=default_document_type_index,
    placeholder="Choose an option",
)


def _get_key_from_display_name(
    display_name: Optional[str], options: Dict[str, str]
) -> Optional[str]:
    """Gets the key from a display name in a dictionary of options.

    Args:
        display_name (Optional[str]): The display name to search for.
        options (Dict[str, str]): A dictionary where keys are internal values and
                                  values are display names.

    Returns:
        Optional[str]: The key corresponding to the display name, or None if not
                       found.
    """
    # 1. Handle None input for display_name.
    if display_name is None:
        return None
    # 2. Iterate through options to find the matching display name.
    for key, value in options.items():
        if value == display_name:
            return key
    # 3. Return None if no match is found.
    return None


# Map display name back to actual value
selected_document_type_key = cast(
    Optional[Literal["epc_certificate", "property_expose"]],
    _get_key_from_display_name(selected_document_type_display, document_type_options),
)

# LLM Provider and Model Selection
llm_provider_options = {
    "openai": "OpenAI",
    "gemini": "Gemini",
}

# Determine the default index for "OpenAI"
default_llm_provider_index = list(llm_provider_options.values()).index("Gemini")

llm_model_options = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-vision-preview"],
    "gemini": [
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-05-20",
    ],
}

selected_llm_provider_display = st.selectbox(
    "Select LLM Provider",
    options=list(llm_provider_options.values()),
    index=default_llm_provider_index,
    placeholder="Choose a provider",
)

selected_llm_provider_key = cast(
    Optional[Literal["openai", "gemini"]],
    _get_key_from_display_name(selected_llm_provider_display, llm_provider_options),
)

available_models: list[str] = []
if selected_llm_provider_key:
    available_models = llm_model_options[selected_llm_provider_key]

# Determine the default model index if the provider is OpenAI
default_llm_model_index = None
if selected_llm_provider_key == "openai" and available_models:
    default_llm_model_index = 0

selected_llm_model = st.selectbox(
    "Select LLM Model",
    options=available_models,
    index=default_llm_model_index,
    placeholder="Choose a model"
    if selected_llm_provider_key
    else "Select a provider first",
    disabled=not selected_llm_provider_key,
)

# 3. A submit button
submit_button = st.button("Submit")

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "processed_pil_images" not in st.session_state:
    st.session_state.processed_pil_images = []


def display_ocr_previews(
    data: Any, pil_images: list[Image.Image], current_path: str = ""
) -> None:
    """Recursively traverses the data and displays OCR previews for fields with bounding boxes.

    This function iterates through a (potentially nested) dictionary representing
    extracted document data. When it encounters a field ending with "_bbox" and its
    corresponding data field, it attempts to display the field name, its value,
    the bounding box coordinates, and a cropped image preview from the original
    document. It handles multi-page documents by looking for a `page_number`
    within the bounding box data.

    Args:
        data (Any): The data structure (likely a nested dictionary) containing
                    extracted information and bounding box details.
        pil_images (list[Image.Image]): A list of PIL.Image.Image objects
                                        representing the pages of the processed
                                        document.
        current_path (str, optional): The current path in the nested data
                                      structure, used for generating descriptive
                                      field names. Defaults to "".
    """
    # 1. Check if images are available for preview.
    if not pil_images:
        st.warning("No processed images available for OCR preview.")
        return

    # 2. Process dictionary data.
    if isinstance(data, dict):
        for key, value in data.items():
            # 2a. Check for bounding box fields.
            if key.endswith("_bbox") and value is not None:
                actual_field_key = key[:-5]
                # 2b. Ensure the corresponding data field exists.
                if actual_field_key in data:
                    field_value = data[actual_field_key]
                    bbox_coords = value

                    full_field_path = (
                        f"{current_path}{actual_field_key}"
                        if current_path
                        else actual_field_key
                    )

                    st.markdown("---")

                    # 2c. Determine the correct image page for cropping.
                    image_to_crop = None
                    page_number_info = ""
                    if (
                        isinstance(bbox_coords, dict)
                        and "page_number" in bbox_coords
                        and bbox_coords["page_number"] is not None
                    ):
                        page_idx = bbox_coords["page_number"]
                        if 0 <= page_idx < len(pil_images):
                            image_to_crop = pil_images[page_idx]
                            page_number_info = f" (Page {page_idx + 1})"
                        else:
                            st.caption(
                                f"Page index {page_idx} out of range for field "
                                f"{full_field_path}."
                            )
                            image_to_crop = pil_images[0]  # Fallback to first page
                            page_number_info = " (Page 1 - Fallback)"
                    else:
                        # Fallback to the first page if page_number is not specified or invalid.
                        image_to_crop = pil_images[0]
                        page_number_info = " (Page 1 - Fallback)"

                    # 2d. Display field information and preview in columns.
                    col1, col2, col3, col4 = st.columns([3, 3, 3, 2])

                    with col1:
                        st.markdown(
                            f"**Field:**\n`{full_field_path}`{page_number_info}"
                        )
                    with col2:
                        st.markdown(f"**Value:**\n`{field_value}`")
                    with col3:
                        if isinstance(bbox_coords, dict) and all(
                            k in bbox_coords for k in ["xmin", "ymin", "xmax", "ymax"]
                        ):
                            st.markdown(
                                f"**BBox (xmin,ymin,xmax,ymax):**\n"
                                f"`({bbox_coords['xmin']},{bbox_coords['ymin']},"
                                f"{bbox_coords['xmax']},{bbox_coords['ymax']})`"
                            )
                            # 2e. Crop and display image preview if possible.
                            if image_to_crop:
                                try:
                                    img_width, img_height = image_to_crop.size
                                    padding = 10  # Padding around the bounding box
                                    # Scale normalized coordinates (0-1000) to absolute pixel values
                                    xmin_abs = max(
                                        0,
                                        int((bbox_coords["xmin"] / 1000) * img_width)
                                        - padding,
                                    )
                                    ymin_abs = max(
                                        0,
                                        int((bbox_coords["ymin"] / 1000) * img_height)
                                        - padding,
                                    )
                                    xmax_abs = min(
                                        img_width,
                                        int((bbox_coords["xmax"] / 1000) * img_width)
                                        + padding,
                                    )
                                    ymax_abs = min(
                                        img_height,
                                        int((bbox_coords["ymax"] / 1000) * img_height)
                                        + padding,
                                    )

                                    int_bbox = (
                                        xmin_abs,
                                        ymin_abs,
                                        xmax_abs,
                                        ymax_abs,
                                    )
                                    cropped_image = image_to_crop.crop(int_bbox)
                                    with col4:
                                        st.markdown("**Preview:**")
                                        st.image(cropped_image, width=150)
                                except Exception as crop_e:
                                    with col4:
                                        st.caption(f"Crop error: {crop_e}")
                            else:
                                with col4:
                                    st.caption("Image for preview not available.")
                        else:
                            st.markdown("**BBox:**\nInvalid format")
            # 2f. Recursively call for nested dictionaries.
            elif isinstance(value, dict):
                new_path = f"{current_path}{key}." if current_path else f"{key}."
                display_ocr_previews(value, pil_images, new_path)
            # 2g. Recursively call for items in lists if they are dictionaries.
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        new_path = (
                            f"{current_path}{key}[{i}]."
                            if current_path
                            else f"{key}[{i}]."
                        )
                        display_ocr_previews(item, pil_images, new_path)


def date_converter(o: Any) -> str:
    """Converts date objects to ISO format strings for JSON serialization.

    Args:
        o (Any): The object to convert.

    Returns:
        str: The ISO format string if the object is a date, otherwise raises TypeError.

    Raises:
        TypeError: If the object is not a date object and cannot be serialized.
    """
    # 1. Check if the object is an instance of date.
    if isinstance(o, date):
        # 2. Convert date to ISO format string.
        return o.isoformat()
    # 3. Raise TypeError if the object is not a date.
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


if submit_button:
    if uploaded_file is None:
        st.error("Please upload a PDF file.")
    elif selected_document_type_key is None:
        st.error("Please select a document type.")
    elif selected_llm_provider_key is None:
        st.error("Please select an LLM provider.")
    elif selected_llm_model is None:
        st.error("Please select an LLM model.")
    else:
        tmp_file_path: Path | None = None
        st.session_state.processed_pil_images = []
        with st.spinner("Processing document..."):
            try:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = Path(tmp_file.name)

                # Convert PDF pages to PIL Images for preview
                if tmp_file_path:
                    try:
                        pdf_doc = fitz.open(tmp_file_path)
                        for page_num in range(len(pdf_doc)):
                            page = pdf_doc.load_page(page_num)
                            pix = page.get_pixmap()  # type: ignore
                            img = Image.frombytes(
                                "RGB", (pix.width, pix.height), pix.samples
                            )  # type: ignore
                            st.session_state.processed_pil_images.append(img)
                        pdf_doc.close()
                        log.info(
                            f"Successfully converted "
                            f"{len(st.session_state.processed_pil_images)} "
                            f"page(s) to PIL Images for preview."
                        )
                    except Exception as img_e:
                        st.error(f"Error processing PDF for image preview: {img_e}")
                        log.error(f"Error converting PDF to PIL Images: {img_e}")
                        st.session_state.processed_pil_images = []

                # Call the document extraction pipeline
                result = run_extraction_pipeline(
                    file_path=tmp_file_path,
                    document_type=selected_document_type_key,
                    llm_provider=selected_llm_provider_key,
                    llm_model=selected_llm_model,
                )

                if result:
                    # Convert Pydantic model to dict, then to JSON string for display
                    st.session_state.extraction_result = json.dumps(
                        result.model_dump(), indent=2, default=date_converter
                    )
                    st.success("Extraction successful!")
                else:
                    st.session_state.extraction_result = (
                        "No data extracted or an error occurred."
                    )
                    st.error("Extraction failed or returned no data.")
                    log.warning(
                        f"Extraction failed or returned no data for file: "
                        f"{uploaded_file.name} and document type: "
                        f"{selected_document_type_key}"
                    )

            except Exception as e:
                st.session_state.extraction_result = f"An error occurred: {str(e)}"
                st.error(f"An error occurred during extraction: {str(e)}")
                log.error(
                    f"An error occurred during extraction for file: "
                    f"{uploaded_file.name if uploaded_file else 'N/A'}: {str(e)}",
                    exc_info=True,
                )
            finally:
                if tmp_file_path and tmp_file_path.exists():
                    try:
                        tmp_file_path.unlink()
                        log.info(
                            f"Successfully deleted temporary file: {tmp_file_path}"
                        )
                    except Exception as e_unlink:
                        log.error(
                            f"Error deleting temporary file {tmp_file_path}: {e_unlink}"
                        )

# Display the output in a clickable (but not editable) output box
if st.session_state.extraction_result:
    st.subheader("Extraction Output:")
    st.text_area(
        "Result", st.session_state.extraction_result, height=300, disabled=True
    )

    # OCR Field Previews Section
    if st.session_state.processed_pil_images:
        st.subheader("OCR Field Previews")
        st.markdown("""
        Below are previews of detected fields based on their bounding boxes.
        *Note: For multi-page documents, previews are currently generated from the **first page** only.
        A future enhancement could allow page selection or associate bounding boxes with specific pages.*
        """)
        try:
            parsed_result = json.loads(st.session_state.extraction_result)
            display_ocr_previews(parsed_result, st.session_state.processed_pil_images)
        except json.JSONDecodeError:
            st.error("Could not parse extraction result for OCR preview.")
            log.error("OCR Preview: JSONDecodeError parsing extraction_result.")
        except Exception as e_preview:
            st.error(f"An error occurred while generating OCR previews: {e_preview}")
            log.error(f"OCR Preview error: {e_preview}")
    elif submit_button:
        st.warning(
            "OCR Previews cannot be shown as document images could not be processed."
        )
