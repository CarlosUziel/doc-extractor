import enum
from pathlib import Path
from typing import Literal

import typer
from config.logger import log
from pipelines.doc_extraction import run_extraction_pipeline
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

app = typer.Typer()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LlmProviderChoice(str, enum.Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


def sanitize_model_name(model_name: str) -> str:
    """Sanitizes a model name to be filesystem-friendly.

    Replaces characters like '/' and ':' with underscores to ensure compatibility
    with most filesystems when using the model name in paths.

    Args:
        model_name (str): The original model name string.

    Returns:
        str: A sanitized string suitable for use in file or directory names.
    """
    # 1. Replace characters that are problematic for file systems.
    return model_name.replace("/", "_").replace(":", "_")


@app.command()
def main(
    llm_provider: LlmProviderChoice = typer.Option(
        LlmProviderChoice.GEMINI,
        help="LLM provider.",
        case_sensitive=False,
    ),
    llm_model: str = typer.Option(
        "gemini-2.5-pro-preview-05-06",
        help="LLM model name (e.g., 'gpt-4o-mini', 'gemini-2.5-pro-preview-05-06').",
    ),
    data_base_path: Path = typer.Option(
        PROJECT_ROOT / ".data",
        help="Base path for input data.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    annotations_base_path: Path = typer.Option(
        PROJECT_ROOT / "annotations",
        help="Base path for output annotations.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    limit: int = typer.Option(
        None,  # Changed from None to allow typer to handle default correctly if not provided
        "--limit",
        "-l",
        help="Limit the number of files to process per document type.",
        min=1,
        show_default=False,  # Explicitly hide default if it's truly optional and no default value is desired
    ),
) -> None:
    """
    Processes PDF documents from specified input directories, extracts information
    using a configurable Large Language Model (LLM), and saves the structured
    output as JSON files.

    The script iterates through predefined document types (e.g., "epc_certificate",
    "property_expose") and processes PDF files found in corresponding
    subdirectories of the `data_base_path`. For each PDF, it calls an extraction
    pipeline and stores the resulting JSON data.

    Output Structure:
        The extracted JSON for each PDF is saved to:
        `annotations_base_path/{document_type}/{sanitized_model_name}/{original_pdf_stem}.json`
        where `sanitized_model_name` is a filesystem-friendly version of the
        `llm_model` name.

    Args:
        llm_provider (LlmProviderChoice): Specifies the LLM provider to use
            (e.g., "openai", "gemini"). Defaults to LlmProviderChoice.GEMINI.
        llm_model (str): Defines the specific model name for the chosen provider.
            Defaults to "gemini-2.5-pro-preview-05-06".
        data_base_path (Path): The root directory containing subfolders for
            different document types (e.g., ".data/EPC certificates",
            ".data/property expose"). Defaults to `PROJECT_ROOT / ".data"`.
        annotations_base_path (Path): The root directory where generated JSON
            annotations will be stored. Defaults to `PROJECT_ROOT / "annotations"`.
        limit (int, optional): An optional integer to restrict processing to the
            first N PDF files found within each document type's subdirectory,
            useful for testing. Defaults to None (no limit).

    Example Usage:
        python src/benchmark/generate_predictions.py \\
            --llm-provider gemini \\
            --llm-model gemini-2.5-pro-preview-05-06 \\
            --data-base-path .data \\
            --annotations-base-path annotations \\
            --limit 10
    """
    # 1. Log initial parameters.
    log.info(
        f"Starting prediction generation with provider: {llm_provider.value}, model: {llm_model}"
    )
    log.info(f"Input data path: {data_base_path}")
    log.info(f"Output annotations path: {annotations_base_path}")

    # 2. Define mapping from document type keys to data folder names.
    document_types_map: dict[Literal["epc_certificate", "property_expose"], str] = {
        "epc_certificate": "EPC certificates",
        "property_expose": "property expose",
    }

    # 3. Sanitize the model name for use in file paths.
    sanitized_model = sanitize_model_name(llm_model)

    # 4. Configure and initialize the Rich progress bar.
    custom_progress_columns = [
        SpinnerColumn(spinner_name="dots", style="magenta"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(
            bar_width=None,
            style="yellow",
            complete_style="green",
            finished_style="dim green",
        ),
        MofNCompleteColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ]
    progress_console = Console()

    # 5. Start the main processing loop with the progress bar.
    with Progress(*custom_progress_columns, console=progress_console) as progress:
        overall_task_description = f"Processing all document types for model: [bold blue]{sanitized_model}[/bold blue]"
        overall_task = progress.add_task(
            overall_task_description, total=len(document_types_map)
        )

        # 6. Iterate over each document type.
        for doc_type_key, data_folder_name in document_types_map.items():
            progress.update(
                overall_task,
                description=f"Overall Progress (Current: [bold cyan]{doc_type_key}[/])",
            )

            # 6a. Define paths for current document type.
            current_data_path = data_base_path / data_folder_name
            output_dir = annotations_base_path / doc_type_key / sanitized_model

            log.info(
                f"Processing document type: {doc_type_key} (from folder: {data_folder_name})"
            )
            log.info(f"Looking for PDFs in: {current_data_path}")
            log.info(f"Output will be saved to: {output_dir}")

            # 6b. Validate data directory.
            if not current_data_path.exists() or not current_data_path.is_dir():
                log.warning(
                    f"Data directory not found or is not a directory: {current_data_path}"
                )
                progress.update(overall_task, advance=1)
                continue

            # 6c. Create output directory if it doesn't exist.
            output_dir.mkdir(parents=True, exist_ok=True)

            # 6d. Collect PDF files to process.
            pdf_files_to_process = list(current_data_path.rglob("*.pdf"))

            if not pdf_files_to_process:
                log.warning(f"No PDF files found in {current_data_path}")
                progress.update(overall_task, advance=1)
                continue

            # 6e. Apply limit if specified.
            if limit is not None:
                pdf_files_to_process = pdf_files_to_process[:limit]

            if not pdf_files_to_process:
                log.warning(
                    f"No PDF files to process for {doc_type_key} after applying limit."
                )
                progress.update(overall_task, advance=1)
                continue

            # 6f. Add a new task to the progress bar for the current document type.
            doc_type_initial_description = (
                f"Preparing [bold magenta]{doc_type_key}[/] files"
            )
            doc_type_task = progress.add_task(
                doc_type_initial_description, total=len(pdf_files_to_process)
            )

            # 7. Iterate over PDF files for the current document type.
            for pdf_file_path in pdf_files_to_process:
                output_file_path = output_dir / f"{pdf_file_path.stem}.json"

                # 7a. Skip if output file already exists.
                if output_file_path.exists():
                    log.debug(
                        f"Output file already exists, skipping: {output_file_path}"
                    )
                    progress.update(
                        doc_type_task,
                        advance=1,
                        description=f"Skipped [bold magenta]{doc_type_key}[/]: [yellow]{pdf_file_path.name}[/]",
                    )
                    continue

                progress.update(
                    doc_type_task,
                    description=f"Processing [bold magenta]{doc_type_key}[/]: [cyan]{pdf_file_path.name}[/]",
                )

                # 7b. Run extraction pipeline with retries.
                extraction_result = None
                last_exception = None
                for attempt in range(3):
                    try:
                        extraction_result = run_extraction_pipeline(
                            file_path=pdf_file_path,
                            document_type=doc_type_key,
                            llm_provider=llm_provider.value,  # Ensure .value is used for Enum
                            llm_model=llm_model,
                        )
                        if extraction_result:
                            break
                    except Exception as e:
                        last_exception = e
                        log.warning(
                            f"Attempt {attempt + 1} failed for {pdf_file_path.name} due to parsing error: {e}. Retrying..."
                        )
                        if attempt == 2:
                            log.error(
                                f"All {3} attempts failed for {pdf_file_path.name}. Last error: {e}",
                                exc_info=True,
                            )

                # 7c. Save extraction result if successful.
                if extraction_result:
                    json_output = extraction_result.model_dump_json(indent=2)
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        f.write(json_output)
                    log.debug(f"Successfully saved prediction to {output_file_path}")
                else:
                    log.warning(
                        f"No extraction result for {pdf_file_path.name} after all attempts. Pipeline returned None or failed."
                    )
                    if last_exception:
                        log.error(
                            f"Last known error for {pdf_file_path.name}: {last_exception}",
                            exc_info=True,
                        )
                progress.update(doc_type_task, advance=1)

            # 8. Mark document type task as complete.
            progress.update(
                doc_type_task,
                completed=len(pdf_files_to_process),
                description=f"[bold magenta]{doc_type_key}[/] files completed :heavy_check_mark:",
            )
            progress.update(overall_task, advance=1)

        # 9. Mark overall task as complete.
        progress.update(
            overall_task,
            completed=len(document_types_map),
            description=f"All document types for [bold blue]{sanitized_model}[/bold blue] processed :party_popper:",
        )

    # 10. Log completion of the script.
    log.info("Prediction generation finished.")


if __name__ == "__main__":
    app()
