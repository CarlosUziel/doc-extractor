import enum
from pathlib import Path
from typing import Literal

import typer
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

from config.logger import log
from pipelines.doc_extraction import run_extraction_pipeline

app = typer.Typer()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LlmProviderChoice(str, enum.Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


def sanitize_model_name(model_name: str) -> str:
    """Sanitizes the model name to be used as a directory name."""
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
        None,
        "--limit",
        "-l",
        help="Limit the number of files to process per document type.",
        min=1,
    ),
):
    """
    Processes PDF documents from specified input directories, extracts information using a
    configurable Large Language Model (LLM), and saves the structured output as JSON files.

    The script iterates through predefined document types (e.g., "epc_certificate", "property_expose")
    and processes PDF files found in corresponding subdirectories of the `data_base_path`.
    For each PDF, it calls an extraction pipeline and stores the resulting JSON data.

    Output Structure:
        The extracted JSON for each PDF is saved to:
        `annotations_base_path/{document_type}/{sanitized_model_name}/{original_pdf_stem}.json`
        where `sanitized_model_name` is a filesystem-friendly version of the `llm_model` name.

    CLI Arguments:
    - `llm_provider`: Specifies the LLM provider to use (e.g., "openai", "gemini").
    - `llm_model`: Defines the specific model name for the chosen provider.
    - `data_base_path`: The root directory containing subfolders for different document types
                        (e.g., ".data/EPC certificates", ".data/property expose").
    - `annotations_base_path`: The root directory where generated JSON annotations will be stored.
    - `limit`: An optional integer to restrict processing to the first N PDF files found
               within each document type's subdirectory, useful for testing.

    Example Usage:
        python src/benchmark/generate_predictions.py \\
            --llm-provider gemini \\
            --llm-model gemini-2.5-pro-preview-05-06 \\
            --data-base-path .data \\
            --annotations-base-path annotations \\
            --limit 10
    """
    log.info(
        f"Starting prediction generation with provider: {llm_provider.value}, model: {llm_model}"  # Use .value
    )
    log.info(f"Input data path: {data_base_path}")
    log.info(f"Output annotations path: {annotations_base_path}")

    # Mapping document type keys (used in code and for output folder names)
    # to the actual folder names within the .data directory.
    document_types_map: dict[Literal["epc_certificate", "property_expose"], str] = {
        "epc_certificate": "EPC certificates",
        "property_expose": "property expose",
    }

    sanitized_model = sanitize_model_name(llm_model)

    # Define custom progress bar columns with more flair
    custom_progress_columns = [
        SpinnerColumn(spinner_name="dots", style="magenta"),  # Added spinner
        TextColumn("[progress.description]{task.description}"),
        BarColumn(
            bar_width=None,
            style="yellow",
            complete_style="green",
            finished_style="dim green",
        ),  # Styled Bar
        MofNCompleteColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),  # Added TimeElapsedColumn
    ]

    progress_console = Console()  # Create a Rich Console for the progress bar

    with Progress(
        *custom_progress_columns, console=progress_console
    ) as progress:  # Use the new console
        overall_task_description = f"Processing all document types for model: [bold blue]{sanitized_model}[/bold blue]"
        overall_task = progress.add_task(
            overall_task_description, total=len(document_types_map)
        )

        for doc_type_key, data_folder_name in document_types_map.items():
            progress.update(
                overall_task,
                description=f"Overall Progress (Current: [bold cyan]{doc_type_key}[/])",
            )

            current_data_path = data_base_path / data_folder_name
            output_dir = annotations_base_path / doc_type_key / sanitized_model

            log.info(
                f"Processing document type: {doc_type_key} (from folder: {data_folder_name})"
            )
            log.info(f"Looking for PDFs in: {current_data_path}")
            log.info(f"Output will be saved to: {output_dir}")

            if not current_data_path.exists() or not current_data_path.is_dir():
                log.warning(
                    f"Data directory not found or is not a directory: {current_data_path}"
                )
                progress.update(overall_task, advance=1)
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            # Collect PDF files to get a total for the progress bar
            pdf_files_to_process = list(current_data_path.rglob("*.pdf"))

            if not pdf_files_to_process:
                log.warning(f"No PDF files found in {current_data_path}")
                progress.update(overall_task, advance=1)
                continue

            # Apply limit if specified
            if limit is not None:
                pdf_files_to_process = pdf_files_to_process[:limit]

            if (
                not pdf_files_to_process
            ):  # If limit was 0 or less, or no files after slicing
                log.warning(
                    f"No PDF files to process for {doc_type_key} after applying limit."
                )
                progress.update(overall_task, advance=1)
                continue

            doc_type_initial_description = (
                f"Preparing [bold magenta]{doc_type_key}[/] files"
            )
            doc_type_task = progress.add_task(
                doc_type_initial_description, total=len(pdf_files_to_process)
            )

            for pdf_file_path in pdf_files_to_process:
                output_file_path = output_dir / f"{pdf_file_path.stem}.json"

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

                extraction_result = None
                last_exception = None
                for attempt in range(3):
                    try:
                        extraction_result = run_extraction_pipeline(
                            file_path=pdf_file_path,
                            document_type=doc_type_key,
                            llm_provider=llm_provider.value,
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

            # Ensure the task is marked as complete if loop finishes early due to skips
            progress.update(
                doc_type_task,
                completed=len(pdf_files_to_process),
                description=f"[bold magenta]{doc_type_key}[/] files completed :heavy_check_mark:",
            )
            progress.update(overall_task, advance=1)

        progress.update(
            overall_task,
            completed=len(document_types_map),
            description=f"All document types for [bold blue]{sanitized_model}[/bold blue] processed :party_popper:",
        )

    log.info("Prediction generation finished.")


if __name__ == "__main__":
    app()
