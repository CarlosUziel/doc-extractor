import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Type

import pandas as pd
import typer
from config.logger import log
from config.schema import (
    AddressDetails,
    EPCCertificateDetails,
    PropertyExposeDetails,
)
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

app = typer.Typer(
    help=(
        "Evaluation script for LLM predictions on document extraction. "
        "Analyzes missing values and compares model performance."
    ),
    add_completion=False,
    rich_markup_mode="markdown",
)
console = Console(record=True)

DOC_TYPE_TO_SCHEMA: Dict[str, Type[BaseModel]] = {
    "epc_certificate": EPCCertificateDetails,
    "property_expose": PropertyExposeDetails,
}
DOC_TYPE_FRIENDLY_NAMES: Dict[str, str] = {
    "epc_certificate": "EPC Certificate",
    "property_expose": "Property Expose",
}


def get_schema_keys(doc_type_key: str) -> List[str]:
    """Retrieves and flattens schema keys for a given document type.

    This function accesses a predefined mapping of document type keys to Pydantic
    schema classes. It iterates through the fields of the identified schema,
    flattening nested structures (specifically for 'realty_address') and
    excluding certain fields like bounding box information ('_bbox') and
    'realty_features'.

    Args:
        doc_type_key (str): The key identifying the document type (e.g.,
                            "epc_certificate", "property_expose").

    Returns:
        List[str]: A list of flattened schema key strings. For nested fields
                   (like under 'realty_address'), keys are combined with a
                   double underscore (e.g., "realty_address__street").
    """
    # 1. Get the schema class based on the document type key.
    schema_class = DOC_TYPE_TO_SCHEMA[doc_type_key]
    flattened_keys: List[str] = []

    # 2. Iterate over model fields to extract and flatten keys.
    for field_name, field_info in schema_class.model_fields.items():
        # 2a. Skip bounding box fields and 'realty_features'.
        if field_name.endswith("_bbox") or field_name == "realty_features":
            continue

        # 2b. Flatten 'realty_address' fields.
        if field_name == "realty_address":
            address_model_fields = AddressDetails.model_fields
            for sub_key in address_model_fields.keys():
                if not sub_key.endswith("_bbox"):
                    flattened_keys.append(f"{field_name}__{sub_key}")
        # 2c. Add other field names directly.
        else:
            flattened_keys.append(field_name)
    return flattened_keys


def sanitize_model_name_for_path(model_name: str) -> str:
    """Sanitizes a model name to be filesystem-friendly.

    Replaces characters like '/' and ':' with underscores.

    Args:
        model_name (str): The original model name string.

    Returns:
        str: A sanitized string suitable for use in file or directory names.
    """
    # 1. Replace characters that are problematic for file systems.
    return model_name.replace("/", "_").replace(":", "_")


def load_prediction_data(
    predictions_base_path: Path,
    doc_type: str,
    sanitized_model_name: str,
    schema_keys: List[str],
) -> pd.DataFrame:
    """Loads all prediction JSON files for a given model and document type.

    This function constructs a pandas DataFrame where each row corresponds to a
    JSON file and columns are derived from the provided schema_keys. It handles
    nested data under 'realty_address' by flattening it into the main DataFrame.

    Args:
        predictions_base_path (Path): The root directory where predictions are
                                      stored.
        doc_type (str): The specific document type (e.g., "epc_certificate").
        sanitized_model_name (str): The filesystem-friendly model name.
        schema_keys (List[str]): A list of keys expected in the JSON structure,
                                 which will become columns in the DataFrame.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded prediction data.
                      Returns an empty DataFrame with 'file_stem' index and
                      schema_keys as columns if the directory or JSON files are
                      not found, or if loading fails.
    """
    # 1. Construct the path to the model's document type directory.
    model_doc_type_path = predictions_base_path / doc_type / sanitized_model_name

    all_file_data: List[Dict[str, Any]] = []

    # 2. Check if the directory exists; return empty DataFrame if not.
    if not model_doc_type_path.is_dir():
        log.warning(
            f"Directory not found for model '{sanitized_model_name}', "
            f"doc_type '{doc_type}': {model_doc_type_path}"
        )
        empty_df = pd.DataFrame(columns=["file_stem"] + schema_keys)
        return empty_df.set_index("file_stem")

    # 3. Get all JSON files in the directory; return empty DataFrame if none found.
    json_files = list(model_doc_type_path.glob("*.json"))
    if not json_files:
        log.warning(f"No JSON files found in {model_doc_type_path}")
        empty_df = pd.DataFrame(columns=["file_stem"] + schema_keys)
        return empty_df.set_index("file_stem")

    # 4. Process each JSON file.
    for json_file in json_files:
        try:
            # 4a. Load JSON data from the file.
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            processed_file_data: Dict[str, Any] = {"file_stem": json_file.stem}
            # 4b. Extract data for each schema key, handling nested 'realty_address'.
            for key in schema_keys:
                if "__" in key and key.startswith("realty_address"):
                    parent_key, sub_key = key.split(
                        "__",
                        1,
                    )
                    if parent_key == "realty_address":
                        nested_data = data.get(parent_key, {})
                        if isinstance(nested_data, dict):
                            processed_file_data[key] = nested_data.get(sub_key, None)
                        else:
                            processed_file_data[key] = None
                else:
                    processed_file_data[key] = data.get(key, None)
            all_file_data.append(processed_file_data)
        except Exception as e:
            log.error(f"Error loading or flattening {json_file}: {e}")

    # 5. Create DataFrame from processed data; return empty if no data was loaded.
    if not all_file_data:
        empty_df = pd.DataFrame(columns=["file_stem"] + schema_keys)
        return empty_df.set_index("file_stem")

    # 6. Convert list of dictionaries to DataFrame, set index, and reorder columns.
    df = pd.DataFrame(all_file_data)
    df = df.set_index("file_stem")
    df = df.reindex(columns=schema_keys)

    return df


def generate_missing_values_reports(
    df_model_data: pd.DataFrame,
    model_name: str,
    doc_type_friendly_name: str,
    doc_type_key: str,
) -> None:
    """Generates and prints reports on missing values in prediction data.

    This function calculates and saves two types of reports:
    1.  Per-file missing values: A CSV indicating which fields are missing in each file.
    2.  Summarized missing values: A CSV showing the percentage of nulls for each field
        across all files.

    Reports are saved to a subdirectory within the main `REPORTS_DIR`.

    Args:
        df_model_data (pd.DataFrame): DataFrame containing the prediction data
                                      for a specific model and document type.
        model_name (str): The sanitized name of the model being evaluated.
        doc_type_friendly_name (str): A user-friendly name for the document type.
        doc_type_key (str): The document type key used for path generation.
    """
    # 1. Print a rule to the console for visual separation.
    console.rule(
        f"[bold]Missing Values Analysis for: {model_name} - "
        f"{doc_type_friendly_name}[/bold]",
        style="magenta",
    )

    # 2. Handle empty DataFrame case.
    if df_model_data.empty:
        console.print(
            f"[yellow]No data loaded for {model_name} - {doc_type_friendly_name}. "
            f"Skipping missing values report.[/yellow]"
        )
        return

    # 3. Create the output directory for reports.
    output_dir = REPORTS_DIR / doc_type_key / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Generate and save the per-file missing values report.
    console.print("[bold green]1. Missing Values Report (Per File)[/bold green]")
    df_missing_per_file = df_model_data.isnull()

    if df_missing_per_file.empty:
        console.print(
            "[italic]No files processed or all files were empty for per-file "
            "missing values report.[/italic]"
        )
    else:
        csv_path_per_file = output_dir / "missing_values_per_file.csv"
        df_missing_per_file.to_csv(csv_path_per_file, quoting=csv.QUOTE_ALL)
        console.print(f"Saved per-file missing values report to: {csv_path_per_file}")

    # 5. Generate and save the summarized missing values report (percentage of nulls).
    console.print(
        "[bold green]2. Missing Values Summary (Percentage of Nulls)[/bold green]"
    )
    if df_missing_per_file.empty:
        console.print("[yellow]No data for missing values summary.[/yellow]")
    else:
        df_missing_summary = df_missing_per_file.mean().sort_values(ascending=False)
        if df_missing_summary.empty:
            console.print(
                "[italic]No fields to summarize or all fields were present in all "
                "files for summary report.[/italic]"
            )
        else:
            csv_path_summary = output_dir / "missing_values_summary.csv"
            df_missing_summary.to_frame(name="Percentage Null").to_csv(
                csv_path_summary, quoting=csv.QUOTE_ALL
            )
            console.print(f"Saved missing values summary report to: {csv_path_summary}")


def compare_values(gt_val: Any, pred_val: Any) -> bool:
    """Compares a ground truth value with a predicted value, with special handling for lists and floats.

    - For lists, elements are converted to strings and sorted before comparison to ensure
      order-insensitivity.
    - For floats, an absolute tolerance of 1e-9 is used for comparison. `pd.isna` is used
      to correctly compare NaN values (NaN == NaN should be True in this context).
    - For all other types, direct equality comparison is used.

    Args:
        gt_val (Any): The ground truth value.
        pred_val (Any): The predicted value.

    Returns:
        bool: True if the values are considered equal based on the comparison
              logic, False otherwise.
    """
    # 1. Handle list comparison: convert elements to strings and sort.
    if isinstance(gt_val, list) and isinstance(pred_val, list):
        return sorted([str(x) for x in gt_val]) == sorted([str(x) for x in pred_val])
    # 2. Handle float comparison: use isna for NaN and absolute tolerance for others.
    if isinstance(gt_val, float) and isinstance(pred_val, float):
        return pd.isna(gt_val) and pd.isna(pred_val) or abs(gt_val - pred_val) < 1e-9
    # 3. Default comparison for other types.
    return gt_val == pred_val


def calculate_comparative_metrics(
    df_gt: pd.DataFrame,
    df_eval: pd.DataFrame,
    schema_keys: List[str],
    gt_model_name: str,
    eval_model_name: str,
    doc_type_friendly_name: str,
    doc_type_key: str,
) -> None:
    """Calculates comparative metrics and saves them to a CSV file.

    This function compares predictions from an evaluation model against a ground
    truth model. It calculates True Positives (TP), False Positives (FP),
    True Negatives (TN), and False Negatives (FN) for each field in the schema.
    Based on these, it computes Accuracy, Precision, Recall, and Balanced Accuracy.
    The results are saved to a CSV file.

    Args:
        df_gt (pd.DataFrame): DataFrame containing the ground truth data.
        df_eval (pd.DataFrame): DataFrame containing the prediction data from the
                                model being evaluated.
        schema_keys (List[str]): List of schema keys to compare.
        gt_model_name (str): Sanitized name of the ground truth model.
        eval_model_name (str): Sanitized name of the model being evaluated.
        doc_type_friendly_name (str): User-friendly name for the document type.
        doc_type_key (str): Document type key for path generation.
    """
    # 1. Print a rule to the console for visual separation.
    console.rule(
        f"[bold]Comparative Metrics: '{eval_model_name}' vs (GT) '{gt_model_name}' "
        f"for {doc_type_friendly_name}[/bold]",
        style="blue",
    )

    # 2. Handle cases where one or both DataFrames are empty.
    if df_gt.empty and df_eval.empty:
        console.print(
            f"[yellow]No data for BOTH ground truth ('{gt_model_name}') and "
            f"evaluated model ('{eval_model_name}') for {doc_type_friendly_name}. "
            f"Skipping.[/yellow]"
        )
        return
    if df_gt.empty:
        console.print(
            f"[yellow]No data for ground truth model '{gt_model_name}' for "
            f"{doc_type_friendly_name}. Cannot compute comparative metrics.[/yellow]"
        )
        return
    if df_eval.empty:
        console.print(
            f"[yellow]No data for evaluated model '{eval_model_name}' for "
            f"{doc_type_friendly_name}. Cannot compute comparative metrics.[/yellow]"
        )
        return

    metrics_results: List[Dict[str, Any]] = []

    # 3. Align ground truth and evaluation DataFrames.
    # This ensures comparison of the same files and fields, using an outer join
    # to keep all files from both DataFrames.
    aligned_gt, aligned_eval = df_gt.align(df_eval, join="outer", axis=0)
    # 3a. Filter columns to include only those present in schema_keys and the DataFrame.
    aligned_gt = aligned_gt[list(set(schema_keys).intersection(aligned_gt.columns))]
    aligned_eval = aligned_eval[
        list(set(schema_keys).intersection(aligned_eval.columns))
    ]

    # 4. Iterate through each schema key to calculate metrics.
    for key in schema_keys:
        # 4a. Initialize TP, FP, TN, FN counters.
        tp, fp, tn, fn = 0, 0, 0, 0

        # 4b. Get corresponding columns from aligned DataFrames.
        gt_col = aligned_gt[key]
        pred_col = aligned_eval[key]

        # 4c. Determine null values for ground truth and predictions.
        gt_is_null = gt_col.isnull()
        pred_is_null = pred_col.isnull()

        # 4d. Compare values row-wise using the compare_values function.
        # A temporary DataFrame is created for proper alignment with the apply method.
        temp_df = pd.DataFrame({"gt": gt_col, "pred": pred_col})
        are_values_equal = temp_df.apply(
            lambda row: compare_values(row["gt"], row["pred"]), axis=1
        )

        # 4e. Calculate TP, FP, TN, FN using vectorized boolean operations.
        tp_mask = ~gt_is_null & ~pred_is_null & are_values_equal
        fp_mask = ~pred_is_null & (gt_is_null | ~are_values_equal)
        tn_mask = gt_is_null & pred_is_null
        fn_mask = ~gt_is_null & pred_is_null

        tp = tp_mask.sum()
        fp = fp_mask.sum()
        tn = tn_mask.sum()
        fn = fn_mask.sum()

        # 4f. Calculate performance metrics.
        total_predictions = tp + fp + tn + fn
        accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (
            (recall + specificity) / 2 if (recall > 0 or specificity > 0) else 0
        )

        metrics_results.append(
            {
                "Field": key,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Balanced Acc.": balanced_accuracy,
            }
        )

    # 5. Handle cases where no metrics were calculated.
    if not metrics_results:
        console.print(
            "[yellow]No metrics calculated (perhaps no common schema keys or files "
            "after alignment).[/yellow]"
        )
        return

    # 6. Create a DataFrame from the metrics results.
    df_metrics = pd.DataFrame(metrics_results)

    # 7. Save the metrics DataFrame to a CSV file.
    if df_metrics.empty:
        console.print("[italic]No fields to display metrics for.[/italic]")
    else:
        # 7a. Define the output directory path.
        output_dir = (
            REPORTS_DIR / doc_type_key / f"{eval_model_name}_vs_{gt_model_name}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # 7b. Define the CSV file path and save the DataFrame.
        csv_path_metrics = output_dir / "comparative_metrics.csv"
        df_metrics.to_csv(csv_path_metrics, index=False, quoting=csv.QUOTE_ALL)
        console.print(f"Saved comparative metrics to: {csv_path_metrics}")


@app.command(rich_help_panel="Main Commands")
def main(
    annotations_base_path: Path = typer.Option(
        lambda: PROJECT_ROOT / "annotations",
        help="Base path for input annotations (JSON prediction files).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        show_default=True,
    ),
    ground_truth_model_name: str = typer.Option(
        "gemini-2.5-pro-preview-05-06",
        help="Model name (as in directory) to use as ground truth.",
        rich_help_panel="Model Comparison Options",
    ),
    test_model_name: str = typer.Option(
        "gpt-4o-mini",
        help="Model name (as in directory) to evaluate against ground truth.",
        rich_help_panel="Model Comparison Options",
    ),
    skip_missing_value_analysis: bool = typer.Option(
        False,
        "--skip-missing",
        help="Skip the missing value analysis part.",
        rich_help_panel="Analysis Options",
    ),
    skip_comparative_metrics: bool = typer.Option(
        False,
        "--skip-comparison",
        help="Skip the comparative metrics part.",
        rich_help_panel="Analysis Options",
    ),
) -> None:
    """
    Evaluates LLM predictions by performing two main analyses:
    1. Missing Value Analysis: For all models and document types found in the
       `annotations_base_path`, this script calculates and reports:
        a. Per-file missing values: Identifies which fields are missing in each
           individual prediction file.
        b. Summarized missing values: Calculates the overall percentage of nulls
           for each field across all files for a given model and document type.
       These reports are saved as CSV files in the `reports` directory, structured
       by document type and model name.

    2. Comparative Performance Metrics: Compares a specified `test_model_name`
       against a `ground_truth_model_name`. For each document type, it calculates
       metrics like True Positives (TP), False Positives (FP), True Negatives (TN),
       False Negatives (FN), Accuracy, Precision, Recall, and Balanced Accuracy for
       each field in the schema. These comparative metrics are also saved as CSV
       files in the `reports` directory.

    The script uses `typer` for command-line interface management and `pandas` for
    data manipulation. Logging is handled via a custom logger, and rich text is
    used for console output.

    Args:
        annotations_base_path (Path): Base path for input annotations (JSON
                                      prediction files). Defaults to
                                      `PROJECT_ROOT / "annotations"`.
        ground_truth_model_name (str): Model name (as found in the directory
                                       structure) to be used as the ground truth
                                       for comparison. Defaults to
                                       "gemini-2.5-pro-preview-05-06".
        test_model_name (str): Model name (as found in the directory structure)
                               to be evaluated against the ground truth. Defaults
                               to "gpt-4o-mini".
        skip_missing_value_analysis (bool): If True, skips the missing value
                                            analysis. Defaults to False.
        skip_comparative_metrics (bool): If True, skips the comparative metrics
                                         analysis. Defaults to False.
    """
    # 1. Initial setup and logging.
    console.print(
        Panel(
            Text(
                "Starting Document Extraction Evaluation Script",
                justify="center",
                style="bold white on blue",
            )
        )
    )
    log.info(f"Annotations Path: {annotations_base_path}")
    log.info(f"Ground Truth Model (for comparison): {ground_truth_model_name}")
    log.info(f"Evaluated Model (for comparison): {test_model_name}")

    doc_type_keys = list(DOC_TYPE_TO_SCHEMA.keys())

    # 2. Perform Missing Value Analysis if not skipped.
    if not skip_missing_value_analysis:
        console.rule(
            "[bold underline]Missing Value Analysis[/bold underline]", style="magenta"
        )

        # 2a. Discover all model names from the directory structure.
        all_found_sanitized_model_names = set()
        for doc_type_key in doc_type_keys:
            doc_type_path = annotations_base_path / doc_type_key
            if doc_type_path.is_dir():
                for model_dir in doc_type_path.iterdir():
                    if model_dir.is_dir() and not model_dir.name.startswith("."):
                        all_found_sanitized_model_names.add(model_dir.name)

        if not all_found_sanitized_model_names:
            log.warning(
                "No model prediction directories found in annotations path for "
                "missing value analysis."
            )
        else:
            log.info(
                f"Found models for missing value analysis: "
                f"{', '.join(sorted(list(all_found_sanitized_model_names)))}"
            )

        # 2b. Iterate through each found model and document type.
        for s_model_name_iter in sorted(list(all_found_sanitized_model_names)):
            for doc_type_key in doc_type_keys:
                doc_type_friendly = DOC_TYPE_FRIENDLY_NAMES[doc_type_key]
                schema_keys = get_schema_keys(doc_type_key)

                if not schema_keys:
                    log.error(
                        f"Could not extract schema keys for {doc_type_friendly}. "
                        f"Skipping this document type for {s_model_name_iter}."
                    )
                    continue

                log.info(
                    f"Processing missing values for model '{s_model_name_iter}' - "
                    f"'{doc_type_friendly}'..."
                )

                # 2c. Load prediction data for the current model and document type.
                df_model_data = load_prediction_data(
                    annotations_base_path,
                    doc_type_key,
                    s_model_name_iter,
                    schema_keys,
                )

                # 2d. Generate and save missing value reports.
                generate_missing_values_reports(
                    df_model_data,
                    s_model_name_iter,
                    doc_type_friendly,
                    doc_type_key,
                )
    else:
        console.print(
            "[italic]Skipping Missing Value Analysis as per user request.[/italic]"
        )

    # 3. Perform Comparative Metrics Analysis if not skipped.
    if not skip_comparative_metrics:
        console.rule(
            "[bold underline]Comparative Metrics Analysis[/bold underline]",
            style="blue",
        )

        # 3a. Sanitize ground truth and evaluation model names for file paths.
        s_gt_model = sanitize_model_name_for_path(ground_truth_model_name)
        s_eval_model = sanitize_model_name_for_path(test_model_name)

        log.info(
            f"Comparing: Evaluated='{s_eval_model}' (from '{test_model_name}') vs "
            f"GroundTruth='{s_gt_model}' (from '{ground_truth_model_name}')"
        )

        # 3b. Iterate through each document type.
        for doc_type_key in doc_type_keys:
            doc_type_friendly = DOC_TYPE_FRIENDLY_NAMES[doc_type_key]
            schema_keys = get_schema_keys(doc_type_key)

            if not schema_keys:
                log.error(
                    f"Could not extract schema keys for {doc_type_friendly}. "
                    f"Skipping comparative metrics for this type."
                )
                continue

            log.info(
                f"Processing comparative metrics for document type "
                f"'{doc_type_friendly}'..."
            )

            # 3c. Load prediction data for ground truth and evaluation models.
            df_gt = load_prediction_data(
                annotations_base_path, doc_type_key, s_gt_model, schema_keys
            )
            df_eval = load_prediction_data(
                annotations_base_path, doc_type_key, s_eval_model, schema_keys
            )

            # 3d. Calculate and save comparative metrics.
            calculate_comparative_metrics(
                df_gt,
                df_eval,
                schema_keys,
                s_gt_model,
                s_eval_model,
                doc_type_friendly,
                doc_type_key,
            )
    else:
        console.print(
            "[italic]Skipping Comparative Metrics Analysis as per user request.[/italic]"
        )

    # 4. Print completion message.
    console.print(
        Panel(
            Text(
                "Evaluation Script Finished",
                justify="center",
                style="bold white on green",
            )
        )
    )


if __name__ == "__main__":
    app()
