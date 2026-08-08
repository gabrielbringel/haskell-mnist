#!/usr/bin/env python3
"""Validates all invariants of results files: CSVs, logs, markdown, and PDFs.

Checks:
  1. results/training-metrics.csv
     - Exact header: "epoch,loss,accuracy"
     - Exactly 25 records
     - Epochs 1-25, unique, sorted
     - Epoch 25: loss=0.1629, accuracy=95.49

  2. results/training-run-full-25-epochs.log
     - Exact match of metrics with CSV

  3. results/confusion-matrix.csv
     - 10x10 matrix
     - Non-negative integers only
     - Sum = 10,000
     - Diagonal sum = 9,549
     - Derived accuracy = 95.49%
     - Cell [4,9] (5th row, 10th column) = 28

  4. Consistency between confusion matrix CSV and log

  5. results/training-log.md
     - Coherent values with CSV metrics

  6. SHA-256 identity of the four MNIST IDX files

  7. PDF equality and semantic labels:
     - results/fig-training.pdf == paper/fig-training.pdf
     - results/fig-confusion.pdf == paper/fig-confusion.pdf
     - Training plot uses "Epoch", never "Training step"

Usage:
  python scripts/validate_results.py
"""

import os
import sys
import re
import csv
import hashlib
from pathlib import Path
from pypdf import PdfReader

# Constants for expected final epoch values
EXPECTED_EPOCH_25_LOSS = 0.1629
EXPECTED_EPOCH_25_ACCURACY = 95.49
EXPECTED_CONFUSION_DIAGONAL = 9549
EXPECTED_CONFUSION_TOTAL = 10000
EXPECTED_NUM_EPOCHS = 25
EXPECTED_CONFUSION_SIZE = 10
EXPECTED_DATASET_HASHES = {
    "train-images-idx3-ubyte": "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    "train-labels-idx1-ubyte": "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
    "t10k-images-idx3-ubyte": "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    "t10k-labels-idx1-ubyte": "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
}


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


def get_results_dir():
    """Get the absolute path to the results directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), "results")


def sha256_file(path):
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_training_metrics_csv(results_dir):
    """Validate results/training-metrics.csv."""
    print("\n[1/7] Validating results/training-metrics.csv...")

    csv_path = os.path.join(results_dir, "training-metrics.csv")
    if not os.path.exists(csv_path):
        raise ValidationError(f"File not found: {csv_path}")

    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Check header
    header = lines[0].strip()
    if header != "epoch,loss,accuracy":
        raise ValidationError(
            f"CSV header mismatch. Expected 'epoch,loss,accuracy', got '{header}'"
        )
    print("  ✓ Header is correct: 'epoch,loss,accuracy'")

    # Parse CSV data (skip header)
    reader = csv.DictReader(lines)
    rows = list(reader)

    # Check exactly 25 records
    if len(rows) != EXPECTED_NUM_EPOCHS:
        raise ValidationError(
            f"Expected {EXPECTED_NUM_EPOCHS} records, got {len(rows)}"
        )
    print(f"  ✓ Exactly {EXPECTED_NUM_EPOCHS} records found")

    # Check epochs are 1-25, unique, and ordered
    epochs = []
    for i, row in enumerate(rows):
        try:
            epoch = int(row["epoch"])
            epochs.append(epoch)
        except (ValueError, KeyError) as e:
            raise ValidationError(f"Row {i+1}: Cannot parse epoch: {e}")

    if epochs != list(range(1, EXPECTED_NUM_EPOCHS + 1)):
        raise ValidationError(
            f"Epochs must be 1-{EXPECTED_NUM_EPOCHS} in order. Got: {epochs}"
        )
    print(f"  ✓ Epochs are unique and ordered 1-{EXPECTED_NUM_EPOCHS}")

    # Check epoch 25 values
    epoch_25_row = rows[-1]
    try:
        loss_25 = float(epoch_25_row["loss"])
        accuracy_25 = float(epoch_25_row["accuracy"])
    except (ValueError, KeyError) as e:
        raise ValidationError(f"Cannot parse epoch 25 metrics: {e}")

    if abs(loss_25 - EXPECTED_EPOCH_25_LOSS) > 0.0001:
        raise ValidationError(
            f"Epoch 25 loss: expected {EXPECTED_EPOCH_25_LOSS}, "
            f"got {loss_25}"
        )
    if abs(accuracy_25 - EXPECTED_EPOCH_25_ACCURACY) > 0.01:
        raise ValidationError(
            f"Epoch 25 accuracy: expected {EXPECTED_EPOCH_25_ACCURACY}, "
            f"got {accuracy_25}"
        )
    print(f"  ✓ Epoch 25: loss={loss_25}, accuracy={accuracy_25}%")

    return rows


def validate_training_log(results_dir, csv_rows):
    """Validate results/training-run-full-25-epochs.log matches CSV."""
    print("\n[2/7] Validating results/training-run-full-25-epochs.log...")

    log_path = os.path.join(results_dir, "training-run-full-25-epochs.log")
    if not os.path.exists(log_path):
        raise ValidationError(f"File not found: {log_path}")

    with open(log_path, "r") as f:
        log_content = f.read()

    # Extract metrics from log using regex: "epoch N: mean loss X.XXXX, test accuracy Y.YY%"
    pattern = r"epoch\s+(\d+):\s+mean loss\s+([\d.]+),\s+test accuracy\s+([\d.]+)%"
    matches = re.findall(pattern, log_content)

    if len(matches) != EXPECTED_NUM_EPOCHS:
        raise ValidationError(
            f"Expected {EXPECTED_NUM_EPOCHS} epoch records in log, got {len(matches)}"
        )

    # Compare with CSV
    for i, (epoch_str, loss_str, acc_str) in enumerate(matches):
        csv_row = csv_rows[i]

        epoch = int(epoch_str)
        loss = float(loss_str)
        accuracy = float(acc_str)

        csv_epoch = int(csv_row["epoch"])
        csv_loss = float(csv_row["loss"])
        csv_accuracy = float(csv_row["accuracy"])

        if epoch != csv_epoch:
            raise ValidationError(
                f"Log epoch {i}: epoch mismatch. Log={epoch}, CSV={csv_epoch}"
            )

        if abs(loss - csv_loss) > 0.0001:
            raise ValidationError(
                f"Log epoch {epoch}: loss mismatch. Log={loss}, CSV={csv_loss}"
            )

        if abs(accuracy - csv_accuracy) > 0.01:
            raise ValidationError(
                f"Log epoch {epoch}: accuracy mismatch. Log={accuracy}, "
                f"CSV={csv_accuracy}"
            )

    print(f"  ✓ All {EXPECTED_NUM_EPOCHS} epochs match between log and CSV")


def validate_confusion_matrix_csv(results_dir):
    """Validate results/confusion-matrix.csv."""
    print("\n[3/7] Validating results/confusion-matrix.csv...")

    csv_path = os.path.join(results_dir, "confusion-matrix.csv")
    if not os.path.exists(csv_path):
        raise ValidationError(f"File not found: {csv_path}")

    # Load as integers without header
    matrix = []
    with open(csv_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split(",")
            row = []
            for col_num, part in enumerate(parts, 1):
                try:
                    val = int(part)
                    if val < 0:
                        raise ValidationError(
                            f"Row {line_num}, col {col_num}: negative value {val}"
                        )
                    row.append(val)
                except ValueError:
                    raise ValidationError(
                        f"Row {line_num}, col {col_num}: not an integer: '{part}'"
                    )
            matrix.append(row)

    # Check dimensions
    if len(matrix) != EXPECTED_CONFUSION_SIZE:
        raise ValidationError(
            f"Confusion matrix: expected {EXPECTED_CONFUSION_SIZE} rows, "
            f"got {len(matrix)}"
        )

    for row_num, row in enumerate(matrix, 1):
        if len(row) != EXPECTED_CONFUSION_SIZE:
            raise ValidationError(
                f"Row {row_num}: expected {EXPECTED_CONFUSION_SIZE} columns, "
                f"got {len(row)}"
            )

    print(f"  ✓ Matrix is {EXPECTED_CONFUSION_SIZE}x{EXPECTED_CONFUSION_SIZE}")

    # Check sum
    total_sum = sum(sum(row) for row in matrix)
    if total_sum != EXPECTED_CONFUSION_TOTAL:
        raise ValidationError(
            f"Matrix sum: expected {EXPECTED_CONFUSION_TOTAL}, got {total_sum}"
        )
    print(f"  ✓ Total sum: {total_sum}")

    # Check diagonal
    diagonal_sum = sum(matrix[i][i] for i in range(EXPECTED_CONFUSION_SIZE))
    if diagonal_sum != EXPECTED_CONFUSION_DIAGONAL:
        raise ValidationError(
            f"Diagonal sum: expected {EXPECTED_CONFUSION_DIAGONAL}, "
            f"got {diagonal_sum}"
        )
    print(f"  ✓ Diagonal sum: {diagonal_sum}")

    # Derived accuracy
    derived_accuracy = (diagonal_sum / EXPECTED_CONFUSION_TOTAL) * 100
    if abs(derived_accuracy - EXPECTED_EPOCH_25_ACCURACY) > 0.01:
        raise ValidationError(
            f"Derived accuracy: expected {EXPECTED_EPOCH_25_ACCURACY}%, "
            f"got {derived_accuracy}%"
        )
    print(f"  ✓ Derived accuracy: {derived_accuracy:.2f}%")

    # Check specific cell: row 5 (index 4), column 10 (index 9)
    cell_value = matrix[4][9]
    if cell_value != 28:
        raise ValidationError(
            f"Cell [row 5, col 10] (index [4, 9]): expected 28, got {cell_value}"
        )
    print(f"  ✓ Cell [row 5, col 10]: {cell_value}")

    return matrix


def validate_confusion_in_log(results_dir, expected_matrix):
    """Validate confusion matrix in training log matches CSV."""
    print("\n[4/7] Validating confusion matrix in training-run-full-25-epochs.log...")

    log_path = os.path.join(results_dir, "training-run-full-25-epochs.log")

    with open(log_path, "r") as f:
        log_content = f.read()

    # Extract the confusion matrix from log
    # Look for the line "Matriz de confusão..." and parse the following lines
    matrix_start = log_content.find("Matriz de confusão")
    if matrix_start == -1:
        raise ValidationError("Confusion matrix section not found in log")

    # Extract lines after "Matriz de confusão..."
    matrix_section = log_content[matrix_start:]
    lines = matrix_section.split("\n")

    # Skip the header line and parse the matrix
    log_matrix = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("Escrito"):
            break
        # Parse the row
        parts = line.split()
        if parts:
            try:
                row = [int(p) for p in parts]
                log_matrix.append(row)
            except ValueError:
                # Skip non-numeric lines
                continue

    # Compare with expected matrix
    if len(log_matrix) != len(expected_matrix):
        raise ValidationError(
            f"Log confusion matrix: expected {len(expected_matrix)} rows, "
            f"got {len(log_matrix)}"
        )

    for i, (log_row, expected_row) in enumerate(zip(log_matrix, expected_matrix)):
        if log_row != expected_row:
            raise ValidationError(
                f"Log row {i+1} mismatch:\n"
                f"  Log:      {log_row}\n"
                f"  Expected: {expected_row}"
            )

    print(f"  ✓ Confusion matrix in log matches CSV")


def validate_training_log_md(results_dir, csv_rows):
    """Validate results/training-log.md for coherence with CSV values."""
    print("\n[5/7] Validating results/training-log.md...")

    md_path = os.path.join(results_dir, "training-log.md")
    if not os.path.exists(md_path):
        raise ValidationError(f"File not found: {md_path}")

    with open(md_path, "r") as f:
        md_content = f.read()

    # Check key metrics are mentioned
    checks = [
        ("95.49%", "final accuracy"),
        ("9 549", "diagonal sum"),
        ("0.1629", "epoch 25 loss"),
        ("10 000", "total test examples"),
    ]

    for value, description in checks:
        if value not in md_content:
            raise ValidationError(
                f"training-log.md: expected '{value}' ({description}) "
                f"not found"
            )

    print("  ✓ training-log.md contains key metrics")
    print("    - 95.49% final accuracy")
    print("    - 9,549 diagonal sum")
    print("    - 0.1629 epoch 25 loss")
    print("    - 10,000 test examples")


def validate_dataset_hashes(results_dir):
    """Validate the exact uncompressed IDX files used by the experiment."""
    print("\n[6/7] Validating MNIST dataset hashes...")
    data_dir = Path(results_dir).parent / "data"

    for filename, expected_hash in EXPECTED_DATASET_HASHES.items():
        path = data_dir / filename
        if not path.exists():
            raise ValidationError(f"Dataset file not found: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValidationError(
                f"Dataset hash mismatch for {filename}:\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}"
            )
        print(f"  ✓ {filename}: {actual_hash[:16]}...")


def pdf_text(path):
    """Extract text from a PDF for semantic label checks."""
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def validate_pdfs(results_dir):
    """Validate PDF copies and the labels most prone to stale-figure regressions."""
    print("\n[7/7] Validating PDF hashes and semantic labels...")

    pairs = [
        ("fig-training.pdf", "training plot"),
        ("fig-confusion.pdf", "confusion matrix"),
    ]

    paper_dir = os.path.join(os.path.dirname(results_dir), "paper")

    for filename, description in pairs:
        results_pdf = os.path.join(results_dir, filename)
        paper_pdf = os.path.join(paper_dir, filename)

        if not os.path.exists(results_pdf):
            raise ValidationError(f"File not found: {results_pdf}")
        if not os.path.exists(paper_pdf):
            raise ValidationError(f"File not found: {paper_pdf}")

        results_hash = sha256_file(results_pdf)
        paper_hash = sha256_file(paper_pdf)

        if results_hash != paper_hash:
            raise ValidationError(
                f"PDF hash mismatch for {filename} ({description}):\n"
                f"  results/{filename}: {results_hash}\n"
                f"  paper/{filename}:   {paper_hash}"
            )

        print(f"  ✓ {filename} ({description}): {results_hash[:16]}...")

    training_pdf = os.path.join(results_dir, "fig-training.pdf")
    training_text = pdf_text(training_pdf)
    if "Epoch" not in training_text:
        raise ValidationError("fig-training.pdf does not contain the x-axis label 'Epoch'")
    if "Training step" in training_text:
        raise ValidationError("fig-training.pdf still contains the stale label 'Training step'")
    print("  ✓ fig-training.pdf uses 'Epoch' and not 'Training step'")

    confusion_pdf = os.path.join(results_dir, "fig-confusion.pdf")
    confusion_text = pdf_text(confusion_pdf)
    for label in ("Predicted label", "True label"):
        if label not in confusion_text:
            raise ValidationError(
                f"fig-confusion.pdf does not contain the expected label '{label}'"
            )
    print("  ✓ fig-confusion.pdf contains both axis labels")


def main():
    """Run all validations."""
    print("=" * 70)
    print("HASKELL-MNIST RESULTS VALIDATOR")
    print("=" * 70)

    results_dir = get_results_dir()
    print(f"\nResults directory: {results_dir}")

    try:
        csv_rows = validate_training_metrics_csv(results_dir)
        validate_training_log(results_dir, csv_rows)
        confusion_matrix = validate_confusion_matrix_csv(results_dir)
        validate_confusion_in_log(results_dir, confusion_matrix)
        validate_training_log_md(results_dir, csv_rows)
        validate_dataset_hashes(results_dir)
        validate_pdfs(results_dir)

        print("\n" + "=" * 70)
        print("SUCCESS: All validations passed!")
        print("=" * 70)
        print("\nValidated invariants:")
        print("  ✓ CSV header: 'epoch,loss,accuracy'")
        print(f"  ✓ CSV records: exactly {EXPECTED_NUM_EPOCHS}")
        print(f"  ✓ Epochs: unique, sorted 1-{EXPECTED_NUM_EPOCHS}")
        print(f"  ✓ Epoch 25: loss={EXPECTED_EPOCH_25_LOSS}, "
              f"accuracy={EXPECTED_EPOCH_25_ACCURACY}%")
        print(f"  ✓ Confusion matrix: {EXPECTED_CONFUSION_SIZE}x"
              f"{EXPECTED_CONFUSION_SIZE}")
        print(f"  ✓ Confusion sum: {EXPECTED_CONFUSION_TOTAL}")
        print(f"  ✓ Confusion diagonal: {EXPECTED_CONFUSION_DIAGONAL}")
        print(f"  ✓ Derived accuracy: {EXPECTED_EPOCH_25_ACCURACY}%")
        print(f"  ✓ Cell [5, 10]: 28")
        print(f"  ✓ Log metrics match CSV")
        print(f"  ✓ Log confusion matrix matches CSV")
        print(f"  ✓ Markdown coherence with values")
        print(f"  ✓ MNIST SHA-256: all four IDX files")
        print(f"  ✓ PDF SHA-256 and semantic labels")
        print("=" * 70)
        return 0

    except ValidationError as e:
        print("\n" + "=" * 70)
        print("VALIDATION FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("=" * 70)
        return 1
    except Exception as e:
        print("\n" + "=" * 70)
        print("UNEXPECTED ERROR")
        print("=" * 70)
        print(f"\nError: {type(e).__name__}: {e}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
