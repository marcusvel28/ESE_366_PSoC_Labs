
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import argparse

# Project directories
ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["baseline", "optimized"],
    required=True
)

parser.add_argument(
    "--block",
    type=int,
    choices=[8, 16, 32],
    required=True
)

args = parser.parse_args()

OUTPUTS = (
    ROOT / "outputs" /
    args.mode /
    f"block_{args.block}"
)

FIGURES = OUTPUTS / "figures"

METRICS = OUTPUTS / "metrics.csv"
PREDICTIONS = OUTPUTS / "predictions.csv"

DIGITS_PGM = OUTPUTS / "test_digits.pgm"
DIGITS_PNG = OUTPUTS / "test_digits.png"

FIGURES.mkdir(parents=True, exist_ok=True)


# Plot configuration
DIGIT_SIZE = 28
GRID_COLS = 10
NUM_EXAMPLES = 3

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})


def save_figure(fig, name):
    # Save each figure in raster and vector formats.
    for ext in ("png", "svg"):
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved {name} (PNG, SVG)")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_digit_sheet():
    # Convert the original PGM contact sheet to PNG.
    image = Image.open(DIGITS_PGM).convert("L")

    image.save(DIGITS_PNG)
    print(f"Saved {DIGITS_PNG.relative_to(ROOT)}")

    return np.asarray(image)


def plot_metrics():
    data = load_csv(METRICS)

    if not data:
        raise RuntimeError("No training metrics found.")

    epochs = np.array([
        int(r["epoch"]) for r in data
    ])

    train_loss = np.array([
        float(r["train_loss"]) for r in data
    ])

    val_loss = np.array([
        float(r["val_loss"]) for r in data
    ])

    train_acc = 100 * np.array([
        float(r["train_accuracy"]) for r in data
    ])

    val_acc = 100 * np.array([
        float(r["val_accuracy"]) for r in data
    ])

    # Training and validation accuracy.
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(
        epochs, train_acc,
        "k-", marker="o", markersize=3,
        label="Training"
    )

    ax.plot(
        epochs, val_acc,
        "k--", marker="s", markersize=3,
        label="Validation"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy")

    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "accuracy")

    # Training and validation cross-entropy loss.
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(
        epochs, train_loss,
        "k-", marker="o", markersize=3,
        label="Training"
    )

    ax.plot(
        epochs, val_loss,
        "k--", marker="s", markersize=3,
        label="Validation"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training and Validation Loss")

    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "loss")


def select_examples(data, count=NUM_EXAMPLES):
    selected = []
    seen_labels = set()

    # Prefer examples with different true labels.
    for row in data:
        label = int(row["true_label"])

        if label not in seen_labels:
            selected.append(row)
            seen_labels.add(label)

        if len(selected) == count:
            return selected

    # Fill remaining positions if necessary.
    for row in data:
        if row not in selected:
            selected.append(row)

        if len(selected) == count:
            break

    return selected


def crop_digit(image, index):
    gx = index % GRID_COLS
    gy = index // GRID_COLS

    x0 = gx * DIGIT_SIZE
    y0 = gy * DIGIT_SIZE

    digit = image[
        y0:y0 + DIGIT_SIZE,
        x0:x0 + DIGIT_SIZE
    ]

    if digit.shape != (DIGIT_SIZE, DIGIT_SIZE):
        raise ValueError(
            f"Digit index {index} is missing from the contact sheet."
        )

    return digit


def plot_classifications(image):
    data = load_csv(PREDICTIONS)

    if not data:
        raise RuntimeError("No prediction records found.")

    correct = [
        row for row in data
        if int(row["true_label"]) == int(row["predicted_label"])
    ]

    incorrect = [
        row for row in data
        if int(row["true_label"]) != int(row["predicted_label"])
    ]

    groups = []

    if correct:
        groups.append((
            "Correct Classifications",
            select_examples(correct)
        ))

    if incorrect:
        groups.append((
            "Incorrect Classifications",
            select_examples(incorrect)
        ))

    nrows = len(groups)
    ncols = max(len(items) for _, items in groups)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.1 * ncols, 2.7 * nrows),
        squeeze=False
    )

    panel = 0

    for row_idx, (group_name, items) in enumerate(groups):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(items):
                ax.axis("off")
                continue

            item = items[col_idx]

            index = int(item["index"])
            true_label = int(item["true_label"])
            pred_label = int(item["predicted_label"])

            digit = crop_digit(image, index)

            ax.imshow(
                digit,
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest"
            )

            ax.set_xticks([])
            ax.set_yticks([])

            # Thin borders and alphabetical panel labels.
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.8)

            ax.set_title(
                f"({chr(97 + panel)})",
                loc="left",
                fontsize=10,
                pad=5
            )

            ax.set_xlabel(
                f"True: {true_label}\nPredicted: {pred_label}",
                fontsize=9,
                labelpad=7
            )

            panel += 1

    # Use row headings when both correct and incorrect samples exist.
    if nrows == 2:
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.85,
            bottom=0.10,
            wspace=0.35,
            hspace=0.9
        )

        fig.text(
            0.5, 0.96,
            groups[0][0],
            ha="center",
            va="center",
            fontsize=11
        )

        fig.text(
            0.5, 0.50,
            groups[1][0],
            ha="center",
            va="center",
            fontsize=11
        )

    else:
        fig.suptitle(
            groups[0][0],
            fontsize=11,
            y=0.98
        )

        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.83,
            bottom=0.27,
            wspace=0.35
        )

    save_figure(fig, "classifications")

    accuracy = 100 * len(correct) / len(data)

    print(
        f"Predictions: {len(data)} samples | "
        f"{len(correct)} correct | "
        f"{len(incorrect)} incorrect | "
        f"{accuracy:.2f}% accuracy"
    )


def main():
    print("Generating FNN figures...\n")

    # Plot training history.
    plot_metrics()

    # Export the original digit grid and prepare its pixels.
    image = load_digit_sheet()

    # Plot representative classification examples.
    plot_classifications(image)

    print(f"\nAll figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
