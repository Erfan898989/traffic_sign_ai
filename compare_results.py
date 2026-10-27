from pathlib import Path
import json
import csv


RUNS = {
    "CNN": Path("runs/gtsrb_cnn"),
    "MobileNetV3-Small": Path("runs/gtsrb_mobilenet_full"),
    "ResNet18": Path("runs/gtsrb_resnet18"),
}


def load_metrics(folder):
    metrics_file = folder / "metrics.json"

    if not metrics_file.exists():
        return None

    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_value(data, *keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def main():

    results = []

    for model_name, folder in RUNS.items():

        metrics = load_metrics(folder)

        if metrics is None:
            print(
                f"[!] {model_name}: metrics.json پیدا نشد"
            )
            continue

        results.append({
            "Model": model_name,
            "Accuracy": get_value(
                metrics,
                "accuracy"
            ),
            "Precision": get_value(
                metrics,
                "precision_macro",
                "precision"
            ),
            "Recall": get_value(
                metrics,
                "recall_macro",
                "recall"
            ),
            "F1-score": get_value(
                metrics,
                "f1_macro",
                "f1"
            ),
        })

    if not results:
        print(
            "هیچ نتیجه‌ای برای مقایسه پیدا نشد."
        )
        return

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        f"{'Model':<22}"
        f"{'Accuracy':<14}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1-score':<14}"
    )

    print("-" * 80)

    for result in results:

        def fmt(value):
            if value is None:
                return "-"

            return f"{value * 100:.2f}%"

        print(
            f"{result['Model']:<22}"
            f"{fmt(result['Accuracy']):<14}"
            f"{fmt(result['Precision']):<14}"
            f"{fmt(result['Recall']):<14}"
            f"{fmt(result['F1-score']):<14}"
        )

    print("=" * 80)

    # ذخیره CSV
    output = Path(
        "runs/model_comparison.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"جدول ذخیره شد: {output}"
    )


if __name__ == "__main__":
    main()