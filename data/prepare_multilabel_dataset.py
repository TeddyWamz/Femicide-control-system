import csv
from pathlib import Path

import pandas as pd

CATEGORIES = ["Physical_violence", "sexual_violence", "emotional_violence", "economic_violence"]

KEYWORDS = {
    "Physical_violence": [
        "slapped",
        "punched",
        "kicked",
        "beat",
        "hit",
        "grabbed my neck",
        "choked",
        "locked me in",
        "dragged",
        "bruises",
        "alinipiga",
        "alinichapa",
        "amenipiga",
        "alinishika kooni",
        "kunipiga",
        "kuchezea ngumi",
        "kuniua",
    ],
    "sexual_violence": [
        "rape",
        "raped",
        "sexual",
        "forced himself",
        "forced sex",
        "molest",
        "drugged",
        "without consent",
        "alinilazimisha kufanya ngono",
        "alinibaka",
        "kunigusa bila ridhaa",
        "hunilazimisha",
        "unyanyasaji wa kingono",
    ],
    "emotional_violence": [
        "insult",
        "belittle",
        "threat",
        "shout",
        "humiliate",
        "control me",
        "gaslight",
        "called me names",
        "ananitusi",
        "ananidharau",
        "vitisho",
        "kunidhalilisha",
        "maneno makali",
    ],
    "economic_violence": [
        "money",
        "salary",
        "allowance",
        "finances",
        "took my pay",
        "withheld",
        "blocked my job",
        "controls the bank",
        "mshahara",
        "ananinyima pesa",
        "ananizuia kufanya kazi",
        "haniruhusu kufanya kazi",
        "uthibiti wa fedha",
    ],
}

AUGMENT_ENGLISH_SUFFIX = [
    "Please help before it gets worse.",
    "I fear for my safety every day.",
]
AUGMENT_SWAHILI_SUFFIX = [
    "Ninaishi kwa hofu kila siku.",
    "Tafadhali naomba msaada kabla hali haijawa mbaya.",
]


def detect_additional_labels(text: str) -> set[str]:
    labels = set()
    lowered = text.lower()
    for category, keywords in KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            labels.add(category)
    return labels


def build_multilabel_dataset(src: pd.DataFrame) -> pd.DataFrame:
    records = []
    seen = set()
    for _, row in src.iterrows():
        text = row["text"].strip()
        norm = text.lower()
        if norm in seen:
            continue
        seen.add(norm)
        base_label = row["category"]
        labels = {cat: 0 for cat in CATEGORIES}
        labels[base_label] = 1
        detected = detect_additional_labels(text)
        for label in detected:
            labels[label] = 1
        records.append(
            {
                "text": text,
                "language": row.get("language", "unknown"),
                **labels,
            }
        )
    return pd.DataFrame(records)


def augment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    augmented_rows = []
    for _, row in df.iterrows():
        suffixes = AUGMENT_ENGLISH_SUFFIX if row["language"] == "english" else AUGMENT_SWAHILI_SUFFIX
        for suffix in suffixes:
            augmented_rows.append(
                {
                    **row.to_dict(),
                    "text": f'{row["text"].strip()} {suffix}',
                    "source": "augmented",
                }
            )
    original = df.copy()
    original["source"] = "original"
    return pd.concat([original, pd.DataFrame(augmented_rows)], ignore_index=True)


def main():
    raw_path = Path("final_training_data.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    cleaned = build_multilabel_dataset(df)
    augmented = augment_dataset(cleaned)
    augmented.to_csv(output_dir / "multilabel_training_data.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved {len(augmented)} rows to {output_dir / 'multilabel_training_data.csv'}")


if __name__ == "__main__":
    main()
