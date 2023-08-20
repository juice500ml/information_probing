"""
Commonphone dataset (v1.0) preparation code.
Modified from https://github.com/juice500ml/dysarthria-gop/blob/1ead704/dataset.py

Commonphone dataset can be found at https://zenodo.org/record/5846137
Can be downloaded with:
    wget -O /path/to/save/cp-1-0.tgz --quiet \
        'https://zenodo.org/record/5846137/files/cp-1-0.tgz?download=1'
"""

import argparse
from pathlib import Path
from itertools import product

import pandas as pd
import textgrids


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset.")
    parser.add_argument("--output_path", type=Path, help="Output csv folder")
    return parser.parse_args()


def _parse(commonphone_path, max_length=10):
    langs = ("de", "en", "es", "fr", "it", "ru")
    splits = ("train", "dev", "test")
    phonemes, words = [], []

    for lang, split in product(langs, splits):
        cp_path = commonphone_path / lang
        df = pd.read_csv(cp_path / f"{split}.csv")

        skipped_count = 0
        for _, row in df.iterrows():
            audio_file_name = cp_path / "wav" / row["audio file"].replace(".mp3", ".wav")
            grid_file_name = cp_path / "grids" / f"{audio_file_name.stem}.TextGrid"

            grid = textgrids.TextGrid(grid_file_name)
            if grid.xmax > max_length:
                skipped_count += 1
                continue

            for g in grid["MAU"]:
                phonemes.append({
                    "audio": audio_file_name,
                    "language": lang,
                    "id": row["id"],
                    "split": split,
                    "sentence": row["text"],
                    "min": g.xmin,
                    "max": g.xmax,
                    "text": g.text,
                })
            for g in grid["ORT-MAU"]:
                words.append({
                    "audio": audio_file_name,
                    "language": lang,
                    "id": row["id"],
                    "split": split,
                    "sentence": row["text"],
                    "min": g.xmin,
                    "max": g.xmax,
                    "text": g.text.lower(),
                })

        print(f"{lang}/{split}: Original size: {len(df)}, skipped audio count: {skipped_count} ({skipped_count / len(df):.3%})")

    return pd.DataFrame(phonemes), pd.DataFrame(words)


def _filter_words(df):
    df["lang_text"] = df.apply(lambda x: f"{x.language}_{x.text}", axis=1)
    long_enough_mask = ((df["max"] - df["min"]) >= 0.025) & ((df["max"] - df["min"]) < 2)
    count_words = df[(df.text != "") & long_enough_mask].lang_text.value_counts()
    target_words = set(count_words[count_words >= 1000].keys())
    new_df = df[df.lang_text.isin(target_words) & long_enough_mask].copy().reset_index(drop=True)
    new_df = new_df.rename(columns={"text": "original_text"}).rename(columns={"lang_text": "text"})

    print(f"Original word count: {df[df.text != ''].lang_text.nunique()}, # of samples: {len(df[df.text != ''])}")
    print(f"New word count: {new_df.text.nunique()}, # of samples: {len(new_df)}")
    return new_df


def _filter_phonemes(df):
    long_enough_mask = ((df["max"] - df["min"]) >= 0.025) & ((df["max"] - df["min"]) < 2)
    count_words = df[(df.text != "(...)") & long_enough_mask].text.value_counts()
    target_words = set(count_words[count_words >= 1000].keys())
    new_df = df[df.text.isin(target_words) & long_enough_mask].copy().reset_index(drop=True)

    print(f"Original phoneme count: {df[df.text != '(...)'].text.nunique()}, # of samples: {len(df[df.text != '(...)'])}")
    print(f"New phoneme count: {new_df.text.nunique()}, # of samples: {len(new_df)}")
    return new_df


if __name__ == "__main__":
    args = _get_args()
    df_phonemes, df_words = _parse(args.dataset_path)
    _filter_phonemes(df_phonemes).to_csv(args.output_path / f"commonphone_phonemes.csv.gz", index=False, compression="gzip")
    _filter_words(df_words).to_csv(args.output_path / f"commonphone_words.csv.gz", index=False, compression="gzip")
