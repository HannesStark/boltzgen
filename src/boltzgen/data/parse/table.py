"""Read metric tables without treating identifiers or sequences as missing data."""

from pathlib import Path

import pandas as pd


_TEXT_COLUMNS = {
    "id",
    "file_name",
    "target_id",
    "query",
    "target",
    "file",
    "clusters_05_tmscore",
}


def read_metrics_table(
    path: str | Path, *, sep: str = ",", names: list[str] | None = None
) -> pd.DataFrame:
    """Preserve literal text fields and normal numeric/missing-value inference.

    Pandas 3 can apply missing-value inference after a text converter. Read
    text columns separately with that inference disabled, leaving scores alone.
    Empty sequence cells remain missing, rather than becoming valid sequences.
    """
    with Path(path).open(encoding="utf-8", newline="") as stream:
        frame = pd.read_csv(stream, sep=sep, names=names)
        sequence_columns = [
            column
            for column in frame.columns
            if column in {"sequence", "full_sequence", "designed_chain_sequence"}
            or column.startswith(("designed_sequence", "full_sequence_"))
        ]
        text_columns = [
            column
            for column in frame.columns
            if column in _TEXT_COLUMNS or column in sequence_columns
        ]
        if text_columns:
            stream.seek(0)
            text = pd.read_csv(
                stream,
                sep=sep,
                names=names,
                usecols=text_columns,
                dtype=str,
                keep_default_na=False,
            )
            for column in text_columns:
                values = text[column]
                if column in sequence_columns:
                    values = values.replace("", None)
                frame[column] = values
    return frame
