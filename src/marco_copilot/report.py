from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_reports(rows: list[dict], out_prefix: Path) -> tuple[Path, Path]:
    df = pd.DataFrame(rows)
    csv_path = out_prefix.with_suffix('.csv')
    xlsx_path = out_prefix.with_suffix('.xlsx')
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path
