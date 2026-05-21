from __future__ import annotations

from pathlib import Path
import typer
import pandas as pd

from marco_copilot.config import load_campaign
from marco_copilot.spec_generator import generate_specs
from marco_copilot.run_manager import create_run_scripts
from marco_copilot.developability import features
from marco_copilot.score_contacts import score_structure
from marco_copilot.report import write_reports

app = typer.Typer()


@app.command()
def plan(campaign: Path = Path("campaign.yaml"), out: Path = Path("specs")):
    cfg = load_campaign(campaign)
    created = generate_specs(cfg, out)
    typer.echo(f"generated {len(created)} specs")


@app.command()
def design(specs_dir: Path = Path("specs"), runs_dir: Path = Path("runs"), budget: int = 100, num_designs: int = 50):
    spec_paths = sorted(specs_dir.glob("*.yaml"))
    scripts = create_run_scripts(spec_paths, runs_dir, budget, num_designs)
    typer.echo(f"created {len(scripts)} run scripts")


@app.command()
def score(input_csv: Path = Path("designs.csv"), out_csv: Path = Path("scores.csv")):
    df = pd.read_csv(input_csv)
    rows = []
    for _, r in df.iterrows():
        s = score_structure(r["structure_path"], r.get("binder_chain", "B"), r.get("target_chain", "A"))
        d = features(r["binder_seq"])
        rows.append({**r.to_dict(), **s, **d})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    typer.echo(f"wrote {out_csv}")


@app.command()
def report(scores_csv: Path = Path("scores.csv"), out_prefix: Path = Path("reports/summary")):
    rows = pd.read_csv(scores_csv).to_dict(orient="records")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_p, xlsx_p = write_reports(rows, out_prefix)
    typer.echo(f"wrote {csv_p} and {xlsx_p}")


@app.command()
def next(scores_csv: Path = Path("scores.csv"), top_n: int = 10):
    df = pd.read_csv(scores_csv)
    sort_cols = [c for c in ["num_contacts", "pI_est", "hydrophobic_fraction"] if c in df.columns]
    top = df.sort_values(sort_cols, ascending=[False, False, True]).head(top_n)
    typer.echo(top[[c for c in ["design_id", "num_contacts", "pI_est", "hydrophobic_fraction"] if c in top.columns]].to_string(index=False))
    typer.echo("Suggestion: increase hotspot residue constraints and keep hydrophobic_fraction < 0.45")


if __name__ == "__main__":
    app()
