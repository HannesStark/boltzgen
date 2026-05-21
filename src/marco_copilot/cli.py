from __future__ import annotations

from pathlib import Path
import typer
import pandas as pd

from marco_copilot.config import load_campaign, dump_yaml
from marco_copilot.spec_generator import generate_specs
from marco_copilot.run_manager import create_run_scripts
from marco_copilot.developability import features
from marco_copilot.score_contacts import score_structure
from marco_copilot.report import write_reports

app = typer.Typer(help="MARCO design campaign manager")


@app.command()
def plan(campaign: Path = Path("campaign.yaml"), out: Path = Path("campaign_set")):
    """Generate BoltzGen spec YAMLs and campaign artifacts."""
    cfg = load_campaign(campaign)
    specs_dir = out / "specs"
    created = generate_specs(cfg, specs_dir)
    summary = {
        "mode": cfg.get("mode", "cross-reactive"),
        "num_tasks": len(created),
        "targets": cfg.get("targets", {}),
    }
    dump_yaml(out / "plan_summary.yaml", summary)
    typer.echo(f"generated {len(created)} specs at {specs_dir}")


@app.command()
def design(campaign_dir: Path = Path("campaign_set"), runs_dir: Path = Path("runs"), budget: int = 100, num_designs: int = 50, devices: int = 2):
    """Create run folders and execution scripts; does not execute by default."""
    spec_paths = sorted((campaign_dir / "specs").glob("*.yaml"))
    scripts = create_run_scripts(spec_paths, runs_dir, budget, num_designs, devices=devices)
    typer.echo(f"created {len(scripts)} run scripts (devices={devices})")


@app.command()
def score(input_csv: Path = Path("designs.csv"), out_csv: Path = Path("scores.csv")):
    """Score structures and compute cross-species/developability features."""
    df = pd.read_csv(input_csv)
    rows = []
    for _, r in df.iterrows():
        s = score_structure(r["structure_path"], r.get("binder_chain", "B"), r.get("target_chain", "A"))
        d = features(r["binder_seq"])
        conserved = set(int(x) for x in str(r.get("conserved_target_residues", "")).split(",") if str(x).strip().isdigit())
        target_contacts = {p[1] for p in s["contact_pairs"]}
        conserved_fraction = len(target_contacts & conserved) / max(len(target_contacts), 1)
        rows.append({**r.to_dict(), **{k: v for k, v in s.items() if k != "contact_pairs"}, **d, "conserved_contact_fraction": round(conserved_fraction, 3)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    typer.echo(f"wrote {out_csv}")


@app.command()
def report(scores_csv: Path = Path("scores.csv"), out_prefix: Path = Path("reports/summary")):
    rows = pd.read_csv(scores_csv).to_dict(orient="records")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_p, xlsx_p, html_p = write_reports(rows, out_prefix)
    typer.echo(f"wrote {csv_p}, {xlsx_p}, {html_p}")


@app.command(name="next")
def next_round(scores_csv: Path = Path("scores.csv"), top_n: int = 10, max_charge: int = 6):
    df = pd.read_csv(scores_csv)
    if "cross_reactivity_score" not in df.columns:
        df["cross_reactivity_score"] = df.get("conserved_contact_fraction", 0.0)
    df["priority_score"] = (
        df.get("num_contacts", 0) * 0.5
        + df.get("cross_reactivity_score", 0) * 50
        - df.get("hydrophobic_fraction", 0) * 10
        - (df.get("net_charge", 0).clip(lower=max_charge) - max_charge)
    )
    top = df.sort_values("priority_score", ascending=False).head(top_n)
    cols = [c for c in ["design_id", "num_contacts", "cross_reactivity_score", "net_charge", "priority_score"] if c in top.columns]
    typer.echo(top[cols].to_string(index=False))
    high_charge = int((df.get("net_charge", 0) > max_charge).sum())
    typer.echo(f"Suggestion: keep conserved epitope tasks, enforce net_charge <= {max_charge}; {high_charge} candidates exceeded charge threshold.")


if __name__ == "__main__":
    app()
