# marco-copilot MVP (Design Campaign Manager)

## Commands
- `marco-copilot plan --campaign campaign.yaml --out campaign_setD`
- `marco-copilot design --campaign-dir campaign_setD --num-designs 100 --budget 20 --devices 2`
- `marco-copilot score --input-csv designs.csv --out-csv scores.csv`
- `marco-copilot report --scores-csv scores.csv --out-prefix reports/summary`
- `marco-copilot next --scores-csv scores.csv --top-n 20`

## campaign.yaml (minimal)
```yaml
mode: cross-reactive
targets:
  human: targets/human_marco_af.pdb
  mouse: targets/mouse_marco_srcr.cif
tasks:
  - name: setD_patch2_scaffold_8coh
    target: targets/human_marco_af.pdb
    scaffold: scaffolds/8coh.cif
    epitope: [423, 425, 432, 461, 467]
```


> Note: BoltzGen `run` manual states `--devices` defaults to all available GPUs.
> `marco-copilot` passes `--devices` explicitly (default `2`) to ensure both GPUs are used on dual-GPU nodes.

## Regression checks
- `bash -n marco_boltzgen_design/runs/run_nanobody_campaign.sh`
- `PYTHONPATH=src pytest -q tests/marco_copilot`

