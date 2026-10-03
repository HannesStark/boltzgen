# LigandMPNN vendor notice

This directory contains a copy of `model_utils.py` from the
[LigandMPNN](https://github.com/dauparas/LigandMPNN) project by Justas
Dauparas.

- **Upstream URL:** https://github.com/dauparas/LigandMPNN
- **Upstream commit:** `26ec57ac976ade5379920dbd43c7f97a91cf82de`
- **License:** MIT (see `LICENSE`)
- **Files vendored:** `model_utils.py`
- **Modifications:** trailing whitespace removed; sampling logits are centered
  and scaled in float64 to support extreme finite positive temperatures with
  hard amino-acid exclusions. Model architecture and weights are unchanged.

The upstream repository does not publish a Python distribution to PyPI, so we
vendor the single self-contained module that defines the `ProteinMPNN` /
`SolubleMPNN` model class. Pre-trained checkpoints (e.g.
`solublempnn_v_48_020.pt`) are downloaded separately at runtime; they are not
shipped with BoltzGen.

If a packaged release becomes available upstream, switch this vendor to a
runtime dependency and remove this directory.
