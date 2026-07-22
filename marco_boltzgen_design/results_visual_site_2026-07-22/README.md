# MARCO SRCR Boltz Result Viewer

This is a static Boltz-style result viewer for the MARCO SRCR nanobody candidate panel.
It shows one selected candidate at a time with the MARCO target sequence, binder sequence,
validation metrics, counter-screen metrics, and an interactive 3D viewer loaded from a
site-local copy of the candidate's MARCO complex CIF file.

Open `index.html` directly in a browser, or serve this folder locally:

```bash
python3 -m http.server 8766
```

The site uses:

- `data/candidates.js`: 30-candidate panel data generated from `panel_30_candidates.csv`
- `cifs/`: site-local copies of the 30 MARCO complex CIF files
- 3Dmol.js from `https://3Dmol.org/build/3Dmol-min.js` for browser-side CIF rendering
