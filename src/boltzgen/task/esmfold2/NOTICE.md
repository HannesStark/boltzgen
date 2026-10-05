The inference adapter in `acceleration.py` implements mask caching (U1),
graph replay of the recycling trunk, and graph replay of the deterministic
diffusion forward from Anthropic's public ESMFold2 optimization kit:

https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/esmfold2/opt/forward/fast_inference/driver/ef2_opt.py

Copyright 2026 Anthropic, PBC. Licensed under Apache-2.0; see
`LICENSE.anthropic`. This is a modified adaptation for native
`esm.models.esmfold2` in ESM 3.4.1.post1. It uses instance-local wrappers,
private graph pools, per-request lifetimes, and the original numerical backend.
It does not bundle the kit's fast kernels, old Transformers fork, or model weights.

`sampling.py` and the attention expression in `acceleration.py` are modified
from `esm/models/esmfold2/layers.py` in ESM 3.4.1.post1 (PyPI wheel SHA-256
`f9e62b363519860d27762989871a531cdbd02d69c824cc8c8a1047b16dce3ef6`).
The source file carries Copyright 2026 Biohub. All rights reserved, and an
Apache-2.0 header; the Apache-2.0 text is included in `LICENSE.anthropic`.
The wheel's package-level MIT notice is also retained in `LICENSE.esm`;
it does not replace the source file's Apache notice.
The sampler applies the kit's U2 schedule-transfer change;
the attention expression hoists its unchanged boolean mask.

The public kit's graph implementation also derives from the Biohub fork of
Hugging Face Transformers at ef32577f: Copyright 2018- The Hugging Face team;
Copyright 2026 Biohub. All rights reserved. Apache-2.0. No functions from that
older model implementation are reissued here; the matching native ESM functions
are used instead. Model weights remain governed by their publishers' licenses.
