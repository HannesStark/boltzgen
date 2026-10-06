# NumPy and Numba compatibility

BoltzGen accepts `numpy>=2.0.2,<2.5` and `numba>=0.61.0,<0.66`.
The resolver also applies Numba's NumPy requirements, so not every arbitrary
pair inside these bounds can be installed together. These bounds preserve the
previous NumPy 2.0.2 / Numba 0.61.0 combination and allow Python 3.13 installations.

The following combinations passed the complete available CPU test suite on
Linux x86-64, with real dependencies and Torch 2.14.1:

| Python | NumPy | Numba | llvmlite |
| --- | --- | --- | --- |
| 3.11.15 | 2.0.2 | 0.61.0 | 0.44.0 |
| 3.11.15 | 2.2.6 | 0.61.2 | 0.44.0 |
| 3.11.15 | 2.3.5 | 0.62.1 | 0.45.1 |
| 3.11.15 | 2.4.6 | 0.65.1 | 0.47.0 |
| 3.13.15 | 2.1.3 | 0.61.0 | 0.44.0 |
| 3.13.15 | 2.4.6 | 0.65.1 | 0.47.0 |

Each combination also ran the compiled Numba MSA transformation and checked
its arrays against explicit expected values. Structure insertion and fusion
were compared against the previous pinned runtime, including coordinates,
indices, bonds, masks and ensemble metadata. Real YAML parsing tests exercise
file insertion, protein fusion and file fusion, with unaffected controls.

This is CPU application qualification, not a guarantee for every version pair,
operating system, Python release or GPU stack. Tests that require external ESM
or SolubleMPNN assets were skipped. Model inference and training were not run.
Numba's dependency metadata alone does not establish BoltzGen compatibility.
The tested combinations above qualify these bounds, including their endpoints;
intermediate versions and every resolver-compatible pair were not individually
tested. Extending either upper bound requires new application qualification.

To reproduce the repository tests in a compatible development environment:

```sh
python -m pytest tests -q
python -m pytest tests/test_structure_insert_fuse.py tests/test_structure_transform_boundaries.py -q
```

The structure regressions treat deprecated array-to-scalar conversions as
errors even on older NumPy, so a pinned environment can catch their return.
