# Merging and resuming runs

`boltzgen merge` reads analyzed designs from their metrics CSV and discovers
backbone files independently when that directory has no metrics CSV. This keeps
backbone IDs separate from the sequence IDs created by generating multiple
sequences per backbone. A metrics CSV with no rows is rejected; rerun analysis
before merging that source. Unanalyzed coordinate files do not count as designs
available for filtering.

Source names are normalized into unique prefixes. Repeated source paths are
included once. Dotted and numeric-looking design IDs keep their exact identity,
and metadata/native companions are preserved, using modern filenames in the
destination even for legacy inputs. Replacing a merged design also removes its
optional companions when they are absent from the replacement source. Merged files are
independent copies, so rewriting a merged prediction cannot change a source run.
An incomplete replacement is rejected if a preexisting legacy metadata alias could
be selected for it. Restore its source metadata or use a fresh directory; aliases
are not deleted because their ownership can be ambiguous.
Legacy lookup also rejects a metadata file that has its own matching coordinate
file, preventing an incomplete `target_gen` design from borrowing metadata from
a separate `target_metadata` design. A complete modern pair always takes priority.
Custom molecule definitions are copied too. Definitions sharing one CCD name must
agree in ordered atoms, bonds, stereochemistry, and saved properties. Reference
conformers can differ because parsing the same SMILES generates them stochastically;
the first source's definition and conformers are retained. This preserves chemical
identity, but does not promise identical numerical reference features if inference
is rerun. Other definition differences are rejected rather than silently choosing
between different molecules. An existing destination must also contain compatible
definitions, since old coordinates may still refer to them; use a fresh output
directory when changing molecule identities. An equivalent old reference conformer
is replaced by the first current source's definition.

For inverse folding, `--reuse` skips a backbone when every expected sequence has
both a regular CIF file and a regular NPZ file. Incomplete backbones remain
scheduled. The writer preserves completed sibling pairs while repairing missing
pairs, so existing siblings retain their matching downstream results.

Use the same inverse-fold sequence multiplicity when resuming an output directory.
The resume check uses the writer's exact numeric padding. Complete indexed pairs
with conflicting padding are rejected before inverse folding; use a fresh output
directory for that naming transition. This check is not a general migration of
old output directories: it does not resolve ambiguous bare/indexed names, remove
old metric rows, or support arbitrary changes of sequence counts and configuration.
Outputs manually modified after downstream steps completed also require those
downstream steps to be recomputed.
