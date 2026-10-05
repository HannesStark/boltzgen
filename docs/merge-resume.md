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
Custom molecule definitions are copied too. Different serialized definitions
sharing one CCD name are rejected, since selecting either definition silently
could change how the other source is interpreted.

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
