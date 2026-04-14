# Claude Code Instructions


*** Project Context ***

Adding a (potential a set of) additional feature for boltzgen (https://github.com/HannesStark/boltzgen): 

Add an additional step along the workflow that filter generated design by developability of the design sequences. Currently, developability I would like to prioritize in accessing is first, selectivity against a set of decoys targets (binder should be of high binding affinity to true target but low binding affinity to decoys); second, would tagging (e.g. HIS-tag) that are crucial to expression affecting binding and stability. 

## Session Log

At the start of every session, read `docs/session_log.md` to recall prior context.

At the end of each work section (when a task or feature is complete, or the user signals a pause), append an entry to `docs/session_log.md` in this format — **without asking the user first**:

```
## YYYY-MM-DD

- Brief summary of what was done.
- Any relevant decisions, blockers, or next steps.
```

Do not duplicate entries. If today's date already has a section, append bullet points under it rather than creating a new heading.
