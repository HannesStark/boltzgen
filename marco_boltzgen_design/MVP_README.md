# marco-copilot MVP

## Commands
- `marco-copilot plan`: read `campaign.yaml` and generate BoltzGen spec YAMLs.
- `marco-copilot design`: create run folders and `run.sh` scripts.
- `marco-copilot score`: parse structures + compute simple contact/developability metrics.
- `marco-copilot report`: export CSV and Excel.
- `marco-copilot next`: summarize top candidates and next-round suggestions.

## Minimal campaign.yaml
```yaml
specs:
  - name: mouse_round1
    spec:
      version: 1
      target: mouse_marco
```
