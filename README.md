# street-pattern-classifier

Classifies urban street-pattern graph samples.

## Scheme

```mermaid
flowchart LR
    A[Inputs] --> B[Run: usage.ipynb]
    B --> C[Checked outputs]
    C --> D[Paper / thesis use]
```

## Main Result

![Main result](docs/readme_result.svg)

## Run

Entrypoint: `usage.ipynb`

Human:

```bash
pip install -r requirements.txt && jupyter notebook usage.ipynb
```

Agent:

Reuse classifier outputs; do not retrain unless dataset/model version is explicit.

## Publication

No standalone publication tracked.

## Next Steps / Heuristics

Heuristic: top-1 classes are useful summaries, but probability mixtures are better for maps.
