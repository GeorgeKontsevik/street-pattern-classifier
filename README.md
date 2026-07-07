# street-pattern-classifier

Block-graph street-pattern classifier. It builds graph/morphology features and predicts interpretable street-pattern labels.

## System Map

```mermaid
flowchart LR
    BLOCKS[urban blocks] --> GRAPH[block graph]
    GRAPH --> FEAT[morphology features]
    FEAT --> MODEL[classifier]
    MODEL --> LABELS[street-pattern labels]
```

## Main Image

![Street-pattern classifier workflow](docs/readme_project_map.svg)

## Run

Entrypoint: `usage.ipynb`

Human:

```bash
pip install -r requirements.txt && jupyter notebook usage.ipynb
```

Agent: inspect class balance and map samples, not only aggregate accuracy; mislabeled morphology is worse than missing labels.

## Publication

No standalone publication tracked; used by the street-pattern dissertation experiments.

## Next Steps / Heuristics

Heuristic: keep labels interpretable and stable across cities. Prefer a smaller class set over fragile fine-grained taxonomy.
