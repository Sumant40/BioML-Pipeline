# BioML Pipeline 🧬

Gene expression cancer classifier using PySpark, TensorFlow, and PyTorch — all CPU, all containerised.

## Quick start
```bash
docker compose pull   # download images once
docker compose up     # run full pipeline
```

## Services
| Service      | Image               | Role                        |
|--------------|---------------------|-----------------------------|
| datagen      | python:3.11-slim    | Generate synthetic gene data|
| spark        | bitnami/spark:3.5   | Clean + normalise with Spark|
| tensorflow   | python:3.11-slim    | Train TF classifier         |
| pytorch      | python:3.11-slim    | Train PyTorch classifier    |
| ensemble     | python:3.11-slim    | Average predictions         |

## Approx disk usage
~3.3 GB total (CPU-only images)
