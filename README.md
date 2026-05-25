# Kiron — KI Runtime Orchestration Node

KI-Backend-Plattform für LLM-Inferenz, Embeddings und Document Understanding: Reverse-Proxy, Monitoring-Dashboard, OpenAI-kompatible API und GPU-Orchestrierung für mehrere Inferenz-Dienste auf einer einzelnen GPU.

Kiron wurde als Backend für [Kiara](https://kiara-ai.de) (Open-Source-RAG-Plattform) entwickelt, ist aber eigenständig nutzbar: Jede Anwendung, die Ollama-, OpenAI- oder Docling-APIs spricht, kann Kiron als Backend verwenden.

## Komponenten

| Dienst | Aufgabe | Port |
|--------|---------|------|
| `kiron-proxy` | Ollama-Reverse-Proxy + Dashboard + OpenAI-kompatible API | 11434 / 8505 / 11440 |
| `kiron-docling` | Docling On-Demand-Proxy (Document Understanding) | 5001 |
| `kiron-embeddings` | Sentence-Transformers Embedding-Service | 11436 |
| `kiron-deberta` | Cross-Encoder: Reranking + NLI | 11437 |
| Ollama (Docker) | LLM-Inferenz-Backend | 11435 |

`kiron-common` ist ein geteiltes Python-Paket (u.a. Ollama-Kompatibilitäts-Layer), das von den Diensten importiert wird. Der VRAM wird zwischen Ollama, Embeddings, Docling und DeBERTa koordiniert (Referenz-Hardware: NVIDIA RTX 3060, 12 GiB).

## Voraussetzungen

- Linux mit systemd
- NVIDIA-GPU + Treiber, Docker + Docker Compose (für den Ollama-Container)
- Python 3.11+
- MariaDB/MySQL (Metrik-Datenbank)

## Architektur: zwei Verzeichnisse

Kiron trennt Quellcode und Laufzeit:

```
<repo>/            ← Quelle (hier klonen, hier entwickeln)
/usr/lib/kiron/    ← Produktiv-Runtime (Dienste laufen hier, inkl. venvs)
```

Änderungen im Quellverzeichnis werden per Deploy-Skript in die Runtime übernommen. Die venvs existieren nur in der Runtime, nicht im Repo.

## Setup

```bash
# 1. Klonen (Beispielpfad)
git clone <repo-url> /opt/kiron && cd /opt/kiron

# 2. Datenbank-Konfiguration aus Vorlage erzeugen und ausfüllen
cp data/db_config.json.example data/db_config.json
# -> data/db_config.json editieren (Host, User, Passwort, DB)

# 3. Host-Konfigurationen installieren (apt-Pins, NVIDIA Drop-ins)
sudo bash scripts/install-system-configs.sh

# 4. Erst-Deploy + venvs anlegen + Restart (sichere Reihenfolge)
bash scripts/deploy-local.sh
bash scripts/setup-venvs.sh
bash scripts/deploy-local.sh --restart

# 5. systemd-Units installieren und Ollama-Container starten
sudo bash scripts/install-systemd.sh
cd /usr/lib/kiron/docker && docker compose up -d
```

Ollama-Kompatibilität wird vor Deploys über ein Gate geprüft:

```bash
python scripts/check-ollama-compat.py --image <ollama-image>
```

## Nutzung

- Dashboard: `http://localhost:8505`
- Ollama-Proxy: `http://localhost:11434`
- OpenAI-kompatible API: `http://localhost:11440/v1/` (Bearer-Auth)

```bash
curl http://localhost:11440/v1/chat/completions \
  -H "Authorization: Bearer sk-..." -H "Content-Type: application/json" \
  -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Hi"}]}'
```

## Tests

Die Test-Suite läuft per `pytest` in den jeweiligen Service-Verzeichnissen und ist zusätzlich als Selbsttest über das Dashboard ausführbar.

```bash
cd services/kiron-proxy && python -m pytest
```

## Lizenz

[GNU AGPL-3.0](LICENSE) — Copyright (C) 2026 Andreas Alber
