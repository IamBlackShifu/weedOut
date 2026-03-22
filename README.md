# WeedOut – Cyberbullying Detection & Profiling

NLP-powered system for detecting, profiling, and moderating cyberbullying on Twitter-like social media platforms.

## Overview

WeedOut ingests text posts in real time, classifies them as **bullying / non-bullying / uncertain**, maintains per-user risk profiles, and enforces graduated moderation actions — all within a lightweight Python package backed by a FastAPI REST service.

## Architecture

```
Raw post text
     │
     ▼
┌──────────────────┐
│  TextCleaner     │  Normalise, remove URLs, demojize, handle @mentions / #hashtags
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ FeatureExtractor │  TF-IDF n-grams + structural features (caps ratio, exclamations,
│                  │  mention count, toxicity lexicon hits)
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│ BullyingClassifier   │  Calibrated LinearSVC → label + probability
│ (LinearSVC + cal.)   │  Labels: bullying | non-bullying | uncertain
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│  UserProfiler    │  Time-decayed risk scoring, targeting breadth
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ModerationEngine │  Policy: WARN / SHADOW_BAN / SUSPEND / ESCALATE / NONE
└──────────────────┘
```

## Components

| Module | Description |
|---|---|
| `weedout.preprocessing.TextCleaner` | Tokenisation, URL removal, emoji conversion, mention/hashtag handling |
| `weedout.features.FeatureExtractor` | TF-IDF n-gram + structural feature pipeline |
| `weedout.models.BullyingClassifier` | Calibrated LinearSVC classifier with save/load |
| `weedout.profiling.UserProfile` / `UserProfileStore` | Per-user time-decayed risk scoring |
| `weedout.moderation.ModerationEngine` | Policy enforcement engine |
| `weedout.api.app` | FastAPI application |

## Quick Start

### Installation

```bash
pip install -e .
```

### Run the API server

```bash
uvicorn weedout.api.app:app --reload
```

The interactive API docs are available at `http://localhost:8000/docs`.

### Train the classifier (via API)

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I hate you so much, you are disgusting",
      "You are worthless, nobody likes you",
      "Have a great day, hope you enjoy it",
      "Thanks for your help!"
    ],
    "labels": ["bullying", "bullying", "non-bullying", "non-bullying"]
  }'
```

### Classify a post

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "You are such a loser, go away!"}'
```

### Moderate a post (classify + update user profile)

```bash
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_abc123",
    "text": "You are so stupid!",
    "targets": ["@victim_handle"]
  }'
```

### Get a user risk profile

```bash
curl http://localhost:8000/profile/user_abc123
```

## Moderation Policy

| User Risk Level | Post Label | Action |
|---|---|---|
| HIGH | bullying | SUSPEND |
| HIGH | uncertain | ESCALATE |
| MEDIUM | bullying | SHADOW_BAN |
| MEDIUM | uncertain | ESCALATE |
| LOW | bullying | WARN |
| any | uncertain | ESCALATE |
| any | non-bullying | NONE |

**Risk levels** are computed from a time-decayed aggregate of the user's post history:
- `HIGH` ≥ 0.8
- `MEDIUM` ≥ 0.5
- `LOW` < 0.5

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/train` | Train / retrain the classifier |
| `POST` | `/classify` | Classify a single post |
| `POST` | `/moderate` | Classify + update user profile + get moderation decision |
| `GET` | `/profile/{user_id}` | Retrieve a user's risk profile |
| `GET` | `/high-risk-users` | List all HIGH-risk users |

## Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Ethical Constraints

- **Privacy**: No more personal data is stored than necessary; user IDs should be pseudonymised by the calling system.
- **Non-discrimination**: The toxicity lexicon and model features do not target any demographic group.
- **Human oversight**: Ambiguous (`uncertain`) posts are always escalated to human moderators rather than automatically actioned.
- **Transparency**: All moderation decisions include a human-readable `reason` field.
