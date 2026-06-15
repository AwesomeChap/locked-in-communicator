# LockedIn Communicator

**Inner Speech Decoding with Personalized Brain–Computer Interfaces**

Course project for **Advanced Systems Engineering (ASE 2026)** — *Jatin Kumar*

---

## Project overview

This project develops a prototype system that converts **inner speech** (imagined speech) into **text or simple commands** and improves reliability through **personalized decoding**. The goal is to support communication for people in a **locked-in state**, who are cognitively intact but unable to speak or move.

---

## Communication challenge

People who are **locked in** lose practical ways to communicate even though their cognition may be fully preserved. BCIs and neural decoding systems often fall short because:

- **Brain signals** tend to be **weak**, **noisy**, and **variable** across people and sessions (severity depends on modality and setup).
- **Error rates** can be high, which makes real-world use frustrating or unsafe.
- Systems often require **heavy user training**, which is burdensome for patients.
- **Inner speech decoding** is especially hard: useful patterns are **subtle** and **highly individual**, so generic, one-size-fits-all decoders struggle.

We want to explore whether **computational decoding** (including, if appropriate, learning-based methods) plus **system-side personalization** can shift more of the burden from the user to the system and make a small vocabulary (e.g. “yes”, “no”, “help”) more usable as a communication aid.

---

## Roadmap and progress

| Step | Status |
|------|--------|
| Define project scope and target vocabulary (e.g., yes/no/help) | in progress |
| Select and document dataset(s), benchmarks, and success metrics | - |
| Build preprocessing pipeline and create training-ready samples | - |
| Implement and test a baseline decoder | - |
| Add post-processing/refinement layer for prediction stability | - |
| Add user-level personalization/adaptation workflow | - |
| Run experiments and compare baseline vs personalized performance | - |
| Package a minimal end-to-end prototype demo and report findings | - |

---

## Expected outcomes

- A **working prototype** that recognizes a small set of imagined words or commands from **brain-derived signals** (exact pipeline TBD).
- Evidence that a **refinement or adaptation step** and/or **personalization** improves practical performance vs. a generic baseline.
- A **quantitative comparison** between baseline and personalized (or adapted) approaches.

---

## Why this approach

The emphasis is **system adaptation** rather than only **user adaptation**: the decoder should absorb **person-to-person and session-to-session variability** where possible. **Learning-based or statistical methods** may be needed for inner speech in realistic settings, but the final mix of classical signal processing and learning will follow the research phase and data constraints.

---

## Setup and run

### Backend

From `backend/`:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the simulated end-to-end BCI validation:

```bash
lockedin-validate
```

Or:

```bash
python3 scripts/validate_pipeline.py
```

Start the WebSocket server for the live dashboard:

```bash
lockedin-verification-server
```

For live Lab Streaming Layer ingestion, install the optional dependency:

```bash
pip install -e ".[lsl]"
```

### Frontend

From `frontend/` (in a second terminal, with the verification server running):

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). The dashboard connects to `ws://localhost:8765`.

---

## Repository structure

```
LOCKEDINCOMMUNICATOR/
├── backend/                  # Everything Python lives here
│   ├── .venv/
│   ├── src/                  # Core BCI processing engine
│   │   ├── pipeline/
│   │   ├── ingestion/
│   │   ├── preprocessing/
│   │   ├── features/
│   │   ├── inference/
│   │   ├── config.py
│   │   ├── validation.py
│   │   └── server.py         # WebSocket server
│   ├── config/               # Hardware / channel mappings
│   ├── scripts/              # Data simulation / validation entry points
│   ├── pyproject.toml
│   └── requirements.txt
│
├── frontend/                 # Everything React lives here
│   ├── src/                  # Components, hooks, styles
│   ├── public/               # Static assets
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
└── README.md                 # Global project documentation
```

### Backend modules

- `backend/src/config.py` — signal, filter, epoching, classifier, simulator, and feedback settings.
- `backend/src/ingestion/` — LSL ingestion and mock EEG stream generation.
- `backend/src/preprocessing/` — bandpass, notch rejection, and epoch slicing.
- `backend/src/features/` — Common Spatial Patterns spatial filtering.
- `backend/src/inference/` — shrinkage LDA classification and feedback dispatch.
- `backend/src/pipeline/` — end-to-end BCI orchestration and cross-validation.
- `backend/src/server.py` — WebSocket server for the live dashboard.
- `backend/scripts/validate_pipeline.py` — simulated validation entry point.

---

## Ethics and scope

This is a **research / course prototype**, not a medical device. Any future use with human participants would require appropriate **ethics approval**, **informed consent**, and **clinical oversight**.

