# InfirmIA

A developer stress level prediction application that uses machine learning to help developers monitor and manage their stress based on work and lifestyle factors.

## Overview

InfirmIA combines a Linear Regression model trained on developer stress data with a FastAPI backend and a web frontend. Developers input metrics like hours worked, sleep, bugs resolved, and more — and receive a stress score (0–100) with actionable advice.

## Project Structure

```
InfirmIA/
├── Dockerfile
├── requirements.txt
├── developer_stress.csv                     # Training dataset (500 samples)
├── developer-stress-linear-regression.ipynb # EDA & model training notebook
├── src/
│   └── main.py                              # FastAPI backend
├── web/
│   ├── index.html                           # Frontend interface
│   └── assets/
│       └── jarvis-more.gif
└── models/
    ├── developer_stress_model.pkl           # Trained model
    └── developer_stress_scaler.pkl          # Feature scaler
```

## Tech Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn
- **ML**: scikit-learn (Linear Regression), pandas, numpy, joblib
- **Frontend**: HTML5, CSS3, vanilla JavaScript
- **Data Science**: Jupyter Notebook, matplotlib, seaborn
- **Deployment**: Docker

## Getting Started

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
git clone <repo-url>
cd InfirmIA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. API docs are served at `/docs`.

### Run with Docker

```bash
docker build -t infirmia .
docker run -p 8000:8000 infirmia
```

### Open the Frontend

Open `web/index.html` in a browser. Make sure the API is running on port 8000.

## API

### `GET /health`

Health check endpoint.

### `POST /predict`

Predict developer stress level.

**Request body:**

| Field             | Type   | Description                          |
|-------------------|--------|--------------------------------------|
| hours_worked      | float  | Hours worked per day                 |
| sleep_hours       | float  | Hours of sleep                       |
| bugs              | float  | Number of bugs encountered           |
| deadline_days     | float  | Days until deadline                  |
| coffee_cups       | float  | Cups of coffee consumed              |
| meetings          | float  | Number of meetings                   |
| interruptions     | float  | Number of interruptions              |
| experience_level  | string | `Junior`, `Mid`, or `Senior`         |
| code_complexity   | string | `Low`, `Medium`, or `High`           |
| remote_work       | string | `Yes` or `No`                        |

**Response:**

```json
{
  "stress_level": 52.3
}
```

## Stress Levels

| Score    | Level    | Message                           |
|----------|----------|-----------------------------------|
| < 40     | Low      | You're doing fine!                |
| 40 – 65  | Moderate | You should take a break soon.     |
| ≥ 65     | High     | You need to rest now.             |

## Model

The model is a Linear Regression trained on 500 developer stress records with an R² score of ~55.8% on the test set. Features include both numeric (hours worked, sleep, bugs, deadlines, coffee, meetings, interruptions) and categorical (experience level, code complexity, remote work) variables. The Jupyter notebook contains the full EDA and training pipeline.
