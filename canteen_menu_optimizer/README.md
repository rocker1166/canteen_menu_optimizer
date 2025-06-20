# Canteen Menu Optimizer

This project implements a Canteen Menu Optimizer using a hybrid approach of Machine Learning (XGBoost) for demand prediction, Reinforcement Learning (Q-learning) for policy optimization, and rule-based overrides for specific scenarios.

## Project Structure

```
canteen_menu_optimizer/
├── data/
│   ├── academic_calendar.csv
│   ├── historical_sales.csv
│   ├── weather_data.csv
│   ├── X_preprocessed.csv
│   └── y_target.csv
├── models/
│   ├── xgboost_model.pkl
│   ├── rl_q_table.pkl
│   ├── scaler.pkl
│   └── le_item_id.pkl
├── src/
│   ├── api_backend.py
│   ├── canteen_env.py
│   ├── data_preprocessing.py
│   ├── decision_engine.py
│   ├── generate_synthetic_data.py
│   ├── rl_agent.py
│   └── train_ml_model.py
└── README.md
```

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # If this were a real repository, you would clone it here.
    # For this project, the files are already in the `canteen_menu_optimizer` directory.
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd canteen_menu_optimizer
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost fastapi uvicorn python-multipart
    ```

5.  **Generate Synthetic Data:**
    Run the script to generate the necessary CSV data files:
    ```bash
    python3 src/generate_synthetic_data.py
    ```
    This will create `historical_sales.csv`, `weather_data.csv`, and `academic_calendar.csv` in the `data/` directory.

6.  **Preprocess Data:**
    Run the data preprocessing script. This will create `X_preprocessed.csv` and `y_target.csv` and save the `scaler.pkl` and `le_item_id.pkl` files in the `models/` directory.
    ```bash
    python3 src/data_preprocessing.py
    ```

7.  **Train ML Model:**
    Train the XGBoost model:
    ```bash
    python3 src/train_ml_model.py
    ```
    This will save `xgboost_model.pkl` in the `models/` directory.

8.  **Train RL Agent:**
    Train the Q-learning RL agent:
    ```bash
    python3 src/rl_agent.py
    ```
    This will save `rl_q_table.pkl` in the `models/` directory.

## Running the API Backend

To start the FastAPI server, run the following command from the `canteen_menu_optimizer` directory:

```bash
uvicorn src.api_backend:app --host 0.0.0.0 --port 8000
```

The API will be accessible at `http://0.0.0.0:8000` (or the exposed public URL if running in a sandboxed environment).

## API Endpoint

### `POST /predict`

**Description:** Predicts the optimal quantity of a food item to prepare for a given date.

**Request Body (JSON):**

```json
{
  "date": "YYYY-MM-DD",
  "item_id": "item_X",
  "current_stock": 100,      // Optional
  "rainfall_today": 15.5     // Optional
}
```

**Example Request (using `curl`):**

```bash
curl -X POST "http://0.0.0.0:8000/predict" \
     -H "Content-Type: application/json" \
     -d 
```

**Example Response (JSON):**

```json
{
  "item_id": "item_1",
  "predicted_quantity": 120
}
```

## Decision Engine Logic

The `decision_engine.py` script combines the outputs of the ML model and RL agent with rule-based overrides:

1.  **ML Prediction:** XGBoost predicts the base demand for the item.
2.  **RL Adjustment:** The Q-learning agent provides an adjustment based on learned policies from the simulation environment.
3.  **Rule-Based Overrides:**
    *   If `current_stock` is 0, `final_quantity` is set to 0.
    *   If `rainfall_today` is greater than 20mm and the item is 


"Maggi" (simplified check), the `final_quantity` is increased by 10%.

## Simulation Environment (`canteen_env.py`)

This module defines the environment for training the RL agent. It simulates daily canteen operations, including demand, costs, revenue, waste, and underproduction penalties.

## Continuous Learning Loop (Conceptual)

In a production environment, the models would be continuously retrained with new data:

*   **Daily:** Collect actual sales, waste, and event notes.
*   **Weekly:** Auto-retrain supervised ML models.
*   **Periodically:** Fine-tune the RL agent in the simulator.
*   **Human Feedback:** Incorporate staff feedback to improve the reward function.

## Dashboard UI (Conceptual)

A dashboard would provide:

*   Today's recommended quantities.
*   Yesterday's performance (waste, sold, understocked).
*   Weather and crowd forecasts.
*   Manual override options.
*   Inventory sync and waste tracking trends.

## Deployment

This project can be deployed using:

*   **Model Hosting:** Hugging Face, Colab Pro, or Render.
*   **Backend APIs:** FastAPI (used here).
*   **Database:** Supabase (PostgreSQL) or SQLite for MVP.
*   **Frontend:** Next.js + Tailwind (not implemented in this project).
*   **CI/CD:** GitHub + Vercel auto-deploy.


