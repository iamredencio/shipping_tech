# Maritime AI: Vessel Tracking & Prediction Dashboard

## Overview

This project provides a platform for maritime vessel tracking, analysis, and trajectory prediction using Artificial Intelligence and Machine Learning techniques. It features a Python backend for data processing and model execution (including Kalman Filters and LSTMs) and a Next.js frontend for real-time (simulated) visualization and interaction.

The primary goal is to process Automatic Identification System (AIS) data to understand vessel behavior, predict future positions, and potentially identify anomalies or optimize routes.

## Features

*   **AIS Data Processing:** Handles loading and cleaning of vessel AIS data.
*   **Advanced Feature Engineering:** Creates relevant features from raw AIS data, including temporal, kinematic, and vessel characteristic features (`maritime-tracking/app/ml/feature_engineering.py`).
*   **Kalman Filter Tracking:** Implements an Enhanced Kalman Filter for robust state estimation and smoothing of vessel tracks (`maritime-tracking/app/ml/algorithms.py`).
*   **Deep Learning Prediction:** Utilizes an LSTM-based model to predict future vessel trajectories based on historical sequences and various features (`maritime-tracking/app/ml/deep_learning.py`).
*   **Interactive Dashboard:** A web-based dashboard built with Next.js and Recharts to visualize (currently simulated) real-time vessel data, model performance, anomalies, and steering recommendations (`dashboard.tsx`).
*   **Modular Design:** Separates concerns between data processing/ML (Python backend) and visualization (Next.js frontend).

## Tech Stack

*   **Frontend:**
    *   Next.js
    *   React
    *   TypeScript
    *   Tailwind CSS
    *   Shadcn UI
    *   Recharts (for charts)
    *   Lucide React (for icons)
*   **Backend (Machine Learning):**
    *   Python 3.x
    *   Pandas
    *   NumPy
    *   Scikit-learn
    *   TensorFlow / Keras
    *   SciPy
*   **Environment Management:**
    *   `venv` (Python)
    *   `npm` or `yarn` (Node.js)

## Project Structure

## Installation

There are two ways to set up the project: using the automated setup script (recommended for Linux/macOS) or manually following the steps.

### Option 1: Using the Setup Script (Linux/macOS)

This script automates the setup of both the Python backend and the Node.js frontend environments.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/iamredencio/shipping_tech.git
    cd shipping_tech
    ```

2.  **Make the script executable:**
    ```bash
    chmod +x setup.sh
    ```

3.  **Run the script:**
    ```bash
    ./setup.sh
    ```

    The script will check prerequisites, create a Python virtual environment, install Python and Node.js dependencies. Follow the instructions printed at the end of the script to run the application.

### Option 2: Manual Installation

Follow these steps if you prefer to set up manually or if you are on an operating system other than Linux/macOS.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/iamredencio/shipping_tech.git
    cd shipping_tech
    ```

2.  **Set up Python Backend:**
    *   Navigate to the backend directory:
        ```bash
        cd maritime-tracking
        ```
    *   Create and activate a virtual environment:
        ```bash
        python3 -m venv venv  # Use python instead of python3 if that's your command
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    *   Install Python dependencies:
        ```bash
        pip install --upgrade pip
        pip install -r requirements.txt
        ```
    *   Deactivate the environment for now (you'll activate it when running backend scripts):
        ```bash
        deactivate
        ```
    *   Return to the root directory:
        ```bash
        cd ..
        ```

3.  **Set up Next.js Frontend:**
    *   Install Node.js dependencies:
        ```bash
        npm install
        # or if you use yarn:
        # yarn install
        ```

4.  **Environment Variables:**
    *   If the application requires specific API keys or configurations, create a `.env.local` file in the root directory and add them there. Refer to `.env.example` if provided.

## Usage

1.  **Prepare Data:**
    *   Place your AIS data CSV files in the appropriate location (e.g., `maritime-tracking/data/raw/`).
    *   Update any data loading scripts or configurations in the Python backend if necessary to point to your data.

2.  **Run Backend Processes (Example - Adapt as needed):**
    *   Activate the Python virtual environment:
        ```bash
        cd maritime-tracking
        source venv/bin/activate # On Windows use `venv\Scripts\activate`
        ```
    *   Run necessary Python scripts (e.g., for training, data processing, or starting an API server - *these scripts might need to be created*):
        ```bash
        # Example: python train_model.py
        # Example: python run_api.py
        ```
    *   Deactivate when finished with backend tasks:
        ```bash
        deactivate
        cd ..
        ```

3.  **Run Frontend Dashboard:**
    *   Start the Next.js development server (from the project root directory):
        ```bash
        npm run dev
        # or
        # yarn dev
        ```
    *   Open your browser and navigate to `http://localhost:3000` (or the port specified in the console).

    *   **Note:** The dashboard currently uses **simulated data**. To display real results, connect the frontend to a backend API providing actual data and predictions.

## Running with Docker (Recommended for Deployment)

This project includes Docker configuration to build and run the frontend and backend services in containers using Docker Compose.

**Prerequisites:**

*   Docker: [Install Docker](https://docs.docker.com/get-docker/)
*   Docker Compose: Usually included with Docker Desktop, or [install separately](https://docs.docker.com/compose/install/).

**Steps:**

1.  **Build and Run Containers:**
    *   Open your terminal in the project's root directory.
    *   Run the following command:
        ```bash
        docker-compose up --build
        ```
        *   `--build`: Forces Docker to build the images before starting the containers. You can omit `--build` on subsequent runs if the code hasn't changed significantly.
        *   `-d`: (Optional) Add `-d` to run the containers in detached mode (in the background). `docker-compose up --build -d`

2.  **Access the Application:**
    *   **Frontend:** Open your browser and navigate to `http://localhost:3000`.
    *   **Backend API (if running):** The API should be accessible at `http://localhost:8000` (or potentially via the frontend depending on your setup).

3.  **Stopping Containers:**
    *   If running in the foreground (without `-d`), press `Ctrl + C` in the terminal where `docker-compose up` is running.
    *   If running in detached mode (with `-d`), run:
        ```bash
        docker-compose down
        ```
        This command stops and removes the containers and network defined in the `docker-compose.yml` file. Add `-v` (`docker-compose down -v`) if you also want to remove named volumes (not used in the current basic setup).

**Important Notes:**

*   The `CMD` in `Dockerfile.backend` needs to be correctly set to run your Python API server (e.g., Uvicorn for FastAPI). The provided file has a placeholder/example.
*   Ensure any necessary environment variables are defined in the `docker-compose.yml` file for both services.
*   Large data files are ignored by git and won't be included in the Docker image unless mounted via volumes or added through a separate data handling strategy.
