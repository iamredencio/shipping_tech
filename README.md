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
