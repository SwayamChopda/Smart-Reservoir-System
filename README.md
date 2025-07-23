# 💧 Smart Reservoir Management System: AI-Powered Water Resource Optimization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL_HERE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.5.0-D00000?style=flat&logo=keras)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT) 

---

## 💡 Overview

This project develops an end-to-end AI-driven system to optimize water release schedules from dams, addressing critical issues of water wastage and inefficient resource management. By leveraging advanced machine learning for forecasting and a genetic algorithm for intelligent decision-making, it aims to enhance water conservation and climate adaptation for dam operations.

---

## 🎯 Problem Statement

Traditional dam operations often rely on fixed water release schedules, which fail to account for real-time demand fluctuations, unpredictable weather patterns, and actual reservoir inflow. This leads to:
-   **Significant Water Wastage:** Up to 30% of reservoir capacity can be lost due to inefficient management or uncontrolled overflows.
-   **Unmet Demand:** Risk of water scarcity during dry spells or peak consumption periods.
-   **Lack of Climate Adaptation:** Inability to respond dynamically to increasingly erratic climate patterns (e.g., flash floods, prolonged droughts).

---

## 🚀 Solution Architecture

The Smart Reservoir Management System is a robust decision-support tool comprising three core components:

1.  **Data Ingestion & Preprocessing (Backend):**
    * Simulated 20 years of daily hydrological and meteorological data (inflow/outflow/level, rainfall, temperature, humidity, water demand).
    * Features engineering (e.g., Day of Year, Month) and chronological splitting for time series analysis.

2.  **Forecasting Module (Backend - Machine Learning):**
    * **LSTM Neural Network:** Built with Keras (TensorFlow backend) to accurately predict future reservoir levels (7-day forecast horizon) based on a 30-day look-back window.
    * **Innovation:** Uses **Dropout layers** for regularization, **EarlyStopping** for efficient training, and **ModelCheckpointing** to save optimal model weights.

3.  **Optimization Module (Backend - Genetic Algorithm):**
    * **Custom Genetic Algorithm (GA):** Computes the optimal daily water release schedule.
    * **Fitness Function:** Simulates reservoir behavior based on LSTM forecasts, penalizing unmet demand, overflow, and deviations from target reservoir levels.
    * **Innovation:** Implemented **Elitism** for robust optimization and a **reward system for maintaining optimal reservoir capacity** at the end of the forecast horizon.

4.  **Interactive Dashboard (Frontend - Streamlit Web Application):**
    * **User Interface:** Provides dam operators with a user-friendly interface to visualize forecasts, recommendations, and perform scenario analysis.
    * **Innovation:**
        * **Scenario Analysis:** Interactive sliders allow users to adjust forecasted inflow/demand percentages, triggering real-time re-optimization by the GA for "what-if" planning.
        * **Comparative Visualization:** Clearly displays the GA's optimized reservoir level trajectory against the LSTM's unoptimized base forecast, highlighting the value of the optimization.

---

## ✨ Key Features & Innovations

* **AI-Powered Forecasting:** Accurate multi-step ahead reservoir level predictions using LSTMs.
* **Intelligent Optimization:** Genetic Algorithm for balancing water demand, supply, and conservation targets.
* **Dynamic Scenario Analysis:** Empowering operators to simulate hypothetical future conditions and get instant optimal strategies.
* **Intuitive Dashboard:** Streamlit application for accessible data visualization and decision support.
* **Robust Training:** Utilized advanced callbacks (EarlyStopping, ModelCheckpoint) and Dropout for model stability.

---

## 📊 Achievements & Results

* **High Forecasting Accuracy:** Achieved an **R-squared of 0.9968** and an **RMSE of 75.36 ML** (less than 1% of capacity) on synthetic test data, demonstrating exceptional predictive capability.
* **Optimal Resource Allocation:** The Genetic Algorithm consistently identified optimal water release schedules that eliminated penalties for unmet demand and overflow in simulated scenarios.
* **End-to-End Deployment:** Successfully developed and deployed a fully functional, interactive web application from data processing to AI models and frontend.

---

## 🛠️ Technical Stack

* **Language:** Python
* **Machine Learning:** TensorFlow (Keras), Scikit-learn
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Web Framework:** Streamlit
* **Version Control:** Git, GitHub
* **Deployment:** Streamlit Community Cloud

---

## 🚀 How to Run Locally

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/SwayamChopda/Smart-Reservoir-System.git](https://github.com/SwayamChopda/Smart-Reservoir-System.git)
    cd Smart-Reservoir-System
    ```
2.  **Create & Activate Conda Environment:**
    ```bash
    conda create -n reservoir_env python=3.11  # Or 3.10, 3.9
    conda activate reservoir_env
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run Jupyter Notebook for Model Training & Data Prep:**
    * Open `Smart_Reservoir_System.ipynb`.
    * Run all cells sequentially to generate `preprocessed_reservoir_data.csv`, `dashboard_data/` files, and `best_reservoir_lstm_model.keras`.
    * **Important:** Ensure your local `tensorflow` version matches the one in `requirements.txt` (currently `2.19.0` for Python 3.11). If not, you might need to install `tensorflow==2.19.0` after activating the conda env and before running the notebook.
5.  **Launch the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    Your app will open in your web browser at `http://localhost:8501`.

---

## 🌐 Deployment

The application is deployed and live on Streamlit Community Cloud.

**Visit the Live App Here:** https://smart-reservoir-system-nkhhsfzptkf2x6icphpf9p.streamlit.app/

---

## 📈 Impact & Future Enhancements

* **Water Conservation:** Direct potential for conserving millions of liters of water by reducing wasteful releases.
* **Climate Adaptation:** Enables proactive planning for extreme weather events.
* **Sustainable Infrastructure:** Promotes data-driven management of vital water resources.

**Future Enhancements:**
* Integration with **real-world historical hydrological and meteorological data**.
* Implementing **dynamic LSTM forecasts for inflow and demand** within the GA's fitness function for true closed-loop optimization.
* Integrating **Explainable AI (XAI)** techniques (e.g., SHAP values) to provide transparent insights into LSTM predictions.
* Exploring advanced UI features for **interactive simulation** where operators can manually adjust future releases and instantly see impacts.

---

## 🤝 Contributing

Feel free to fork this repository, open issues, or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---