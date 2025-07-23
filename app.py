# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import copy
import os # For checking file existence

# --- Configuration Constants (matching your notebook) ---
RESERVOIR_MAX_CAPACITY_ML = 10000
MIN_RESERVOIR_LEVEL_ML = 1000
LOOK_BACK = 30
FORECAST_HORIZON = 7 # 7 days ahead forecast
GA_POP_SIZE = 100
GA_NUM_GENERATIONS = 100
GA_MUTATION_RATE = 0.1
GA_MIN_RELEASE = 0
GA_MAX_RELEASE = 500 # Adjust based on your dam's typical max release capacity


# --- Load Model and Scaler (Run once when app starts) ---
@st.cache_resource # Caches the resource (model, scaler) so it's loaded only once
def load_resources():
    model_path = 'best_reservoir_lstm_model.keras'
    if not os.path.exists(model_path):
        st.error(f"Error: LSTM model file '{model_path}' not found. Please ensure it's in the same directory as app.py.")
        st.stop() # Stop the app if model can't be loaded

    try:
        best_model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}. Check if TensorFlow/Keras are correctly installed and model file is valid.")
        st.stop()

    df_original_path = 'preprocessed_reservoir_data.csv'
    if not os.path.exists(df_original_path):
        st.error(f"Error: Preprocessed data file '{df_original_path}' not found. Please ensure it's in the same directory as app.py.")
        st.stop()

    df_original = pd.read_csv(df_original_path, index_col='Date', parse_dates=True)

    target_column = 'Reservoir_Level_ML'
    features = [col for col in df_original.columns if col != target_column and col != 'Year']
    data_for_model = df_original[features + [target_column]]

    # Fit scaler on the data segment that corresponds to training data used for model training
    # This is critical for consistent scaling with model training.
    # We need to replicate the train_data_raw from the notebook for fitting the scaler.
    train_size_for_scaler = int(len(data_for_model) * 0.8)
    train_data_for_scaler_fit = data_for_model.iloc[:train_size_for_scaler]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data_for_scaler_fit) # Fit scaler only on training data portion

    # Calculate target_col_idx_in_raw_data based on data_for_model columns
    target_col_idx_in_raw_data = data_for_model.columns.get_loc(target_column)

    return best_model, scaler, data_for_model, df_original, target_column, features, target_col_idx_in_raw_data

# Load resources globally to avoid re-loading on every rerun
best_model, scaler, data_for_model, df_original, target_column, features, target_col_idx_in_raw_data = load_resources()

# --- Re-use your `create_sequences` function (from Step 2, Cell 4) ---
# This function is used to prepare input for the LSTM model
def create_sequences_for_prediction(data_array_scaled, look_back_days):
    # This function is slightly simplified for single prediction scenario
    # It assumes data_array_scaled is already the correct `look_back_days` slice
    X = np.expand_dims(data_array_scaled[-look_back_days:, :], axis=0) # Get last `look_back_days` and add batch dimension
    return X

# --- Re-use GA components (Individual, initialize_population, selection, crossover, mutate) ---
# From Step 3, Cell 2
class Individual:
    def __init__(self, release_schedule):
        self.release_schedule = np.array(release_schedule, dtype=float)
        self.fitness = 0.0

def initialize_population(pop_size, forecast_horizon, min_release, max_release):
    population = []
    for _ in range(pop_size):
        schedule = np.random.uniform(min_release, max_release, forecast_horizon)
        population.append(Individual(schedule))
    return population

def selection(population, num_parents):
    parents = []
    for _ in range(num_parents):
        tournament_size = 5
        tournament_candidates = random.sample(population, tournament_size)
        fittest_candidate = max(tournament_candidates, key=lambda ind: ind.fitness)
        parents.append(fittest_candidate)
    return parents

def crossover(parent1, parent2):
    if len(parent1.release_schedule) < 2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    crossover_point = random.randint(1, len(parent1.release_schedule) - 1)
    child1_schedule = np.concatenate((parent1.release_schedule[:crossover_point],
                                      parent2.release_schedule[crossover_point:]))
    child2_schedule = np.concatenate((parent2.release_schedule[:crossover_point],
                                      parent1.release_schedule[crossover_point:]))
    return Individual(child1_schedule), Individual(child2_schedule)

def mutate(individual, mutation_rate, min_release, max_release, mutation_strength=0.1):
    for i in range(len(individual.release_schedule)):
        if random.random() < mutation_rate:
            change = np.random.normal(0, mutation_strength * (max_release - min_release))
            individual.release_schedule[i] = np.clip(individual.release_schedule[i] + change, min_release, max_release)

# --- IMPORTANT: Refined `calculate_fitness` for GA - this is the core of the "smartness" ---
# This function is designed to be called within the GA.
# It simulates reservoir levels using forecasted_inflow/demand and GA's proposed release.
def calculate_fitness(individual, current_reservoir_level_ml,
                      forecasted_inflow_override, forecasted_demand_override,
                      max_capacity, min_level, forecast_horizon_days):
    """
    Calculates the fitness of a release schedule given specific forecasts.
    The GA is optimizing the release schedule against these forecasts.
    """
    simulated_reservoir_levels = [current_reservoir_level_ml]
    total_unmet_demand = 0
    total_overflow = 0

    current_level = current_reservoir_level_ml

    for day in range(forecast_horizon_days):
        daily_inflow = forecasted_inflow_override[day]
        daily_demand = forecasted_demand_override[day]
        daily_release = individual.release_schedule[day]

        # Ensure release is within sensible bounds (cannot release more than available, cannot be negative)
        daily_release = np.clip(daily_release, 0, current_level + daily_inflow)

        # Calculate next reservoir level
        next_level = current_level + daily_inflow - daily_release

        # Handle overflow
        if next_level > max_capacity:
            total_overflow += (next_level - max_capacity)
            next_level = max_capacity # Cap at max capacity

        # Handle unmet demand
        if daily_release < daily_demand:
            total_unmet_demand += (daily_demand - daily_release)

        # Penalize if level drops below minimum safe level
        if next_level < min_level:
            total_unmet_demand += (min_level - next_level) * 2 # Heavier penalty for critical low levels

        current_level = next_level
        simulated_reservoir_levels.append(current_level)

    # Fitness calculation: Maximize water conservation and meeting demand, minimize overflow.
    final_level = simulated_reservoir_levels[-1]
    optimal_level_target = max_capacity * 0.7 # Aim for 70% capacity at end of horizon
    level_deviation_penalty = abs(final_level - optimal_level_target) * 0.1 # Small penalty for deviation

    total_penalty = total_unmet_demand * 10 + total_overflow * 5 + level_deviation_penalty

    fitness = 1_000_000 - total_penalty
    if fitness < 0: fitness = 1 # Ensure fitness is at least 1
    individual.fitness = fitness
    return fitness


@st.cache_data(show_spinner=False) # Cache results of GA run
def run_genetic_algorithm_st(current_level, forecasted_inflow_ga, forecasted_demand_ga,
                             pop_size, num_generations, mutation_rate, min_release, max_release):
    """
    Streamlit-compatible GA runner.
    """
    population = initialize_population(pop_size, FORECAST_HORIZON, min_release, max_release)
    best_overall_individual = None
    best_overall_fitness = -np.inf

    for generation in range(num_generations):
        for individual in population:
            calculate_fitness(individual, current_level, forecasted_inflow_ga, forecasted_demand_ga,
                              RESERVOIR_MAX_CAPACITY_ML, MIN_RESERVOIR_LEVEL_ML, FORECAST_HORIZON)

        current_best_individual = max(population, key=lambda ind: ind.fitness)
        if current_best_individual.fitness > best_overall_fitness:
            best_overall_fitness = current_best_individual.fitness
            best_overall_individual = copy.deepcopy(current_best_individual)

        num_parents = pop_size // 2
        parents = selection(population, num_parents)
        next_population = []
        next_population.append(copy.deepcopy(best_overall_individual)) # Elitism

        while len(next_population) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child1, child2 = crossover(p1, p2)
            mutate(child1, mutation_rate, min_release, max_release)
            mutate(child2, mutation_rate, min_release, max_release)
            next_population.extend([child1, child2])

        population = next_population[:pop_size]

    return best_overall_individual


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Smart Reservoir System")

st.title("💧 Smart Reservoir Management System")
st.markdown("""
    This application leverages **Machine Learning (LSTM)** for forecasting and a **Genetic Algorithm** for optimizing water release schedules.
    It helps dam operators make informed decisions to **conserve water**, **meet demand**, and **prevent overflow**.
""")

# --- Sidebar for Controls and Inputs ---
st.sidebar.header("System Controls")

# Select date for simulation start
max_date_selectable = df_original.index.max().date() - pd.Timedelta(days=LOOK_BACK + FORECAST_HORIZON + 1)
min_date_selectable = df_original.index.min().date() + pd.Timedelta(days=LOOK_BACK + FORECAST_HORIZON + 1) # Ensure enough history + future for GA

simulation_start_date_input = st.sidebar.date_input(
    "Select Simulation Start Date:",
    value=max_date_selectable, # Default to a recent date that allows full forecast
    min_value=min_date_selectable,
    max_value=max_date_selectable
)

# Convert to datetime object and check for data availability
simulation_start_datetime = pd.to_datetime(simulation_start_date_input)

try:
    current_reservoir_level = df_original.loc[simulation_start_datetime, target_column]
    # Extract the required historical data segment for the LSTM input
    idx_for_sim_start = df_original.index.get_loc(simulation_start_datetime)
    past_data_segment_raw = df_original.iloc[idx_for_sim_start - LOOK_BACK : idx_for_sim_start][features + [target_column]]
    
    # Check if past_data_segment_raw has enough rows
    if len(past_data_segment_raw) != LOOK_BACK:
        st.error(f"Error: Not enough historical data for a {LOOK_BACK}-day look-back before {simulation_start_date_input}. Please select an earlier date in the historical data.")
        st.stop()

    past_data_segment_scaled = scaler.transform(past_data_segment_raw)
    
except KeyError:
    st.error(f"Error: Data for {simulation_start_date_input} not found. Please choose a date within the dataset's range.")
    st.stop()


st.sidebar.markdown("---")
st.sidebar.subheader("Scenario Analysis (Forecast Adjustments)")
st.sidebar.info("Adjust the predicted Inflow and Demand for the next 7 days to simulate different future scenarios.")

inflow_adjust = st.sidebar.slider("Adjust Forecasted Inflow (%)", -50, 50, 0, help="Adjust the historical average inflow for the forecast horizon by this percentage.")
demand_adjust = st.sidebar.slider("Adjust Forecasted Demand (%)", -30, 30, 0, help="Adjust the historical average demand for the forecast horizon by this percentage.")


# --- LSTM Base Forecast (This section generates the base future Inflow/Demand) ---
# In a real system, you'd have LSTMs trained for Inflow and Demand too.
# For this synthetic data demo, we use historical averages for the forecast horizon.
# For simplicity, we get the next 7 days' actual historical Inflow/Demand as our "base forecast"
# starting *from* `simulation_start_datetime` for LSTM's prediction horizon.
# However, the GA needs future forecasts *starting from tomorrow*.

# Let's use the LSTM model to predict the *next* `FORECAST_HORIZON` days of reservoir levels.
# This serves as a "base" forecast if no optimization strategy is applied.
# This assumes the LSTM can predict levels given past data.
# Then, for GA, we'll need independent Inflow/Demand forecasts.

# For demo purposes, we will take the actual future inflow and demand from `df_original`
# starting from the day *after* `simulation_start_datetime` as our baseline "forecasts"
# for the GA and scenario analysis. In a real system, LSTMs would predict these.
try:
    base_forecasted_inflow = df_original['Inflow_ML'].loc[simulation_start_datetime + pd.Timedelta(days=1) : simulation_start_datetime + pd.Timedelta(days=FORECAST_HORIZON)].values
    base_forecasted_demand = df_original['Demand_ML'].loc[simulation_start_datetime + pd.Timedelta(days=1) : simulation_start_datetime + pd.Timedelta(days=FORECAST_HORIZON)].values
    
    if len(base_forecasted_inflow) < FORECAST_HORIZON or len(base_forecasted_demand) < FORECAST_HORIZON:
         st.error(f"Error: Not enough future data for a {FORECAST_HORIZON}-day forecast from {simulation_start_date_input}. Please select an earlier simulation start date.")
         st.stop()

except KeyError:
    st.error(f"Error: Not enough future data for a {FORECAST_HORIZON}-day forecast from {simulation_start_date_input}. Please choose an earlier simulation start date within the dataset.")
    st.stop()


# Apply adjustments for scenario analysis
adjusted_forecasted_inflow = base_forecasted_inflow * (1 + inflow_adjust / 100)
adjusted_forecasted_demand = base_forecasted_demand * (1 + demand_adjust / 100)


# --- Main Content Area ---
st.header("Current Status & Optimization")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Current Reservoir Level", value=f"{current_reservoir_level:.2f} ML", delta=None)
    # Add other current metrics if desired from df_original.loc[simulation_start_datetime]

with col2:
    st.metric(label="Max Reservoir Capacity", value=f"{RESERVOIR_MAX_CAPACITY_ML} ML", delta=None)
    st.metric(label="Min Safe Level", value=f"{MIN_RESERVOIR_LEVEL_ML} ML", delta=None)

# Button to trigger optimization
if st.sidebar.button("Run Optimization for Scenario"):
    st.subheader("Optimization Results")

    with st.spinner(f'Running Genetic Algorithm for optimal release (Generations: {GA_NUM_GENERATIONS})...'):
        optimal_schedule_individual = run_genetic_algorithm_st(
            current_level=current_reservoir_level,
            forecasted_inflow_ga=adjusted_forecasted_inflow,
            forecasted_demand_ga=adjusted_forecasted_demand,
            pop_size=GA_POP_SIZE, num_generations=GA_NUM_GENERATIONS,
            mutation_rate=GA_MUTATION_RATE, min_release=GA_MIN_RELEASE, max_release=GA_MAX_RELEASE
        )

    st.success("Optimization Complete!")

    # --- Display Optimal Release Schedule ---
    st.subheader("Optimal Water Release Schedule (ML/day)")
    optimal_releases_df = pd.DataFrame({
        'Day': [f'Day {i+1}' for i in range(FORECAST_HORIZON)],
        'Date': pd.date_range(start=simulation_start_datetime + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq='D').strftime('%Y-%m-%d'),
        'Optimal Release (ML/day)': optimal_schedule_individual.release_schedule.round(2)
    })
    st.dataframe(optimal_releases_df, hide_index=True, use_container_width=True)

    # --- Simulate and Display Future Reservoir Levels based on Optimal Release ---
    st.subheader("Simulated Future Reservoir Levels (Optimal Strategy)")

    ga_simulated_levels = [current_reservoir_level] # Start with current day's level
    forecast_dates_for_plot = [simulation_start_datetime] # Include current day for plot

    for day in range(FORECAST_HORIZON):
        daily_inflow_sim = adjusted_forecasted_inflow[day]
        daily_optimal_release_sim = optimal_schedule_individual.release_schedule[day]

        daily_optimal_release_sim = np.clip(daily_optimal_release_sim, 0, ga_simulated_levels[-1] + daily_inflow_sim)
        next_level_sim = ga_simulated_levels[-1] + daily_inflow_sim - daily_optimal_release_sim
        next_level_sim = np.clip(next_level_sim, 0, RESERVOIR_MAX_CAPACITY_ML)
        ga_simulated_levels.append(next_level_sim)
        forecast_dates_for_plot.append(simulation_start_datetime + pd.Timedelta(days=day+1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forecast_dates_for_plot, ga_simulated_levels, marker='o', linestyle='-', color='green', label='Simulated Reservoir Level (Optimal)')

    # --- LSTM Base Forecast Plot ---
    # This is line 358 in my code, ensure its indentation matches the above 'ax.plot' etc.
    lstm_predicted_levels_scaled = best_model.predict(past_data_segment_scaled.reshape(1, LOOK_BACK, len(features) + 1))[0] # Get the 1D array of predictions

    dummy_pred_array_single = np.zeros((FORECAST_HORIZON, data_for_model.shape[1]))
    dummy_pred_array_single[:, target_col_idx_in_raw_data] = lstm_predicted_levels_scaled
    lstm_predicted_levels_unscaled = scaler.inverse_transform(dummy_pred_array_single)[:, target_col_idx_in_raw_data]

    # This is the line that caused the error for you. Make sure its indentation matches the lines above and below it.
    lstm_forecast_dates = pd.date_range(start=simulation_start_datetime + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq='D')
    ax.plot(lstm_forecast_dates, lstm_predicted_levels_unscaled, marker='x', linestyle='--', color='blue', label='LSTM Base Forecast (No Optimization)')

    ax.axhline(RESERVOIR_MAX_CAPACITY_ML, color='red', linestyle=':', label='Max Capacity')
    ax.axhline(MIN_RESERVOIR_LEVEL_ML, color='orange', linestyle=':', label='Min Safe Level')

    ax.set_title("Future Reservoir Level Projections", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Reservoir Level (ML)", fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig) # Display plot in Streamlit

    st.markdown("---")

    # --- Display Forecasted Inflow & Demand ---
    st.subheader("Forecasted Conditions for Optimization")
    forecast_df = pd.DataFrame({
        'Day': [f'Day {i+1}' for i in range(FORECAST_HORIZON)],
        'Date': pd.date_range(start=simulation_start_datetime + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq='D').strftime('%Y-%m-%d'),
        'Forecasted Inflow (ML/day)': adjusted_forecasted_inflow.round(2),
        'Forecasted Demand (ML/day)': adjusted_forecasted_demand.round(2)
    })
    st.dataframe(forecast_df, hide_index=True, use_container_width=True)


# --- Separator for next sections ---
st.markdown("---")

# --- Section for Model Performance & Historical Context ---
st.header("Model Performance & Historical Context")

st.subheader("LSTM Model Test Set Performance")
st.write("These metrics indicate the overall accuracy of the LSTM's predictions on unseen historical data.")

col_met1, col_met2, col_met3 = st.columns(3)
with col_met1:
    st.metric(label="RMSE", value=f"{75.36:.2f} ML", help="Root Mean Squared Error: Average prediction error in Million Liters.")
with col_met2:
    st.metric(label="MAE", value=f"{50.43:.2f} ML", help="Mean Absolute Error: Average absolute difference between actual and predicted values.")
with col_met3:
    st.metric(label="R-squared (R2)", value=f"{0.9968:.4f}", help="Proportion of variance in actual reservoir level predictable by the model (closer to 1 is better).")


st.subheader("Historical Reservoir Level & LSTM Forecast Comparison")
st.write("This plot shows how well the LSTM model predicted historical reservoir levels during the test period.")

# Load the historical test forecasts you saved
try:
    lstm_test_forecasts = pd.read_csv('./dashboard_data/lstm_test_forecasts.csv', index_col='Date', parse_dates=True)
except FileNotFoundError:
    st.error("LSTM test forecasts CSV not found. Please ensure 'lstm_test_forecasts.csv' is in the 'dashboard_data' folder.")
    st.stop()


fig_hist, ax_hist = plt.subplots(figsize=(15, 7))
ax_hist.plot(lstm_test_forecasts.index, lstm_test_forecasts['Actual_Reservoir_Level'], label='Actual Reservoir Level (Test)', color='blue')
ax_hist.plot(lstm_test_forecasts.index, lstm_test_forecasts['Predicted_Reservoir_Level'], label='Predicted Reservoir Level (Test)', color='red', linestyle='--')
ax_hist.set_title('Historical Actual vs. Predicted Reservoir Level (Test Set)', fontsize=16)
ax_hist.set_xlabel('Date', fontsize=14)
ax_hist.set_ylabel('Reservoir Level (ML)', fontsize=14)
ax_hist.legend()
ax_hist.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig_hist)

st.subheader("Full Historical Data Overview")
st.write("Long-term trends of reservoir level, inflow, and outflow.")

# Load the full historical data you saved
try:
    historical_data_full_df = pd.read_csv('./dashboard_data/historical_reservoir_data.csv', index_col='Date', parse_dates=True)
except FileNotFoundError:
    st.error("Full historical data CSV not found. Please ensure 'historical_reservoir_data.csv' is in the 'dashboard_data' folder.")
    st.stop()

fig_full_hist_detail, ax_full_hist_detail = plt.subplots(figsize=(15, 7))
ax_full_hist_detail.plot(historical_data_full_df.index, historical_data_full_df['Reservoir_Level_ML'], label='Reservoir Level', color='purple')
ax_full_hist_detail.plot(historical_data_full_df.index, historical_data_full_df['Inflow_ML'], label='Inflow', color='green', alpha=0.7)
ax_full_hist_detail.plot(historical_data_full_df.index, historical_data_full_df['Outflow_ML'], label='Outflow', color='brown', alpha=0.7)
ax_full_hist_detail.axhline(RESERVOIR_MAX_CAPACITY_ML, color='red', linestyle=':', label='Max Capacity')
ax_full_hist_detail.axhline(MIN_RESERVOIR_LEVEL_ML, color='orange', linestyle=':', label='Min Safe Level')
ax_full_hist_detail.set_title('Long-term Historical Reservoir Data', fontsize=16)
ax_full_hist_detail.set_xlabel('Date', fontsize=14)
ax_full_hist_detail.set_ylabel('Volume (ML)', fontsize=14)
ax_full_hist_detail.legend()
ax_full_hist_detail.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig_full_hist_detail)


# --- Innovative Feature: Explainable AI (Conceptual Section) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Explainability (XAI) Insights")
st.sidebar.info("This section would provide insights into why the LSTM makes certain predictions (e.g., highlighting key influencing factors like rainfall or demand).")
st.sidebar.write("*(Actual XAI implementation requires advanced libraries like SHAP/LIME and more complex integration.)*")

st.markdown("---")
st.markdown("Developed as part of a Smart Reservoir Management System project. Designed to aid water resource optimization and climate adaptation planning.")