# 🌫️ Sequential Hybrid AIS + CSA — Air Pollution Forecasting System.

An intelligent deep-learning system that forecasts daily **air pollution levels (PM2.5)** using a **Sequential Hybrid Artificial Immune System (AIS)** + **Cuckoo Search Algorithm (CSA)** to optimize a **CNN + LSTM** model for maximum accuracy and stability.

---

## 🧠 Overview

Air quality prediction plays a vital role in environmental management and public health.  
This project combines the strengths of **AIS** (diverse global search) and **CSA** (refined local search) in a **sequential hybrid** structure:

1. **Stage 1 — AIS (Exploration):**  
   Generates a diverse population of hyper-parameters (filters, LSTM units, dropout, learning rate).  
   Mimics immune adaptation to explore the full search space.

2. **Stage 2 — CSA (Refinement):**  
   Starts from the best AIS solution and performs Lévy-flight-based parameter refinement.  
   Ensures convergence toward the global optimum.

The final CNN + LSTM model is trained on real-world multi-source air-pollution data and provides both **quantitative results** and **visual analytics**.

---

## 📁 Dataset Structure

Place all datasets inside:
C:\Users\NXTWAVE\Downloads\Air Pollution Forecasting\archive\



yaml
Copy code

Required files:
station_hour.csv
stations.csv
city_hour.csv
station_day.csv
city_day.csv

yaml
Copy code

> These files contain hourly and daily air-quality readings from multiple monitoring stations and cities.

---

## ⚙️ Preprocessing

* Missing values handled by forward/backward fill.  
* Non-numerical columns (IDs, names) are dropped.  
* Data normalized via `MinMaxScaler`.  
* Target variable — **PM2.5** (fine-particulate concentration).  
* Split: 80 % train  /  20 % test.  

---

## 🧩 Model Architecture

| Layer | Type | Parameters / Activation |
|--------|------|--------------------------|
| 1 | Conv1D | `filters` (optimized) × 3 kernel – ReLU |
| 2 | MaxPooling1D | Pool = 2 |
| 3 | LSTM | `units` (optimized) – tanh |
| 4 | Dense | 32 – ReLU |
| 5 | Dropout | `rate` (optimized) |
| 6 | Dense (Output) | 1 – Linear |

**Optimizer:** Adam  
**Loss:** MSE  
**Metrics:** MAE  

---

## 🔬 Sequential Hybrid Optimization

### Stage 1 — AIS (Artificial Immune System)
* Generates random hyper-parameter populations.  
* Applies immune mutation for diversity.  
* Selects individuals with lowest validation loss.

### Stage 2 — CSA (Cuckoo Search Algorithm)
* Uses Lévy-flight randomization to refine best AIS solution.  
* Updates parameters iteratively if new loss improves.  
* Produces globally refined best model configuration.

---

## 🧾 Outputs

All artifacts are saved in:
C:\Users\NXTWAVE\Downloads\Air Pollution Forecasting\

yaml
Copy code

| File | Description |
|------|--------------|
| `sequential_air_forecasting_model.h5` | Trained CNN + LSTM model |
| `sequential_air_forecasting_results.json` | Evaluation metrics (MAE, RMSE, R²) |
| `sequential_air_forecasting_config.yaml` | Final hyper-parameters + optimizer details |
| `sequential_air_forecasting_scaler.pkl` | Saved data normalizer |
| `sequential_air_forecasting_loss_graph.png` | Training vs validation loss |
| `sequential_air_forecasting_prediction_graph.png` | Predicted vs actual scatter |
| `sequential_air_forecasting_heatmap.png` | Feature correlation heatmap |
| `sequential_air_forecasting_comparison_graph.png` | Metric comparison bar chart |
| `sequential_air_forecasting_result_graph.png` | Line plot – actual vs predicted |

---

## 📈 Visualization Summary

1. **Heatmap** – correlation of pollutants & weather variables.  
2. **Loss Graph** – convergence of training vs validation loss.  
3. **Prediction Scatter** – predicted vs actual PM2.5 values.  
4. **Comparison Bar** – MAE, RMSE, R² performance.  
5. **Result Line Plot** – temporal view (first 200 samples).  

---

## 📊 Performance Metrics

| Metric | Meaning | Ideal |
|---------|----------|--------|
| **MAE** | Mean Absolute Error | ↓ Lower = Better |
| **RMSE** | Root Mean Squared Error | ↓ Lower = Better |
| **R²** | Coefficient of Determination | ↑ Closer to 1 = Better |


![Confusion Matrix Heatmap](air_forecasting_prediction_graph.png)

---

## 🧮 Example Output
```json
{
    "MAE": 0.0185,
    "RMSE": 0.0327,
    "R2": 0.9762
}
🚀 Run Instructions
Step 1. Install Dependencies
bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pyyaml
Step 2. Run the Script
bash
Copy code
python sequential_air_forecasting_model.py
Step 3. View Outputs
All graphs will be displayed on-screen and saved in the project folder.

🧠 Why Sequential Hybrid?
Algorithm	Strength
AIS	Provides population diversity and broad exploration of search space.
CSA	Performs local exploitation via Lévy-flight search for rapid convergence.
Sequential Hybrid (AIS → CSA)	Combines exploration (global) and exploitation (local) sequentially — achieving balance between accuracy and generalization.

📘 Theoretical Insight
Mathematically, the sequential hybrid minimizes:

Loss
=
MSE
(
𝑦
𝑡
𝑟
𝑢
𝑒
,
𝑦
𝑝
𝑟
𝑒
𝑑
)
subject to optimal parameters 
𝜃
∗
=
arg
⁡
min
⁡
𝜃
𝑓
𝐴
𝐼
𝑆
→
𝐶
𝑆
𝐴
(
𝜃
)
Loss=MSE(y 
true
​
 ,y 
pred
​
 )subject to optimal parameters θ 
∗
 =arg 
θ
min
​
 f 
AIS→CSA
​
 (θ)
where 
𝑓
𝐴
𝐼
𝑆
→
𝐶
𝑆
𝐴
f 
AIS→CSA
​
  denotes the two-stage adaptive search combining immune cloning and Lévy-flight mutation.

📅 Version Info
Model Type: CNN + LSTM Sequential Hybrid (AIS → CSA)

Epochs: 50

Batch Size: 32

Framework: TensorFlow 2.x / Keras

Author: Annan Sadr (NIAT Club / Probox Media)

🧩 Future Improvements
Integrate Parallel Hybrid (AIS ↔ CSA) with concurrent threads.

Include weather forecasting correlation (temperature, humidity, wind).

Deploy via Streamlit / Flask dashboard for real-time prediction.

Automate optimization via n8n flow + Google Sheets reporting.

🏁 Conclusion
This Sequential Hybrid AIS + CSA model demonstrates a practical and high-accuracy framework for AI-driven air-pollution forecasting.
By leveraging evolutionary bio-inspired optimization and deep neural architectures, the system provides reliable environmental intelligence that can assist in urban planning, pollution control, and policy decision-making.
