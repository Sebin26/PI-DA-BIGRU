# Physics-Informed Wind Speed Prediction Using BiGRU and Dual Attention

## Abstract

Accurate short-term wind speed prediction is essential for maintaining grid stability and improving renewable energy integration. Wind patterns are highly nonlinear and dynamic due to changing weather conditions, turbulence, and terrain effects. Traditional decomposition-based methods often suffer from high computational cost and data leakage issues, while purely deep learning-based approaches lack physical interpretability.

This project proposes a Physics-Informed Bidirectional GRU (BiGRU) framework integrated with a dual attention mechanism and cyclic temporal encoding for improved wind speed forecasting. The model captures both temporal dependencies and physical consistency while improving prediction accuracy.

The framework combines:

* Bidirectional GRU for sequence learning
* Dual attention mechanism for feature importance weighting
* Physics-informed regularization for physical consistency
* Cyclic temporal encoding for periodic wind behavior

Experimental results demonstrate improved forecasting performance in terms of MAE, RMSE, and MAPE.

---

## Features

* Physics-Informed Deep Learning
* Bidirectional GRU Architecture
* Dual Attention Mechanism
* Cyclic Temporal Encoding
* Wind Speed Forecasting
* Evaluation Metrics Visualization
* Time-Series Prediction

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Google Colab

---

## Research Contributions

* Introduced physics-informed regularization for wind forecasting
* Combined BiGRU with dual attention mechanism
* Applied cyclic temporal encoding to capture periodic wind behavior
* Improved forecasting accuracy and temporal dependency learning

---

## Project Structure

```text
physics-informed-wind-speed-prediction/
│
├── notebooks/
│   └── research_model.ipynb
│
├── data/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebook using Google Colab or Jupyter Notebook and run all cells.

Main notebook:

```text
notebooks/research_model.ipynb
```

---

## Results

The proposed model demonstrated improved forecasting performance with better MAE, RMSE, and MAPE values compared to baseline approaches.

---

## Author

Sebin Aji

---

## License

This project is licensed under the MIT License.
