# DA-BiGRU Wind Speed Forecasting System

A production-ready, modular implementation of a Dual-Attention Bidirectional GRU (DA-BiGRU) for hourly wind speed forecasting with physics-informed feature engineering.

## 📖 Abstract

Accurate short-term wind speed prediction is essential for maintaining grid stability and improving renewable energy integration. Wind patterns are highly nonlinear and dynamic due to changing weather conditions, turbulence, and terrain effects. Traditional decomposition-based methods often suffer from high computational cost and data leakage issues, while purely deep learning-based approaches lack physical interpretability.

This project proposes a **refactored, modular implementation** of a Physics-Informed Bidirectional GRU (BiGRU) framework integrated with a dual attention mechanism and cyclic temporal encoding for improved wind speed forecasting. The model captures both temporal dependencies and physical consistency while improving prediction accuracy.

## 🎯 Key Features

* ✅ **Physics-Informed Feature Engineering** - Wind shear, air density, turbulence intensity
* ✅ **Bidirectional GRU Architecture** - Captures forward and backward temporal patterns
* ✅ **Dual Attention Mechanism** - Feature-level and temporal attention
* ✅ **Cyclic Temporal Encoding** - Captures hourly and monthly wind patterns
* ✅ **Modular Architecture** - Separated concerns for data, models, training, evaluation
* ✅ **Production Ready** - Bash orchestration, checkpointing, reproducibility
* ✅ **Comprehensive Evaluation** - Metrics, visualizations, error analysis
* ✅ **Interactive Notebook** - Updated Jupyter notebook using modular components

## 📁 Project Structure (NEW - Refactored)

```
PI DA BIGRU/
├── data/                      # Data loading and preprocessing
│   ├── __init__.py
│   ├── loader.py              # Load CSV and prepare data
│   ├── preprocessor.py        # Feature engineering and sequences
│   ├── raw/                   # Raw input data
│   └── processed/             # Pre-processed output
│
├── models/                    # Neural network architecture
│   ├── __init__.py
│   ├── da_bigru.py            # DA-BiGRU model definition
│   └── checkpoints/           # Saved model checkpoints
│
├── training/                  # Training pipeline
│   ├── __init__.py
│   ├── config.py              # Training hyperparameters
│   └── trainer.py             # Training loop with early stopping
│
├── evaluation/                # Model evaluation
│   ├── __init__.py
│   ├── evaluator.py           # Metrics calculation
│   └── visualizer.py          # 6-panel diagnostic dashboard
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── seeding.py             # Global reproducibility
│   ├── constants.py           # Configuration constants
│   └── logger.py              # Logging utilities
│
├── scripts/                   # Orchestration scripts
│   ├── prepare_data.sh        # Data prep bash script
│   ├── preprocess.py          # Data prep Python module
│   ├── train.sh               # Training bash script
│   ├── train.py               # Training Python module
│   ├── evaluate.sh            # Evaluation bash script
│   ├── evaluate.py            # Evaluation Python module
│   └── run_pipeline.sh        # Complete pipeline runner
│
├── notebooks/
│   └── DA_BIGRU.ipynb         # Refactored Jupyter notebook
│
├── results/                   # Output directory
├── requirements.txt           # Dependencies
├── config.yaml               # Configuration file
└── README.md                 # This file
```

## 🚀 Technologies Used

* **Python** - Core language
* **PyTorch** - Deep learning framework (GPU acceleration)
* **NumPy & Pandas** - Numerical/data processing
* **Scikit-learn** - Preprocessing and metrics
* **Matplotlib & Seaborn** - Visualizations
* **Joblib** - Model serialization
* **Bash** - Pipeline orchestration

---

## 📥 Installation & Quick Start

### 1. Install Dependencies

```bash
# Navigate to project directory
cd "PI DA BIGRU"

# Install all required packages
pip install -r requirements.txt
```

### 2. Quick Start Options

#### Option A: Run Complete Pipeline (Recommended)

```bash
# Automatically: prepare data → train → evaluate
bash scripts/run_pipeline.sh
```

#### Option B: Interactive Jupyter Notebook

```bash
# Open the refactored notebook
jupyter notebook notebooks/DA_BIGRU.ipynb

# Run cells sequentially
```

#### Option C: Step-by-Step Commands

```bash
# Step 1: Prepare and preprocess data
bash scripts/prepare_data.sh

# Step 2: Train the model
bash scripts/train.sh

# Step 3: Evaluate and visualize
bash scripts/evaluate.sh
```

---

## 🏗️ Model Architecture

### DA-BiGRU Components

```
Input Tensor
(batch_size, seq_len=24, n_features=12)
    ↓
╔═════════════════════════════════════════╗
║  Feature Attention Module              ║
║  - Learns which features matter        ║
║  - Applies learned importance weights  ║
╚═════════════════════════════════════════╝
    ↓
╔═════════════════════════════════════════╗
║  Bidirectional GRU (2 layers)          ║
║  - Forward GRU: left to right          ║
║  - Backward GRU: right to left         ║
║  - Concatenated outputs (hidden=128)   ║
╚═════════════════════════════════════════╝
    ↓
╔═════════════════════════════════════════╗
║  Temporal Attention Module             ║
║  - Learns which timesteps matter       ║
║  - Context vector via weighted sum     ║
╚═════════════════════════════════════════╝
    ↓
╔═════════════════════════════════════════╗
║  Dense Classification Head             ║
║  - Linear: 128 → 32                    ║
║  - ReLU activation + Dropout (0.2)     ║
║  - Linear: 32 → 1 (prediction)         ║
╚═════════════════════════════════════════╝
    ↓
Output Tensor
(batch_size, 1)
```

---

## 📊 Data Pipeline

### Input Features (12 Total)

**Meteorological Variables** (5):
- `WS50M` - Wind speed at 50m (TARGET)
- `WS10M` - Wind speed at 10m
- `WS100M` - Wind speed at 100m
- `T2M` - Temperature at 2m
- `PS` - Surface pressure

**Physics-Informed Features** (3):
- `WindShear` - Logarithmic wind profile: log(WS100/WS10) / log(100/10)
- `AirDensity` - Ideal gas law: ρ = P / (R × T)
- `TurbulenceIntensity` - Rolling std/mean of WS50M

**Temporal Features** (4):
- `Hour_Sin`, `Hour_Cos` - Cyclical hour encoding (0-23)
- `Month_Sin`, `Month_Cos` - Cyclical month encoding (1-12)

### Processing Pipeline

```
Raw CSV
(YEAR, MO, DY, HR, WS10M, WS50M, WS100M, T2M, PS)
    ↓
[Load & Index]
Create datetime index from YEAR/MO/DY/HR
    ↓
[Train/Test Split]
80% train, 20% test (temporal order preserved)
    ↓
[Feature Engineering]
Compute WindShear, AirDensity, TurbulenceIntensity
Encode cyclical hours and months
    ↓
[Scaling]
MinMaxScaler fit on TRAIN, applied to both
    ↓
[Sequence Creation]
24-hour sliding windows (24 timesteps → 1 prediction)
    ↓
[PyTorch DataLoaders]
Batched tensors with shuffling (train only)
```

---

## 🎓 Training Pipeline

### Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 (with early stopping) |
| Batch Size | 64 |
| Learning Rate | 0.001 (adaptive reduction) |
| Optimizer | Adam with weight decay (1e-5) |
| Loss Function | Mean Squared Error (MSE) |
| Early Stopping | Patience: 10 epochs |
| Device | CUDA (GPU) or CPU |

### Training Loop

```python
For each epoch:
  1. Train Phase
     - Forward pass through model
     - Calculate MSE loss
     - Backward pass (gradients)
     - Update weights with Adam optimizer
     
  2. Validation Phase
     - Evaluate on test set
     - Calculate validation loss
     - Check for improvement
     - Potentially save checkpoint
     
  3. Learning Rate Schedule
     - Reduce by factor of 0.5 if no improvement
     - After 3 epochs without improvement
```

### Checkpointing

- Best model saved to `models/checkpoints/best_model.pth`
- Includes: model weights, optimizer state, epoch, validation loss
- Loaded for evaluation and inference

---

## 📈 Evaluation & Metrics

### Performance Metrics

```
MAPE  (%)    - Mean Absolute Percentage Error
MAE   (m/s)  - Mean Absolute Error
RMSE  (m/s)  - Root Mean Squared Error
R²           - Coefficient of Determination (0-1)
```

### Wind Category Breakdown

Performance analyzed across wind speed ranges:
- **Calm** (0-2 m/s)
- **Light** (2-5 m/s)
- **Moderate** (5-8 m/s)
- **High** (8+ m/s)

### Diagnostic Dashboard (6 Panels)

1. **Time Series Tracking**
   - First 100 hours actual vs predicted
   - Visual pattern matching

2. **Scatter Plot**
   - Actual vs predicted with perfect fit line
   - R² metric

3. **Error Distribution**
   - Residuals histogram
   - Checks for bias and symmetry

4. **Error by Wind Speed**
   - Residuals vs wind speed
   - Identifies heteroscedasticity

5. **Diurnal Error Profile**
   - Mean error by hour of day
   - Identifies time-of-day patterns

6. **Training History**
   - Train and validation loss curves
   - Shows convergence and overfitting

---

## 💻 Usage Examples

### Python API

```python
# 1. Load preprocessed data
from data.loader import load_processed_data
X_train, y_train, X_test, y_test, scaler, config = load_processed_data('data/processed')

# 2. Create model
from models.da_bigru import DA_BiGRU
model = DA_BiGRU(input_dim=12, hidden_dim=64, n_layers=2, dropout=0.2)

# 3. Train
from training.trainer import Trainer
trainer = Trainer(model, device='cuda')
train_losses, val_losses = trainer.train(train_loader, test_loader, epochs=50)

# 4. Evaluate
from evaluation.evaluator import Evaluator
evaluator = Evaluator(model, device='cuda', scaler=scaler)
preds, actuals = evaluator.generate_predictions(test_loader)
metrics = evaluator.calculate_metrics(actuals, preds)
```

### Command Line

```bash
# Prepare data with custom CSV
python scripts/preprocess.py \
    --input_file data/raw/weather.csv \
    --output_dir data/processed

# Train with custom parameters
bash scripts/train.sh --epochs 100 --batch_size 32 --learning_rate 0.0005

# Full pipeline with verbose logging
bash scripts/run_pipeline.sh
```

---

## 📁 Output Directory Structure

```
results/
├── metrics.txt              # Summary metrics (readable)
├── metrics.json            # Machine-readable metrics
├── predictions.csv         # All predictions and errors
├── diagnostic_dashboard.png # 6-panel visualization
└── training_history.npz    # Loss curves data

models/
├── checkpoints/
│   └── best_model.pth      # Best checkpoint during training
└── complete_model_package.pth  # Final model with scaler + config
```

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
seed: 2026
device: cuda

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  patience: 10

model:
  hidden_dim: 64
  n_layers: 2
  dropout: 0.2

sequence:
  seq_len: 24

data:
  train_test_split: 0.8
```

---

## ⚡ Performance Tips

### For GPU Training
- Ensure CUDA is installed: `torch.cuda.is_available()`
- Monitor GPU memory: `nvidia-smi`
- Reduce batch size if out of memory

### For Faster Training
- Reduce `patience` for early stopping
- Lower learning rate with larger epochs
- Adjust model size (hidden_dim, n_layers)

### For Better Predictions
- Increase sequence length (24 → 48 hours)
- Add more engineered features
- Collect more training data
- Fine-tune hyperparameters

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No processed data found" | Run `bash scripts/prepare_data.sh` |
| "No model checkpoint" | Run `bash scripts/train.sh` |
| CUDA out of memory | Reduce batch_size in config/args |
| Poor predictions | Check data quality, adjust hyperparameters |
| Slow training | Use GPU, reduce seq_len or model size |
| Import errors | Verify all `__init__.py` files exist |

---

## 🔬 Research Implementation

This implementation combines concepts from:
- **BiGRU**: Bi-directional RNNs for sequence learning
- **Attention Mechanisms**: Feature and temporal importance weighting
- **Physics Informatics**: Domain knowledge in feature engineering
- **Time Series Forecasting**: Best practices for temporal data

---

## 📝 Project Improvements (Refactoring)

**From Original Notebook to Modular Architecture:**

✅ Separated data/model/training logic into modules
✅ Created reusable components for different datasets
✅ Added comprehensive error handling
✅ Implemented reproducible seeding
✅ Added bash orchestration scripts
✅ Created production-ready evaluation pipeline
✅ Documented all functions with docstrings
✅ Added configuration management
✅ Improved code organization and maintainability

---

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Time Series Forecasting](https://en.wikipedia.org/wiki/Time_series)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)
- [Wind Energy Resources](https://www.nrel.gov/)

---

## 📄 License

[Add your license here]

## 👤 Authors

[Original & Refactored Implementation]

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: May 23, 2026
**Version**: 2.0 (Modular Architecture Refactor)
**Status**: Production Ready ✅

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebook using Google Colab or Jupyter Notebook and run all cells.

Main notebook:

```text
notebooks/DA_BIGRU.ipynb
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