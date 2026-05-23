#!/bin/bash

# ============================================================================
# QUICK START GUIDE - DA-BiGRU Wind Speed Forecasting
# ============================================================================

# This script provides a quick reference for getting started

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════╗
║     DA-BiGRU WIND SPEED FORECASTING - QUICK START GUIDE          ║
╚══════════════════════════════════════════════════════════════════╝

📋 PREREQUISITES:
  ✓ Python 3.8+
  ✓ pip or conda
  ✓ 4GB+ RAM (8GB+ with GPU)
  ✓ CUDA 11.0+ for GPU acceleration (optional)

═══════════════════════════════════════════════════════════════════

🚀 INSTALLATION (5 minutes)

1. Navigate to project directory:
   cd "PI DA BIGRU"

2. Install dependencies:
   pip install -r requirements.txt

   (Takes ~5-10 minutes depending on system)

═══════════════════════════════════════════════════════════════════

🎯 USAGE OPTION 1: Complete Pipeline (Recommended)

Run everything in one command:

   bash scripts/run_pipeline.sh

This automatically:
  1. Prepares and preprocesses data
  2. Trains the DA-BiGRU model
  3. Evaluates and generates visualizations

⏱️  Typical Runtime:
  - Data Prep: 2-5 minutes
  - Training: 5-15 minutes (depends on data size and GPU)
  - Evaluation: 1-3 minutes
  Total: 10-30 minutes

📁 Outputs:
  - results/metrics.txt              → Performance metrics
  - results/predictions.csv          → Predictions data
  - results/diagnostic_dashboard.png → Visualizations
  - models/checkpoints/best_model.pth → Trained model

═══════════════════════════════════════════════════════════════════

🔧 USAGE OPTION 2: Step by Step

If you want to run each step separately:

Step 1 - Prepare Data:
  bash scripts/prepare_data.sh

Step 2 - Train Model:
  bash scripts/train.sh

Step 3 - Evaluate Results:
  bash scripts/evaluate.sh

═══════════════════════════════════════════════════════════════════

📓 USAGE OPTION 3: Jupyter Notebook (Interactive)

For interactive exploration:

  jupyter notebook notebooks/DA_BIGRU.ipynb

Then run cells in sequence. This gives you:
  - Control over each step
  - Interactive visualization
  - Easy parameter experimentation
  - Access to intermediate results

═══════════════════════════════════════════════════════════════════

⚙️  IMPORTANT: PREPARE YOUR DATA

Before running, place your CSV file at:

  data/raw/merged_weather_data.csv

CSV Format Requirements:
  - Columns: YEAR, MO, DY, HR, WS10M, WS50M, WS100M, T2M, PS
  - Each row: one hour of data
  - Example header:
    YEAR,MO,DY,HR,WS10M,WS50M,WS100M,T2M,PS
    2020,1,1,0,5.2,7.3,8.1,15.2,101325

If you don't have data:
  1. Add your own CSV to data/raw/
  2. Update DATA_PATH in scripts/prepare_data.sh
  3. Run the pipeline

═══════════════════════════════════════════════════════════════════

📊 VIEW RESULTS

After pipeline completes:

1. Text Metrics:
   cat results/metrics.txt

2. CSV Predictions:
   head -20 results/predictions.csv

3. Visualization:
   open results/diagnostic_dashboard.png
   (or use your image viewer)

═══════════════════════════════════════════════════════════════════

🔧 CUSTOMIZATION (Optional)

Modify training parameters in config.yaml:

  epoch: 50              → Number of training epochs
  batch_size: 64         → Batch size (reduce if GPU error)
  learning_rate: 0.001   → Learning rate
  patience: 10           → Early stopping patience

Or pass via command line:
  bash scripts/train.sh --epochs 100 --batch_size 32

═══════════════════════════════════════════════════════════════════

🐛 COMMON ISSUES & FIXES

❌ "No such file or directory: data/raw/merged_weather_data.csv"
✅ Solution: Place your CSV in data/raw/ directory

❌ "CUDA out of memory"
✅ Solution: Reduce batch_size in config or train on CPU

❌ "ImportError: No module named 'torch'"
✅ Solution: pip install -r requirements.txt

❌ "Permission denied: scripts/*.sh"
✅ Solution: chmod +x scripts/*.sh

═══════════════════════════════════════════════════════════════════

📚 LEARN MORE

For detailed information:
  - README.md              → Complete documentation
  - REFACTORING_SUMMARY.md → Architecture overview
  - notebooks/DA_BIGRU.ipynb → Code examples
  - config.yaml            → All configuration options

═══════════════════════════════════════════════════════════════════

💡 TIPS & TRICKS

1. Monitor training:
   tail -f results/training_log.txt  (if enabled)

2. Use GPU acceleration:
   CUDA is auto-detected. Verify with:
   python -c "import torch; print(torch.cuda.is_available())"

3. Experiment with hyperparameters:
   Edit config.yaml and re-run scripts/train.sh

4. Batch processing multiple datasets:
   Create shell script that calls run_pipeline.sh multiple times

5. Access trained model:
   model_package = torch.load('models/complete_model_package.pth')
   best_model = model_package['model_state_dict']

═══════════════════════════════════════════════════════════════════

✅ NEXT STEPS

After first run:

1. Review results in results/
2. Adjust parameters if needed
3. Re-run with new settings
4. Deploy to production (see README.md)

═══════════════════════════════════════════════════════════════════

❓ STILL HAVE QUESTIONS?

1. Check Troubleshooting in README.md
2. Review code comments and docstrings
3. Examine scripts/run_pipeline.sh
4. Look at example notebook usage

═══════════════════════════════════════════════════════════════════

🎉 YOU'RE READY!

Start with:
  cd "PI DA BIGRU"
  bash scripts/run_pipeline.sh

That's it! The system will handle everything else.

═══════════════════════════════════════════════════════════════════

Happy forecasting! 📈

EOF

echo ""
echo "To get started, run:"
echo "  cd 'PI DA BIGRU'"
echo "  bash scripts/run_pipeline.sh"
echo ""
