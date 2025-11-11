#!/bin/bash

# Mailing options
##SBATCH --mail-user=nirosh.sivanesan@students.unibe.ch
##SBATCH --mail-type=FAIL,END

# Job name
#SBATCH --job-name="mia_ablation_study"

# Runtime and memory
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

# Partition
#SBATCH --qos=job_gpu_caim
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx4090:1
##SBATCH --gres=gpu:h100:1  # Alternative GPU option
##SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-gpu=32G

# Output and error files
#SBATCH --output=logs/mia_ablation_%j.out
#SBATCH --error=logs/mia_ablation_%j.err

#### Your shell commands below this line ####

# Create logs directory if it doesn't exist
mkdir -p logs
cd /storage/homefs/ns12l060/mialab

# Load and activate conda properly
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate mialab
module load CUDA/12.6.0

# Verify activation worked
echo "Python: $(which python)"
echo "Environment: $CONDA_DEFAULT_ENV"

# Install SimpleITK if missing
# python -c "import SimpleITK" 2>/dev/null || pip install SimpleITK scikit-learn pymia
# pip install scikit-learn pymia pathos pandas matplotlib seaborn 

export PYTHONPATH="/storage/homefs/ns12l060/mialab:$PYTHONPATH"

echo "=========================================="
echo "Starting single-run experiment..."
echo "=========================================="
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Add timestamps and verbose output to Python
export PYTHONUNBUFFERED=1

python -m mia_experiments.cli run \
  --data-atlas ./data/atlas \
  --data-train ./data/train \
  --data-test ./data/test \
  --output-dir ./experiments_001

exit_code=$?

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Exit code: $exit_code"
echo "Timestamp: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✓ Experiment completed successfully"
else
    echo "✗ Experiment failed with exit code $exit_code"
fi

exit $exit_code
