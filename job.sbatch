#!/bin/bash
#SBATCH --account=renjie2-ic
#SBATCH --job-name=molformer
#SBATCH --output=molformer_output_%j.log
#SBATCH --error=molformer_error_%j.log
#SBATCH --partition=IllinoisComputes-GPU   # Partition name
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --gres=gpu:2                # Number of GPUs                

# Optional: set account or qos if required by your cluster
# SBATCH --qos=your_qos

# Load any necessary modules (adjust for your cluster)
module load python/3.11
module load cuda/12.4
source /projects/illinois/eng/ece/lgoddard/renjie2/fine-tuning/myChem2/bin/activate
pip3 install POT

# Run your program
python train_molformer.py