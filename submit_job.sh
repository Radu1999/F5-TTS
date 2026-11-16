#!/bin/bash
#SBATCH -p dgxa100                   # Partition name
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --ntasks-per-node=1            # Single task per node
#SBATCH -A student                     # Account name
#SBATCH --cpus-per-task=16             # CPUs per task
#SBATCH --mem-per-cpu=4G               # Memory per CPU
#SBATCH --time=12:00:00                # Time limit
#SBATCH -J training_job                # Job name
#SBATCH -o output_%j.log               # Standard output log
#SBATCH -e error_%j.log                # Standard error log

# Load any necessary modules (e.g., for Singularity if required by the system)
module load singularity

# Path to your Singularity image and training bash script
SIF_PATH="tts.sif"         # Update with the path to your SIF image
TRAINING_SCRIPT="run_training.sh" # Update with the path to your training bash script

# Execute the training bash script within the SIF image
DATA_PATH="/export/home/acs/stud/r/radu.chivereanu/F5-TTS/data"
CKPTS_PATH="/export/home/acs/stud/r/radu.chivereanu/F5-TTS/ckpts"

singularity exec --nv \
  --bind $DATA_PATH:/F5-TTS/data \
  --bind $CKPTS_PATH:/F5-TTS/ckpts \
  $SIF_PATH bash $TRAINING_SCRIPT
