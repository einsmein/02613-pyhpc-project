#!/bin/bash
#BSUB -J project_gpu
#BSUB -o batch_output/project_gpu_%J.out
#BSUB -e batch_output/project_gpu_%J.err
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpua100

## #BSUB -q c02613
## #BSUB -R "select[gpu32gb]"
## #BSUB -R "select[model==XeonGold6226R]"
## Use `nodestat -F hpc` to see a list of resources/models on hpc queue


# Init and activate conda env for py-hpc
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run python script
# python [-u] program.py
# -u option to unbuffer the print lines so we can see in bpeek
# or use sys.flush in the script

# /usr/bin/time -f"mem=%M KB runtime=%e s" python -u simulate.py 10
# Run all
/usr/bin/time -f"mem=%M KB runtime=%e s" python -u simulate.py 4571