#!/bin/bash
#BSUB -J project_cpu
#BSUB -o batch_output/project_cpu_%J.out
#BSUB -e batch_output/project_cpu_%J.err
#BSUB -q hpc
#BSUB -W 10
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"

## #BSUB -R "select[model==XeonGold6226R]"
## Use `nodestat -F hpc` to see a list of resources/models on hpc queue


# Init and activate conda env for py-hpc
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run python script
# python [-u] program.py
# -u option to unbuffer the print lines so we can see in bpeek
# or use sys.flush in the script
/usr/bin/time -f"mem=%M KB runtime=%e s" python -u simulate.py 10