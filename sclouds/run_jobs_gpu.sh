
#!/bin/bash
SBATCH -p dgx2q                  # partition (queue)
SBATCH -N 1                      # number of nodes
SBATCH -n 16                     # number of cores
#    #SBATCH -w g001              # for a specific nod
SBATCH --gres=gpu:1              # for e.g. 1 V100 GPUs
SBATCH --mem 1024G               # memory pool for all cores
SBATCH -t 1-24:00                # time (D-HH:MM)
SBATCH -o output/slurm.%N.%j.out # STDOUT
SBATCH -e output/slurm.%N.%j.err # STDERR
#     #SBATCH --exclusive         # If you want to benchmark and have node for yourself
SBATCH --mail-user=hannasv@fys.uio.no
SBATCH --mail-type=ALL

ulimit -s 10240
mkdir -p ~/output

module purge
module load slurm/18.08.8
# module load gcc/9.2.0
# module load openmpi/gcc/64/4.0.2
# module load ex3-modules
module load cuda10.1/toolkit/10.1.243
module load python/3.7.4
/home/hannasv py37-venv/bin/activate # original: .  py37-venv/bin/activate



#srun -n $SLURM_NTASKS  /home/hannasv/a.out
# a.out er standard for kompilerte program i Unix/Linux ...

srun -n $SLURM_NTASKS python model.py
