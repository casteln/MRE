#!/bin/bash
#SBATCH --job-name=7-43-1-0        # Jobname
#SBATCH --nodes=1               # Number of Nodes
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00         # Expected runtime HH:MM:SS
#SBATCH --output=%x.o%j        # nom du fichier de sortie, %x is the jobname, %j the jobid
#SBATCH --error=%x.o%j           # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=drd@v100              # To specify cpu accounting: <account> = echo $IDRPROJ
##SBATCH --qos=qos_gpu-dev           # Uncomment for job requiring less than 2 hours

# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
export PATH=$WORK/.local/bin:$PATH
module load pytorch-gpu/py3/1.10.1
 
# echo des commandes lancees
set -x
 
# execution du code


python3 -u nvt_md.py ZIF4_initial.xyz l2_liquid_stress.pth 300 0.5 400 200000 ZIF4
