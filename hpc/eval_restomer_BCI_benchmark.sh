✔ [15:05] vsc21216@login2.leibniz $VSC_DATA/projects/code/NAFNet/hpc $ cat $VSC_DATA/projects/jobs/eval_restormer_BCI_benchmark.sh
#!/bin/bash
#SBATCH --job-name=eval_restormer_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab
#SBATCH -p broadwell
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/eval_restormer_BCI_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/eval_restormer_BCI_%j.err
set -euo pipefail
CONTAINER="/scratch/antwerpen/grp/ap_invilab_td_thesis/evaluate_nvidia.sif"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/BCI.sqsh"
GRP_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis"
srun apptainer exec -B "$DATA_SQSH":/data:image-src=/  -B "$GRP_DIR":/grp -B "$VSC_DATA/evaluate":/evaluate "$CONTAINER" python3 /evaluate/evaluate.py         --pred   /grp/benchmark_inference/restormer_BCI/predicted         --gt     /data/test/IHC         --model_name   restormer_BCI         --dataset_name BCI         --split_name   test         --match_by     stem         --output       /grp/benchmark_results.csv         --device       cpu