"""
    SBatch command builder and Slurm launcher
"""
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

@dataclass
class StoolArgs:
    # Slurm Parameters
    job_name = "aim_v2_training"
    job_dir: str = "./slurm-jobs/" # Where we'll dump the SBATCH shell script generated.
    nnodes: int = 1
    ntasks_per_node: int = 1
    ngpus: int = 1
    ncpus_per_task: int = 16
    partition: str = 'main'

    # Train Parameters
    batch_size: int = 1

    # Environment Parameters
    mlflow_uri: str = "https://localhost:5000"
    conda_env: str = "cloud-slurm-aimv2"
    output_dir: str = "./logs"

def launch_job(args: StoolArgs) -> None:
    SBATCH_CMD = """
    #!/bin/bash
    #SBATCH --job-name={job_name}
    #SBATCH --nodes={nnodes}
    #SBATCH --ntasks-per-node={ntasks_per_nodes}
    #SBATCH --gpus={ngpus}
    #SBATCH --cpus-per-task={ncpus_per_task}
    #SBATCH --partition={partition}
    #SBATCH --output={output_dir}/{job_name}-%j.log

    export MASTER_PORT=12355
    export WORLD_SIZE=$(({nnodes} * {ntasks_per_nodes}))

    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr

    export MLFLOW_TRACKING_URI={mlflow_uri}
    
    echo "SLURM WORLD_SIZE="$WORLD_SIZE
    echo "SLURM MASTER_PORT="$MASTER_PORT
    echo "SLURM MASTER_ADDR="$MASTER_ADDR
    echo "MLFLOW MLFLOW_TRACKING_URI"=$MLFLOW_TRACKING_URI

    conda activate {conda_env}

    torchrun --nnodes={nnodes} --nproc-per-node={ngpus} train.py
    """
    sbatch = SBATCH_CMD.format(
        job_name=args.job_name,
        nnodes=args.nnodes,
        ntasks_per_nodes=args.ntasks_per_node,
        ngpus=args.ngpus,
        ncpus_per_task=args.ncpus_per_task,
        partition=args.partition,

        mlflow_uri=args.mlflow_uri,
        conda_env=args.conda_env,
        output_dir=args.output_dir
    )

    # Dump sbatch command
    sbatch_out = str(Path(args.job_dir) / args.job_name) + '.slurm'
    with open(sbatch_out, 'w') as fi:
        fi.write(sbatch)

    # System call
    os.system(f"sbatch {sbatch_out}")
    
if __name__ == '__main__':
    args = OmegaConf.from_cli()

    default = OmegaConf.structured(StoolArgs)
    OmegaConf.set_struct(default, True)

    # Override any StoolArgs from user CLI
    override = OmegaConf.create(args)
    args = OmegaConf.to_object(OmegaConf.merge(default, override))

    # Launch Job
    launch_job(args)