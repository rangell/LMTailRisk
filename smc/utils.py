from pathlib import Path
import datetime
import time
import os
import random
import json
import torch
import torch.distributed as dist


def setup_distributed(backend="nccl", verbose=True):
    """
    Initialize PyTorch distributed training environment.

    Returns:
        rank (int): global rank
        world_size (int): total number of processes
        local_rank (int): rank on the local node
        device (torch.device)
        distributed (bool): whether distributed was initialized
    """

    # Default (single-process)
    distributed = False
    rank = 0
    local_rank = 0
    world_size = 1

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun / torch.distributed.launch
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = True
        torch.cuda.set_device(local_rank)

    elif "SLURM_PROCID" in os.environ:
        # SLURM
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()
        world_size = int(os.environ["SLURM_NTASKS"])
        distributed = True

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=rank,
        )
        dist.barrier()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if verbose and rank == 0:
        print(
            f"Distributed: {distributed} | "
            f"World size: {world_size} | "
            f"Backend: {backend}"
        )

    return rank, world_size, local_rank, device, distributed


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    is_main = (not is_distributed()) or dist.get_rank() == 0
    return is_main


def wait_for_everyone():
    if is_distributed():
        dist.barrier()


def custom_print(obj):
    if is_main_process():
        print(str(obj).replace("\n", "\n\r"), end="\n\r", flush=True)


def create_output_dir(args):
    output_dir = args.output_dir

    if is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_dir) / args.model_shortname / timestamp

        try:
            os.makedirs(output_dir)
        except FileExistsError:
            custom_print(f"Output directory {output_dir} already exists.")
            # wait for a random time to avoid overwriting
            wait_time = random.randint(1, 200)
            custom_print(f"Waiting for {wait_time} seconds before proceeding...")
            time.sleep(wait_time)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = args.output_dir
            output_dir = Path(output_dir) / args.model_shortname / timestamp
            os.makedirs(output_dir, exist_ok=False)

        metadata = vars(args)
        metadata["timestamp"] = timestamp
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
        custom_print(f"Experiment metadata saved to {metadata_file}")

    return output_dir
