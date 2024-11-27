import warnings
import os
import sys
import random

sys.path.append("../../../")

import yaml
import wandb
import torch
import argparse
import numpy as np
from typing import Dict
from termcolor import colored
from accelerate import Accelerator
from losses.losses import build_loss_fn
from optimizers.optimizers import build_optimizer
from optimizers.schedulers import build_scheduler
from train_scripts.trainer_ddp import Segmentation_Trainer
from architectures.build_architecture import build_architecture
from dataloaders.build_dataset import build_dataset, build_dataloader

##################################################################################################
# Sweep Config

sweep_config = {
    "method": "bayes",  
    "metric": {
        "name": "val_loss",  
        "goal": "minimize"
    },
    "parameters": {
        "embed_dims": {
            "values": [[32, 64, 160, 256], [64, 128, 256, 512]]
        },
        "patch_kernel_size": {
            "values": [[7, 3, 3, 3], [5, 3, 3, 3]]
        },
        "patch_stride": {
            "values": [[4, 2, 2, 2], [2, 2, 2, 2]]
        },
        "mlp_ratios": {
            "values": [[4, 4, 4, 4], [4, 8, 4, 4]]
        },
        "num_heads": {
            "values": [[1, 2, 5, 8], [2, 4, 8, 16]]
        },
        "depths": {
            "values": [[2, 2, 2, 2], [2, 3, 4, 5]]
        },
        "decoder_dropout": {
            "values": [0.0, 0.1, 0.2]
        },
        # Flatten optimizer_args
        "lr": {
            "values": [0.0001, 0.0005, 0.00001]
        },
        "weight_decay": {
            "values": [0.01, 0.001, 0.0001]
        },
        # Flatten scheduler_args
        "min_lr": {
            "values": [0.000006, 0.00001, 0.000005]
        }
    }
}




##################################################################################################
def launch_experiment(config_path, weights_path) -> Dict:
    """
    Builds Experiment
    Args:
        config (Dict): configuration file

    Returns:
        Dict: _description_
    """
    
    # load config
    config = load_config(config_path)

    # set seed
    seed_everything(config)

    # build directories
    build_directories(config)

    # build training dataset & training data loader
    trainset = build_dataset(
        dataset_type=config["dataset_parameters"]["dataset_type"],
        dataset_args=config["dataset_parameters"]["train_dataset_args"],
    )
    trainloader = build_dataloader(
        dataset=trainset,
        dataloader_args=config["dataset_parameters"]["train_dataloader_args"],
        config=config,
        train=True,
    )

    # build validation dataset & validataion data loader
    valset = build_dataset(
        dataset_type=config["dataset_parameters"]["dataset_type"],
        dataset_args=config["dataset_parameters"]["val_dataset_args"],
    )
    valloader = build_dataloader(
        dataset=valset,
        dataloader_args=config["dataset_parameters"]["val_dataloader_args"],
        config=config,
        train=False,
    )

    # build the Model
    model = build_architecture(config)
    if weights_path:
        print("LOADING WEIGHTS FROM: ", weights_path)
        model.load_state_dict(torch.load(weights_path), strict=False)
    # set up the loss function
    criterion = build_loss_fn(
        loss_type=config["loss_fn"]["loss_type"],
        loss_args=config["loss_fn"]["loss_args"],
    )

    # set up the optimizer
    optimizer = build_optimizer(
        model=model,
        optimizer_type=config["optimizer"]["optimizer_type"],
        optimizer_args=config["optimizer"]["optimizer_args"],
    )

    # set up schedulers
    warmup_scheduler = build_scheduler(
        optimizer=optimizer, scheduler_type="warmup_scheduler", config=config
    )
    training_scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="training_scheduler",
        config=config,
    )

    # use accelarate
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=config["training_parameters"][
            "grad_accumulate_steps"
        ],
    )
    accelerator.init_trackers(
        project_name=config["project"],
        config=config,
        init_kwargs={"wandb": config["wandb_parameters"]},
    )

    # display experiment info
    display_info(config, accelerator, trainset, valset, model)

    # convert all components to accelerate
    model = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    trainloader = accelerator.prepare_data_loader(data_loader=trainloader)
    valloader = accelerator.prepare_data_loader(data_loader=valloader)
    warmup_scheduler = accelerator.prepare_scheduler(scheduler=warmup_scheduler)
    training_scheduler = accelerator.prepare_scheduler(scheduler=training_scheduler)

    # create a single dict to hold all parameters
    storage = {
        "model": model,
        "trainloader": trainloader,
        "valloader": valloader,
        "criterion": criterion,
        "optimizer": optimizer,
        "warmup_scheduler": warmup_scheduler,
        "training_scheduler": training_scheduler,
    }

    # set up trainer
    trainer = Segmentation_Trainer(
        config=config,
        model=storage["model"],
        optimizer=storage["optimizer"],
        criterion=storage["criterion"],
        train_dataloader=storage["trainloader"],
        val_dataloader=storage["valloader"],
        warmup_scheduler=storage["warmup_scheduler"],
        training_scheduler=storage["training_scheduler"],
        accelerator=accelerator,
    )

    # run train
    
    trainer.train()

def sweep_train(sweep_dict):
    with wandb.init() as run:
        config = run.config
        config.update(sweep_dict)
        
        build_directories(config)
        
        launch_experiment(config)

def launch_exp_with_sweep(config_path, sweep_dict) -> Dict:
    """
    Initialize and run a sweep experiment with W&B based on the configuration.
    
    Args:
        config_path (str): Path to the YAML config file.
        sweep_dict (Dict): Sweep configuration dictionary.

    Returns:
        Dict: A dictionary with sweep and run information.
    """
    # Load the configuration from the YAML file
    config = load_config(config_path)
    
    # Initialize the W&B sweep
    sweep_id = wandb.sweep(sweep_dict, project=config["project"])

    # Define the sweep function
    def sweep_train_wrapper():
        with wandb.init() as run:
            config.update(run.config)
            launch_experiment(config_path)

    # Run the sweep with the specified sweep ID and function
    wandb.agent(sweep_id, function=sweep_train_wrapper)

    # Return the sweep ID for reference
    return {"sweep_id": sweep_id}

##################################################################################################
def seed_everything(config) -> None:
    seed = config["training_parameters"]["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##################################################################################################
def load_config(config_path: str) -> Dict:
    """loads the yaml config file

    Args:
        config_path (str): _description_

    Returns:
        Dict: _description_
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

##################################################################################################
def build_directories(config: Dict) -> None:
    """
    Create necessary directories, ensuring a unique checkpoint path for each run.

    Args:
        config (Dict): Configuration dictionary.
    """
    # Use W&B run ID to create a unique folder if W&B is initialized
    run_id = wandb.run.id if wandb.run else "default_run"
    checkpoint_dir = os.path.join(config["training_parameters"]["checkpoint_save_dir"], run_id)
    
    # Update config with the dynamic checkpoint directory
    config["training_parameters"]["checkpoint_save_dir"] = checkpoint_dir

    # Create the directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"[info] Checkpoint directory created: {checkpoint_dir}")


##################################################################################################
def display_info(config, accelerator, trainset, valset, model):
    # print experiment info
    accelerator.print(f"-------------------------------------------------------")
    accelerator.print(f"[info]: Experiment Info")
    accelerator.print(
        f"[info] ----- Project: {colored(config['project'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Name: {colored(config['wandb_parameters']['name'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Batch Size: {colored(config['dataset_parameters']['train_dataloader_args']['batch_size'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Num Epochs: {colored(config['training_parameters']['num_epochs'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Loss: {colored(config['loss_fn']['loss_type'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Optimizer: {colored(config['optimizer']['optimizer_type'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Train Dataset Size: {colored(len(trainset), color='red')}"
    )
    accelerator.print(
        f"[info] ----- Test Dataset Size: {colored(len(valset), color='red')}"
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(
        f"[info] ----- Num of Learnable Parameters: {colored(pytorch_total_params, color='red')}"
    )
    accelerator.print(f"-------------------------------------------------------")

##################################################################################################
if __name__ == "__main__":
    warnings.simplefilter("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="Experiment Launcher")
    parser.add_argument("--config", required=True, type=str, help="Path to config file")
    parser.add_argument("--sweep", action="store_true", help="Run with sweep configuration")
    parser.add_argument("--weights", default="", type=str, help="Path to pretrained weights file")
    args = parser.parse_args()
    

    config_path = args.config

    if args.sweep:
        # Launch experiment with sweep
        launch_exp_with_sweep(config_path, sweep_config)
    else:
        # Launch standard experiment
        launch_experiment(config_path, args.weights)
