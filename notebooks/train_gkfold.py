# wandb essentials
import argparse, os
from types import SimpleNamespace
import wandb

# Data manipulation
import numpy as np

# Utils
from utils import make
from train import train

# File management
from datetime import datetime

default_config = SimpleNamespace(
    epochs=50,
    classes=1,
    n_channels=1,
    batch_size=8,
    learning_rate=0.000015528833190894192,
    crop=True,
    normalize=True,
    augmented=False,
    # resize=150,
    optimizer='adam',
    dataset="ThermalBreastCancer",
    architecture="alexnet")

def parse_args():
    "Override default argments"
    argparser = argparse.ArgumentParser(description="Process hyper-parameters")
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help="batch size")
    argparser.add_argument('--learning_rate', type=float, default=default_config.learning_rate, help="learning rate")
    argparser.add_argument('--optimizer', type=str, default=default_config.optimizer, help="optimizer")
    argparser.add_argument('--normalize', type=bool, default=default_config.normalize, help="normalize")
    # argparser.add_argument('--resize', type=bool, default=default_config.resize, help="resize")
    argparser.add_argument('--augmented', type=bool, default=default_config.augmented, help="augmented")
    argparser.add_argument('--architecture', type=str, default=default_config.architecture, help="architecture")
    argparser.add_argument('--crop', type=bool, default=default_config.crop, help="crop")
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def model_pipeline(num, sweep_id, sweep_run_name, hyperparameters, timestamp:str):

    # tell wandb to get started
    run_name = f'{sweep_run_name}--{num}'
    with wandb.init(config=hyperparameters, group=sweep_id, job_type=sweep_run_name, name=run_name, reinit=True):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer, accuracy_fn, f1_score_fn, recall_fn, precision_fn, epochs = make(config, num)
        # print(model)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        gkfold_path = f"../models/{config.architecture}_{timestamp}"

        # and use them to train the model
        test_accuracy, test_f1, test_recall, test_precision = train(model, train_loader, test_loader, criterion, optimizer, accuracy_fn, f1_score_fn, recall_fn, precision_fn, epochs, gkfold_path)
            
        # get metrics of the model    
        # test_accuracy, test_f1, test_recall, test_precision = test(model, test_loader, accuracy_fn, f1_score_fn, recall_fn, precision_fn)

    return test_accuracy, test_f1, test_recall, test_precision

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]

def cross_validate(config):

    sweep_run = wandb.init(config=config, project="dip-project") # Inicio el sweep
    sweep_id = sweep_run.sweep_id or "unknown" # Agarro el id del sweep
    # sweep_url = sweep_run.get_sweep_url() # Agarro el url del sweep
    project_url = sweep_run.get_project_url() # Agarro la url del proyecto del sweep
    sweep_group_url = f'{project_url}/groups/{sweep_id}' # Armo un string con la url del
    # proyecto y el id del sweep, para poder agrupar supongo 
    sweep_run.notes = sweep_group_url # Asigno en las notas del sweep la ruta que acabe
    # de crear
    sweep_run.save() # Guardo el sweep con las rutas y esas cosas hechas anteriormente
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2" # Armo el string con el
    # nombre del run del sweep
    sweep_run_id = sweep_run.id # Consigo el id del run del sweep
    sweep_run.finish() # Puaso el sweep
    wandb.sdk.wandb_setup._setup(_reset=True) #  resets the settings which are set during the
    # sweep run initialization. This is crucial and acts as a workaround because it resets the
    # settings which causes the new run in a sweep to use the same run ID and settings upon it's initialization.

    # Diccionario para guardar las métricas de cada run
    metrics = {
        "test_accuracy": [],
        "test_recall": [],
        "test_precision": [],
        "test_f1_score": []
    }

    # to save the model at the current time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    for fold in range(1, 8):

        reset_wandb_env() # Reinicio las variables de entorno en cada run

        # Entreno y valido cada run pasandole el sweep y la config
        test_accuracy, test_f1, test_recall, test_precision = model_pipeline(
            sweep_id=sweep_id, num=fold,
            sweep_run_name=sweep_run_name,
            hyperparameters=dict(sweep_run.config),
            timestamp=timestamp
        )

        # Agrego las métricas del run actual
        metrics["test_accuracy"].append(test_accuracy.cpu())
        metrics["test_recall"].append(test_f1.cpu())
        metrics["test_precision"].append(test_recall.cpu())
        metrics["test_f1_score"].append(test_precision.cpu())

    # resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")

    # Log metrics to sweep run
    for metric, values in metrics.items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        # Average of each metric over all the folds in an experiment
        print(f'{metric.capitalize()}:')
        print(f'  Average: {avg_value*100:.2f}% (+/- {std_value*100:.2f}%)')
        for fold, value in enumerate(values):
            sweep_run.log({
                f'fold_{metric}': value,
                'fold': fold+1
            })

    sweep_run.finish()

if __name__ == "__main__":
    parse_args()
    cross_validate(config=default_config)