from typing import Dict, Callable
from wandb import Api
from wandb.apis.public.runs import Run

def no_update(_: Run) -> Dict:
    return {}

def add_dropout(run: Run) -> Dict:
    dropout_rate = 0.3 if "-dr0.3-" in run.name else 0.1
    return {"dropout_rate": dropout_rate}

hpe_dnn_filter: Dict = {"$and": [{"group": { "$eq": "hpe_dnn" }}]}

def update_run(filters: Dict = {},
        value_calculator: Callable[[Run], Dict] = no_update):
    api = Api()
    entity = api.default_entity
    project = "detect-climbing-technique"

    runs = api.runs(f"{entity}/{project}", filters)
    
    for run in runs:
        config_patch = value_calculator(run)
        run.config.update(config_patch)
        run.update()

def update_dropout_rate():
    update_run(filters = hpe_dnn_filter, value_calculator=add_dropout)