from typing import List, Callable
from wandb.apis import PublicApi
from wandb.apis.public.runs import Run

# TODO: update project name
PROJECT_NAME = "video-analysis-project"


def __no_update(_: Run) -> dict:
    return {}


def __add_dropout(run: Run) -> dict:
    dropout_rate = 0.3 if "-dr0.3-" in run.name else 0.1
    return {"dropout_rate": dropout_rate}


def __add_base_name(run: Run) -> dict:
    idx = run.name.find("-fold")
    return {"base_name": run.name[:idx] if idx != -1 else run.name}


def __add_on_full(run: Run) -> dict:
    value = "-full-" in run.name
    print(f"Setting {run.name} to {value}")
    return {"on_full": value}


__hpe_dnn_filter: dict = {"$and": [{"group": {"$eq": "hpe_dnn"}}]}

__unset_on_full_filter: dict = {
    "$and": [{"config.on_full": {"$ne": True}}, {"config.on_full": {"$ne": False}}]
}


def update_run_config(
    filters: dict = {}, config_patcher: Callable[[Run], dict] = __no_update
):
    api = PublicApi()
    entity = api.default_entity

    runs: List[Run] = api.runs(f"{entity}/{PROJECT_NAME}", filters)

    for run in runs:
        config_patch = config_patcher(run)
        run.config.update(config_patch)
        run.update()


def update_dropout_rate():
    update_run_config(filters=__hpe_dnn_filter, config_patcher=__add_dropout)


def update_base_name():
    update_run_config(config_patcher=__add_base_name)


def update_on_full():
    update_run_config(filters=__unset_on_full_filter, config_patcher=__add_on_full)
