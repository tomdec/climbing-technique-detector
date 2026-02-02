from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedGroupKFold
from numpy import ones, unique, ndarray, save, load, average
from typing import Tuple, Iterator, List
from os import makedirs
from os.path import join, exists
import matplotlib.pyplot as plt

from src.common.helpers import read_dataframe
from src.rnn.data import WindowGenerator
from src.rnn.model import (
    Rnn,
    RnnConstructorArgs,
    RnnTrainArgs,
    RnnIntMultiRunTrainArgs,
    RnnMultiRunTrainArgs,
    RnnTestArgs,
)


class ExtendedStratifiedGroupKFold:

    def __init__(self):
        self._n_splits = 10
        self._shuffle = False

        self._splitter = StratifiedGroupKFold(
            n_splits=self._n_splits, shuffle=self._shuffle
        )

        self._splits: List[Tuple[ndarray, ndarray, ndarray]] | None = None

    def split(
        self, X: DataFrame, y: Series, groups: Series
    ) -> Iterator[Tuple[ndarray, ndarray, ndarray]]:

        for n in range(self._n_splits):
            train_temp, test_index = list(self._splitter.split(X, y, groups))[n]
            train_index, val_index = list(
                self._splitter.split(train_temp, y[train_temp], groups[train_temp])
            )[n]

            train_index = train_temp[train_index]
            val_index = train_temp[val_index]

            yield train_index, val_index, test_index

    def split_groups(
        self, X: DataFrame, y: Series, groups: Series
    ) -> Iterator[Tuple[ndarray, ndarray, ndarray]]:

        for split in self.split(X, y, groups):
            train_index, val_index, test_index = split
            train_groups = unique(groups[train_index])
            val_groups = unique(groups[val_index])
            test_groups = unique(groups[test_index])

            yield train_groups, val_groups, test_groups


def iterate_group_splits(
    full_data: DataFrame, splitter: ExtendedStratifiedGroupKFold
) -> Iterator[Tuple[ndarray, ndarray, ndarray]]:
    feature_placeholder = ones(shape=(full_data.shape[0]))
    labels = full_data["label"]
    groups = full_data["group"]

    return splitter.split_groups(feature_placeholder, labels, groups)


class RnnFoldCrossValidation:

    @property
    def model_constructor_args(self) -> RnnConstructorArgs:
        return self._model_args

    def __init__(self, model_args: RnnConstructorArgs):
        self._model_args = model_args
        self._splitter = ExtendedStratifiedGroupKFold()

    def get_full_data_list(self) -> DataFrame:
        path_to_all = join(
            self.model_constructor_args.data_root_path, "df", "rnn", "cvs_features.pkl"
        )
        return read_dataframe(path_to_all)

    def train_folds(
        self, train_run_args: RnnIntMultiRunTrainArgs, verbose: bool = False
    ):
        full_data = self.get_full_data_list()

        feature_placeholder = ones(shape=(full_data.shape[0]))
        labels = full_data["label"]
        groups = full_data["group"]

        split_iterator = self._splitter.split(feature_placeholder, labels, groups)

        for fold_index, split in enumerate(split_iterator):
            train_index, val_index, test_index = split
            train_groups = unique(groups[train_index])
            val_groups = unique(groups[val_index])
            test_groups = unique(groups[test_index])

            if verbose:
                print(f"Fold {fold_index+1}:")
                print(f"  Train: groups={train_groups}")
                print(f"  Val:  groups={val_groups}")
                print(f"  Test:  groups={test_groups}")

            fold_num = fold_index + 1
            self.__train_fold(
                train_run_args,
                fold_num,
                full_data,
                train_groups,
                val_groups,
                test_groups,
                verbose,
            )

        self.print_box_plot()

    def print_box_plot(self):
        metrics = self.get_test_accuracy_metrics()

        print(f"Average Top 1 categorical accuracy: {average(metrics)}")

        plt.figure()
        plt.boxplot(metrics)
        plt.show()

    def get_test_accuracy_metrics(self) -> List[float]:
        def get_metric(model: Rnn) -> float:
            return model.get_test_accuracy_metric()

        models = list(map(self.__init_fold_model, range(1, 11)))
        return list(map(get_metric, models))

    def test_folds(self):
        full_data = self.get_full_data_list()

        for fold_index in range(self._splitter._n_splits):
            fold_num = fold_index + 1

            model = self.__init_fold_model(fold_num)
            model_dir = model._get_model_dir()

            if self.__split_files_exist(model_dir):
                (train_groups, val_groups, test_groups) = self.__load_split(model_dir)
            else:
                raise Exception(
                    "split files should already exist when just testing the models"
                )

            wg = self.__build_fold(
                fold_num, full_data, train_groups, val_groups, test_groups
            )

            additional_config = self.__get_additional_config(
                context_config={"fold": fold_num}
            )
            model.test_model(
                args=RnnTestArgs(
                    window_generator=wg,
                    write_to_wandb=True,
                    additional_config=additional_config,
                )
            )

        self.print_box_plot()

    def __get_additional_config(self, context_config: dict = {}) -> dict:
        return context_config | {
            # add values from model_initialize_args
        }

    def __train_fold(
        self,
        train_run_args: RnnIntMultiRunTrainArgs,
        fold_num: int,
        data: DataFrame,
        train_groups: list,
        val_groups: list,
        test_groups: list,
        verbose: bool,
    ):
        model = self.__init_fold_model(fold_num)
        model_dir = model._get_model_dir()

        if self.__split_files_exist(model_dir):
            (train_groups, val_groups, test_groups) = self.__load_split(model_dir)
        else:
            self.__save_split(model_dir, (train_groups, val_groups, test_groups))

        wg = self.__build_fold(
            fold_num, data, train_groups, val_groups, test_groups, verbose
        )

        additional_config = self.__get_additional_config(
            context_config={"fold": fold_num}
        )

        rnn_train_run_args = RnnMultiRunTrainArgs.from_intermediate(wg, train_run_args)
        rnn_train_run_args.train_args.add_config(additional_config)
        model.execute_train_runs(rnn_train_run_args)

        model.test_model(
            args=RnnTestArgs(
                window_generator=wg,
                write_to_wandb=True,
                additional_config=additional_config,
            )
        )

    def __init_fold_model(self, fold_num: int) -> Rnn:
        adapted_args = self._model_args.copy_with(
            name=f"{self._model_args.name}-fold{fold_num}"
        )
        return Rnn(adapted_args)

    def __split_files_exist(self, model_dir) -> bool:
        return (
            exists(join(model_dir, "split", "train.npy"))
            and exists(join(model_dir, "split", "val.npy"))
            and exists(join(model_dir, "split", "test.npy"))
        )

    def __load_split(self, model_dir) -> Tuple[ndarray, ndarray, ndarray]:
        return (
            load(join(model_dir, "split", "train.npy")),
            load(join(model_dir, "split", "val.npy")),
            load(join(model_dir, "split", "test.npy")),
        )

    def __save_split(self, model_dir, split: Tuple[ndarray, ndarray, ndarray]):
        (train, val, test) = split

        makedirs(join(model_dir, "split"), exist_ok=True)
        save(join(model_dir, "split", "train.npy"), train)
        save(join(model_dir, "split", "val.npy"), val)
        save(join(model_dir, "split", "test.npy"), test)

    def __build_fold(
        self,
        fold_num: int,
        data: DataFrame,
        train_groups: list,
        val_groups: list,
        test_groups: list,
        verbose: bool,
    ) -> WindowGenerator:
        if verbose:
            print(f"Building fold {fold_num} ...")

        wg = WindowGenerator(
            data,
            train_groups,
            val_groups,
            test_groups,
            input_width=self.model_constructor_args.model_initialize_args.input_width,
            spacing=self.model_constructor_args.model_initialize_args.spacing,
        )
        if verbose:
            print(wg)
            wg.inspect_fold_split()
        return wg
