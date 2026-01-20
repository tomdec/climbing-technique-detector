from typing import List, Tuple
from matplotlib import pyplot as plt
from pandas import DataFrame
from numpy import ones, average, nan, isnan
from numpy.ma import masked_array
from seaborn import stripplot

from src.common.plot import plot_histograms, plot_histogram_grid, save_current_figure
from src.hpe.common.typing import PredictedKeyPoint, HpeEstimation, MyLandmark
from src.hpe.common.metrics import calc_precision_and_recall, calc_average_precisions


def plot_average_distances(distances: DataFrame, title: str, save_location: str = ""):
    x_width = distances.shape[0]
    limit = ones(x_width)

    plt.figure()
    plt.title(title)
    plt.plot(limit, color="red", label="PCKh50 limit")
    plt.xlabel("images")
    plt.ylabel("average relative error")
    plt.legend(loc="upper right")

    masked_distances = masked_array(distances, isnan(distances))
    masked_dist_av = average(masked_distances, 1)
    dist_av = masked_dist_av.filled(nan)

    plt.bar(range(0, x_width), dist_av)

    if save_location != "":
        plt.savefig(save_location)


def plot_confidence_distributions(
    names: List[str],
    data: List[DataFrame],
    save_location: str = "",
    combine_plots: bool = True,
):

    def extract_confidence_arr(df: DataFrame) -> List[float]:
        arr: List[PredictedKeyPoint] = df.map(
            lambda x: x.predicted_landmark
        ).values.reshape(-1)
        return [
            prediction.visibility for prediction in arr if not prediction.is_missing()
        ]

    arrs = [extract_confidence_arr(df) for df in data]

    if combine_plots:
        plot_histograms(
            names,
            arrs,
            save_location,
            title="Confidence distribution",
            legend_location="upper left",
            xlabel="confidence",
        )
    else:
        plot_histogram_grid(
            names=names,
            data=arrs,
            grid=(2, 2),
            save_location=save_location,
            title="Confidence distribution",
            xlabel="confidence",
        )


def plot_landmark_confidence_distributions(
    data: DataFrame, names: List[str], columns: List[List[str]], save_location: str = ""
):

    def extract_confidence_arr(included_cols: List[str]) -> List[float]:
        selection = data[included_cols]
        arr: List[PredictedKeyPoint] = selection.map(
            lambda x: x.predicted_landmark
        ).values.reshape(-1)
        return [
            prediction.visibility for prediction in arr if not prediction.is_missing()
        ]

    arrs = [extract_confidence_arr(col_set) for col_set in columns]

    plot_histogram_grid(
        names=names,
        data=arrs,
        grid=(len(names), 1),
        save_location=save_location,
        title="Confidence distribution",
        xlabel="confidence",
    )


def plot_distances_boxplot(
    title: str = "Comparison of relative distances",
    ylim: Tuple[int, int] | None = None,
    save_location: str | None = None,
    ylabel: str | None = None,
    estimations: List[Tuple[str, DataFrame]] = [],
):

    def get_name(estimation: Tuple[str, DataFrame]) -> str:
        return estimation[0]

    def get_values(estimation: Tuple[str, DataFrame]) -> List[float]:

        def filter(x: HpeEstimation) -> bool:
            return (
                x.can_predict
                and x.true_landmark is not None
                and not x.true_landmark.is_missing()
                and not x.predicted_landmark.is_missing()
            )

        values: List[HpeEstimation] = estimation[1].values.reshape((-1))
        return [value.get_relative_distance() for value in values if filter(value)]

    metrics = list(map(get_values, estimations))
    names = list(map(get_name, estimations))

    figure = plt.figure()

    if title:
        plt.title(title)
    plt.axhline(1, c="r", label="PCKh50 limit", linestyle="--", linewidth=1)
    plt.boxplot(metrics, tick_labels=names)
    if ylim:
        figure.axes[0].set_ylim(ylim)
    if ylabel:
        plt.ylabel(ylabel)

    if save_location:
        save_current_figure(save_location)


def plot_distances_swarmplot(
    names: List[str],
    estimations: List[DataFrame],
    xlim: Tuple[int, int] | None = None,
    save_location: str = "",
):

    res_data = []
    res_columns = ["Name", "Relative Distance"]

    def get_relative_distances(estimation: DataFrame) -> List[float]:

        def filter(x: HpeEstimation) -> bool:
            return (
                x.can_predict
                and x.true_landmark is not None
                and not x.true_landmark.is_missing()
                and not x.predicted_landmark.is_missing()
            )

        values: List[HpeEstimation] = estimation.values.reshape((-1))
        return [value.get_relative_distance() for value in values if filter(value)]

    for name, df in zip(names, estimations):
        for rel_dist in get_relative_distances(df):
            res_data.append([name, rel_dist])

    result = DataFrame(data=res_data, columns=res_columns)
    ax = stripplot(data=result, x="Relative Distance", y="Name")

    if xlim:
        ax.set_xlim((0, 10))

    if save_location:
        fig = ax.get_figure()
        fig.savefig(save_location, dpi=300, bbox_inches="tight")


def plot_absolute_distances_boxplot(
    ylim: Tuple[int, int] | None = None,
    save_location: str = "",
    *estimations: List[Tuple[str, DataFrame]],
):

    def get_name(estimation: Tuple[str, DataFrame]) -> str:
        return estimation[0]

    def get_values(estimation: Tuple[str, DataFrame]) -> List[float]:

        def filter(x: HpeEstimation) -> bool:
            return (
                x.can_predict
                and x.true_landmark is not None
                and not x.true_landmark.is_missing()
                and not x.predicted_landmark.is_missing()
            )

        values: List[HpeEstimation] = estimation[1].values.reshape((-1))
        return [value.get_distance() for value in values if filter(value)]

    metrics = list(map(get_values, estimations))
    names = list(map(get_name, estimations))

    figure = plt.figure()
    plt.title("Comparison of absolute distances")

    plt.boxplot(metrics, tick_labels=names)
    if ylim is not None:
        figure.axes[0].set_ylim(ylim)

    if save_location != "":
        plt.savefig(save_location)


def plot_precision_recall_curve(
    pnr: DataFrame,
    tight: bool = True,
    columns: List[MyLandmark] | None = None,
    save_location: str = "",
):
    """Plot the precision-recall curves for each class in the DataFrame.

    Args:
        pnr (DataFrame): DataFrame with
        tight (bool, optional): _description_. Defaults to True.
        columns (_type_, optional): _description_. Defaults to None.
    """
    show_legend = True

    if columns is None:
        columns = [landmark for landmark in MyLandmark]
        show_legend = False

    for landmark in columns:
        plt.plot(
            pnr[landmark.name].map(lambda x: x["r"])[1:-1],
            pnr[landmark.name].map(lambda x: x["p"])[1:-1],
            "o-",
            label=landmark.name,
        )
        plt.plot(
            pnr[landmark.name].map(lambda x: x["r"])[0],
            pnr[landmark.name].map(lambda x: x["p"])[0],
            "o",
            label=f"{landmark.name} @ 0 conf",
        )

    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    if show_legend:
        plt.legend(loc="upper left")

    if not tight:
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)

    if save_location:
        save_current_figure(save_location)


def plot_precision_and_recall(pnr: DataFrame, tight: bool = True):

    plt.figure()
    plt.subplot(121)
    for landmark in MyLandmark:
        plt.plot(pnr["CONFIDENCE"], pnr[landmark.name].map(lambda x: x["p"]))

    plt.title("Precision")
    plt.xlabel("Confidence")
    plt.ylabel("Precision")
    # plt.legend([landmark.name for landmark in MyLandmark])

    if not tight:
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)

    plt.subplot(122)
    for landmark in MyLandmark:
        plt.plot(pnr["CONFIDENCE"], pnr[landmark.name].map(lambda x: x["r"]))

    plt.title("Recall")
    plt.xlabel("Confidence")
    plt.ylabel("Recall")

    if not tight:
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)


def plot_AP_per_landmark(
    names: List[str], estimations: List[DataFrame], save_location: str = ""
):
    fig, axs = plt.subplots(2, 2, sharex="all", sharey="row")
    fig.set_figheight(15)
    axs: List[plt.Axes] = axs.reshape(-1)
    for df, name, ax in zip(estimations, names, axs):
        pnr = calc_precision_and_recall(df)
        APs = calc_average_precisions(pnr)
        APs = APs.drop("NECK")
        # APs = APs[APs.notnull()]
        labels = [label.replace("_", " ") for label in APs.index]

        ax.set_title(name)
        ax.barh(labels, APs)

    if save_location:
        save_current_figure(save_location)
