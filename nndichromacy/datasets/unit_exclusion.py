from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import alphashape
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from dataport.bcm.experiment import ScanSet


def concave_hull_pca_selector(
    key,
    dataloaders,
    correlation_threshold=0.1,
    exclude="higher",
    concave_alpha=0.8,
    use_polygon=False,
    invert_scatter=False,
):
    """

    Args:
        key (dictionary): used to restrict the ScanSet.UnitInfo table to get the neuronal positions in cortex
        dataloaders (dict, nnfabrik dataloaders object): dictionary of dictionaries that contains, "train",
            "validation", and "test" keys
        correlation_threshold (float): Threshold, which units should be excluded based on the correlation with
            the units response with the reponse as projected on the first principle component
        exclude(str): can be "higher", or "lower". Excludes neurons higher, or lower of the corr threshold
        concave_alpha (float): Sets the alpha parameter of the concave hull fit
        use_polygon: Use the Polygon fitted on the concave hull to excluce units if True.
            Otherwise uses the correlation threshold exclusively.
        invert_scatter: If True, inverts the y-axis of the scatterplot, so the the origin is at the top left
            (as in plt.imshow).

    Returns:
        good_ids (np.array): a list of the unit_ids that dont have recording artefacts.
        fig (plt.figure): A figure with 3 subplots, showing the Signal*PC correlations, excluded units, and
            concave-hull/polygon fit.
    """

    data_key = list(dataloaders["train"].keys())[0]

    dat = dataloaders["train"][data_key].dataset
    if "probe" in dat.trial_info.tiers:
        responses = np.array(
            [
                dat[i].responses.cpu().numpy()
                for i in np.where(dat.trial_info.tiers == "probe")[0]
            ]
        )
    else:
        responses = []
        for b in dataloaders["train"][data_key]:
            responses.append(b.responses)
        responses = torch.vstack(responses).cpu().numpy()
    pipeline = Pipeline(
        [("scaling", StandardScaler()), ("pca", PCA(n_components=None))]
    )
    sklearn_transf = pipeline.fit_transform(responses)
    corr_1st_pc = []
    for i in range(responses.shape[1]):
        corr = np.corrcoef(sklearn_transf[:, 0], responses[:, i])
        corr_1st_pc.append(corr[0, 1])

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    ax1 = axs[0].hist(corr_1st_pc, np.linspace(-1, 1, 100))
    max_hist_hight = ax1[0].max()
    axs[0].plot(
        [correlation_threshold, correlation_threshold],
        [0, max_hist_hight],
        "--",
        color="gray",
        label="threshold",
    )
    axs[0].set_xlabel("correlation with PC1")
    axs[0].set_ylabel("N")
    axs[0].legend()

    neuron_ids = dataloaders["train"][data_key].dataset.neurons.unit_ids
    abnormal_ids = (
        neuron_ids[np.array(corr_1st_pc) > correlation_threshold]
        if exclude == "higher"
        else neuron_ids[np.array(corr_1st_pc) < correlation_threshold]
    )
    good_ids = (
        neuron_ids[np.array(corr_1st_pc) < correlation_threshold]
        if exclude == "higher"
        else neuron_ids[np.array(corr_1st_pc) > correlation_threshold]
    )

    neuron_id_key = [dict(unit_id=i) for i in neuron_ids]
    x_coords, y_coords = [], []
    x, y, unit_ids = (ScanSet.UnitInfo & key & neuron_id_key).fetch(
        "px_x",
        "px_y",
        "unit_id",
        order_by="unit_id",
    )

    assert (neuron_ids == unit_ids).sum() == len(
        unit_ids
    ), "Neuron ID mismatch between dataset and ScanSet.UnitInfo"
    x_coords.append(x)
    y_coords.append(y)
    x, y = (ScanSet.UnitInfo & key & [dict(unit_id=i) for i in abnormal_ids]).fetch(
        "px_x",
        "px_y",
        order_by="unit_id",
    )
    x_coords.append(x)
    y_coords.append(y)

    axs[1].scatter(x_coords[0], y_coords[0], c="g", edgecolor="w")
    axs[1].grid("on")
    axs[1].scatter(x_coords[1], y_coords[1], c="r", edgecolor="w")
    if invert_scatter:
        axs[1].invert_yaxis()

    # Create Polygon
    points = np.vstack([x_coords[1], y_coords[1]]).T
    alpha = concave_alpha * alphashape.optimizealpha(points)
    hull = alphashape.alphashape(points, alpha)
    hull_pts = np.array(hull.exterior)
    polygon = Polygon(hull_pts)

    in_polygon, outside_polygon, selected_units = [], [], []
    for i, coordinate in enumerate(zip(x_coords[0], y_coords[0])):
        point = Point(coordinate)
        if polygon.contains(point):
            in_polygon.append([coordinate])
        else:
            outside_polygon.append([coordinate])
            selected_units.append(i)

    in_polygon = np.stack(in_polygon)
    outside_polygon = np.stack(outside_polygon)
    axs[2].plot(*hull_pts.T, "k", linewidth=2)
    axs[2].scatter(*in_polygon.T, c="r", edgecolor="w")
    axs[2].scatter(*outside_polygon.T, c="g", edgecolor="w")
    if invert_scatter:
        axs[2].invert_yaxis()
    plt.close()

    good_ids = (
        good_ids if use_polygon is False else neuron_ids[np.array(selected_units)]
    )
    return good_ids, fig
