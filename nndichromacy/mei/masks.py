from scipy import ndimage
from skimage import morphology
import numpy as np
import torch


def compute_mei_mask_color(mei, zscore_threshold, closing_iters, gauss_sigma):
    if len(mei.shape) > 2:
        mei = mei[0, :2, ...].mean(0)
    if torch.is_tensor(mei):
        mei = mei.detach().cpu().numpy()
    # Normalize and threshold
    norm_mei = (mei - np.mean(mei)) / np.std(mei)
    thresholded = np.abs(norm_mei) > zscore_threshold
    # Remove small holes in the thresholded image and connect any stranding pixels
    closed = ndimage.binary_closing(thresholded, iterations=closing_iters)
    # Remove any remaining small objects
    labeled = morphology.label(closed, connectivity=2)
    most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
    oneobject = labeled == most_frequent
    # Create convex hull just to close any remaining holes and so it doesn't look weird
    hull = morphology.convex_hull_image(oneobject)
    # Smooth edges
    smoothed = ndimage.gaussian_filter(hull.astype(np.float32), sigma=gauss_sigma)
    return smoothed