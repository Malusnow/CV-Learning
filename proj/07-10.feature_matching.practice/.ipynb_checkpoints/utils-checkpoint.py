from itertools import cycle

from IPython.display import clear_output, display
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


def load_bear(data_dir):
    r"""Loads two images of a plush bear captured from slightly different viewpoints.

    Returns
    -------
    imgs : list of np.ndarray
        of shape [2, h, w, 3] and type uint8.
    """
    img_paths = [f'{data_dir}/plush_bear/0011.png', f'{data_dir}/plush_bear/0003.png']
    img_slice_tuples = [(slice(750, -1), slice(500, 1500)), (slice(600, -150), slice(550, 1550))]
    imgs = []
    for img, img_slices in zip(img_paths, img_slice_tuples):
        img = Image.open(img)
        img = np.array(img)
        img = img[img_slices]
        imgs.append(img)
    return imgs


def load_chessboard(data_dir):
    r"""Loads two images of a chessboard captured from significantly different viewpoints.

    Returns
    -------
    imgs : list of np.ndarray
        of shape [2, h, w, 3] and type uint8.
    """
    img_paths = [f'{data_dir}/small_wooden_chessboard/0003.png', f'{data_dir}/small_wooden_chessboard/0093.png']
    img_slice_tuples = [(slice(950, -250), slice(850, 1500)), (slice(750, -500), slice(850, 1500))]
    imgs = []
    for img, img_slices in zip(img_paths, img_slice_tuples):
        img = Image.open(img)
        img = np.array(img)
        img = img[img_slices]
        imgs.append(img)
    return imgs


def load_fox(data_dir):
    r"""Loads an image of a fox figurine and makes a rotated and scaled copy of it.

    Returns
    -------
    imgs : list of np.ndarray
        of shape [2, h, w, 3] and type uint8.
    """
    img_path = f'{data_dir}/white_fox_figurine/0011.png'
    img_slices_l = (slice(1150, -380), slice(900, 1250))
    img_slices_r = (slice(1150-30, -380), slice(900-40, 1250))

    img = Image.open(img_path)
    img = np.array(img)
    img_l = img[img_slices_l]
    
    img = img[img_slices_r]
    img = Image.fromarray(img)
    img = img.rotate(30, resample=Image.Resampling.BICUBIC)
    w, h = img.size
    s = .7
    img = img.resize([round(w * s), round(h * s)], resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    img_r = img
    
    return [img_l, img_r]


def draw_matches(imgs, kpts, matches, figsize=(10, 8), match_color=None):
    r"""Draws left-to-right matches for a pair of images.

    Parameters
    ----------
    imgs : list of np.ndarray
        (img_l, img_r) each of shape [height, width], grayscale uint8 images.
    kpts : list of lists of cv.KeyPoint
        (kpts_l, kpts_r), all keypoints detected on each of the images.
    matches : list of cv.DMatch
        Left-to-right matches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    img_l, img_r = imgs
    kpts_l, kpts_r = kpts    
    img_matches = cv.drawMatches(img_l, kpts_l, img_r, kpts_r, matches, None, matchColor=match_color, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_matches)
    ax.set_axis_off()
    plt.tight_layout()
    plt.close()
    return fig


def draw_matches_rtol(imgs_ltor, kpts_ltor, matches_rtol, figsize=(10, 8), match_color=None):
    r"""Draws right-to-left matches for a pair of images.

    Parameters
    ----------
    imgs_ltor : list of np.ndarray
        (img_l, img_r) each of shape [height, width], grayscale uint8 images.
    kpts_ltor : list of lists of cv.KeyPoint
        (kpts_l, kpts_r), all keypoints detected on each of the images.
    matches_rtol : list of cv.DMatch
        Right-to-left matches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    matches_ltor = list(map(reverse_match, matches_rtol))
    fig = draw_matches(imgs_ltor, kpts_ltor, matches_ltor, figsize, match_color)
    return fig


def reverse_match(match_ltor):
    r"""Creates a fake right-to-left match from a real left-to-right match (or vice versa) linking the same pair of points in the respective different direction.

    Parameters
    ----------
    match_ltor : cv.DMatch

    Returns
    -------
    match_rtol : cv.DMatch
    """
    match_rtol = cv.DMatch(match_ltor.trainIdx, match_ltor.queryIdx, match_ltor.distance)
    return match_rtol


def show_fig_switcher(*figs):
    r"""Shows a widget for switching between different figures.
    
    Parameters
    ----------
    figs : list of matplotlib.figure.Figure
    """
    switch_btn = widgets.Button(description='Switch figure')
    fig_area = widgets.Output()
    figs = cycle(figs)
    
    def on_click(*args):
        with fig_area:
            clear_output(wait=True)
            display(next(figs))
    
    switch_btn.on_click(on_click)
    
    display(switch_btn, fig_area)
    with fig_area:
        display(next(figs))
