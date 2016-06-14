#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2016, European Synchrotron Radiation Facility
# Main author: Pierre Paleo <pierre.paleo@esrf.fr>
#
# ----------------------
import numpy as np
from math import sqrt, pi, log, ceil
import matplotlib.pyplot as plt
# ----------------------




# Linalg utils
# -------------

def dot(a, b):
    return a.ravel().dot(b.ravel())

def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())


def norm1(mat):
    return np.sum(np.abs(mat))






# Gaussian basis utils
# ---------------------

def gaussian1D(sigma):
    ksize = int(ceil(8 * sigma + 1))
    if (ksize % 2 == 0): ksize += 1
    t = np.arange(ksize) - (ksize - 1.0) / 2.0
    g = np.exp(-(t / sigma) ** 2 / 2.0).astype('f')
    g /= g.sum(dtype='f')
    return g



def generate_coords(img_shp, center=None):
    l_r, l_c = float(img_shp[0]), float(img_shp[1])
    R, C = np.mgrid[:l_r, :l_c]
    if center is None:
        center0, center1 = l_r / 2., l_c / 2.
    else:
        center0, center1 = center
    R += 0.5 - center0
    C += 0.5 - center1
    return R, C


# much slower if convolve2 is used => use fftconvolve
#~ from scipy.signal import convolve2d as conv
from scipy.signal import fftconvolve as conv

def put_gaussians_on_image(shp, sigma, spacing, g_coeffs=None, cmask=None, debug=False):
    g1 = gaussian1D(sigma)
    g1 /= g1.max()
    l = g1.shape[0]
    g2 = np.outer(g1, g1)
    res = np.zeros(shp)
    res[l//2:-(l//2):spacing, l//2:-(l//2):spacing] = 1 if g_coeffs is None else g_coeffs
    if cmask is not None: res *= cmask # do it before convol, otherwise truncation artefacts !
    if debug: print("%d Gaussians used for %s" % (res[res!=0].sum(), str(shp)))
    res = conv(res, g2, mode="same")
    return res


def retrieve_gaussian_components(img, sigma, spacing, cmask=None):
    g1 = gaussian1D(sigma)
    g1 /= g1.max()
    l = g1.shape[0]
    g2 = np.outer(g1, g1)
    img2 = conv(img, g2, mode="same")
    if cmask is not None: img2 *= cmask
    coeffs = img2[l//2:-(l//2):spacing, l//2:-(l//2):spacing] # view !
    return coeffs



# Operators
# ----------

def clip_to_radius(img, radius=None):
    if radius is None: radius = (shp[1] - 1)/2
    res = np.copy(img)
    R, C = generate_coords(img.shape)
    res[R**2 + C**2 > radius**2] = 0
    return res


def extend_sino(img, crop):
    res = np.zeros((img.shape[0], img.shape[1] + 2*crop))
    res[:, crop:-crop] = np.copy(img)
    return res


def extend_image(img, crop):
    res = np.zeros((img.shape[0]+2*crop, img.shape[1] + 2*crop))
    res[crop:-crop, crop:-crop] = np.copy(img)
    return res


def crop_sino(s, crop):
    return np.copy(s[:, crop:-crop])

def crop_image(img, crop):
    return np.copy(img[crop:-crop, crop:-crop])


def circle_mask(img_shape, r, c, radius):
    R, C = generate_coords(img_shape)
    mask = R**2 + C**2 < radius**2
    return (R - r)**2 + (C - c)**2 < radius**2


def ellipse_mask(img_shape, r, c, a, b):
    """
    Generates an ellipse mask on image of shape "img_shape".
    The resulting ellipse will be 2*a in axis 0, and 2*b in axis 1.
    """
    R, C = generate_coords(img_shape)
    mask = np.zeros(img_shape)
    mask[(R - r)**2/a**2 + (C - c)**2/b**2 <= 1.] = 1
    return mask


def bin2(img):
    """
    Perform a 2-binning of an image, i.e an averaging followed by a 2-subsampling.
    """
    Nr, Nc = img.shape
    if Nr != Nc: raise ValueError("Image is not square")
    if (Nr & 1): raise ValueError("Image has odd dimensions")
    return img.reshape(Nr//2, 2, Nc//2, 2).sum(axis=-1).sum(axis=1)*0.25


# Misc.
# ------


def compute_window(x, crop1, crop2, MARGIN=0, Pow=1):
    #
    #    __margin__
    #   /          \
    #  c1          c2


    # 1D Function
    # -----------
    w = np.ones(x)
    c1 = crop1 + MARGIN
    c2 = crop2+MARGIN
    w[:c1] = np.hanning(2*c1)[:c1]**Pow
    w[-c2:] = np.hanning(2*c2)[-c2:]**Pow

    #~ import matplotlib.pyplot as plt
    #~ plt.figure()
    #~ plt.plot(w)
    #~ plt.plot(1-w)
    #~ plt.show()

    # Turn into 2D : distance to the center
    # -------------------------------------
    R, C = generate_coords((x, x))
    rmax = int(R.max()-0.5)
    M = np.int32(rmax-np.sqrt((R+0.5)**2+(C+0.5)**2));
    M[M<0] = 0
    return w[M]


def apodize1d(N, iradius, transition):
    """
    shp : image shape
    iradius : radius of the interior
    transition: length of transition at each side

    legend:
        ^^^^^^----_____(I)_____----^^^^^^

    (I) : interior region (2*iradius+1) => zeros
    ----: transition region => transition
    ^^^^: exterior => zeros
    """
    t = transition - 1 # !
    w1 = np.hanning(2*t+1)
    tr1 = w1[:transition]
    tr2 = w1[-transition:]
    # 1D mask for int + transition
    w = np.ones(N)
    e1 = (N - 2*transition - 2*iradius)/2
    w[:e1] = 1.
    w[e1:e1+transition] = tr2
    w[e1+transition:e1+transition+2*iradius] = 0
    w[-e1-transition:-e1] = tr1
    w[-e1:] = 1.
    return w


def apodize(shp, iradius, transition):
    R, C = generate_coords(shp)
    w = apodize1d(shp[0], iradius, transition)
    rmax = int(R.max()-0.5)
    M = np.int32(rmax-np.sqrt(R**2+C**2));
    M[M<0] = 0
    return w[M]






# Metrics. Inspired from skimage.measure
# ---------------------------------------


def compare_mse(im1, im2):
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true0, im_test0):
    im_true = im_true0.astype(np.float64)
    im_test = im_test0.astype(np.float64)
    im_true = convert_range(im_true, out_range=(-1., 1.))
    im_test = convert_range(im_test, out_range=(-1., 1.))
    dmin, dmax = -1., 1.
    dynamic_range = dmax - dmin
    err = compare_mse(im_true, im_test)
    return 10 * np.log10((dynamic_range ** 2) / err)


def convert_range(arr, in_range=None, out_range=None):
    if in_range is None: in_range = (1.0*arr.min(), 1.0*arr.max())
    if out_range is None: out_range = (0., 1.)
    in_min, in_max = in_range
    out_min, out_max = out_range
    return (out_max - out_min)/(in_max - in_min)*(arr - in_min) + out_min





# Visualization utils
# --------------------

def visualize_localtomo_problem(vol, crop_width, (omega_r, omega_c), omega_radius):

    #~ import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    roi_radius = vol.shape[0]//2-crop_width

    Omega = Ellipse(xy=(vol.shape[1]//2+omega_c, vol.shape[0]//2+omega_r),
        width=2*omega_radius, height=2*omega_radius,
        color="green", linewidth=2.0, fill=False)
    ROI = plt.Circle((vol.shape[1]//2, vol.shape[0]//2),
        radius=roi_radius, color='blue', linewidth=2.0, fill=False)

    fig, ax = plt.subplots()
    plt.imshow(vol, cmap="gray", interpolation="nearest")
    fig.gca().add_artist(Omega)
    fig.gca().add_artist(ROI)
    #~ fig.gca().axes.get_xaxis().set_visible(False)
    #~ fig.gca().axes.get_yaxis().set_visible(False)


    import matplotlib.lines as lines
    Nr, Nc = vol.shape
    positions_x = [Nc//2]
    positions_y = [Nc//2]
    lines_x = []
    lines_y = []
    for pos in positions_x: lines_x.append([(crop_width, pos), (Nc-crop_width, pos)])
    for pos in list(positions_y): lines_y.append([(pos, crop_width), (pos, Nr-crop_width)])

    for line1, line2, ls in zip(lines_x, lines_y, ["--", "-.", "-"]):
        (line1_xs, line1_ys) = zip(*line1)
        (line2_xs, line2_ys) = zip(*line2)
        ax.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='orange', linestyle=ls))
        ax.add_line(lines.Line2D(line2_xs, line2_ys, linewidth=2, color='purple', linestyle=ls))

    plt.show()



def visualize_lines(img, positions_x, positions_y):
    import matplotlib.lines as lines
    fig, ax = plt.subplots()
    plt.imshow(img, cmap="gray", interpolation="nearest")
    Nr, Nc = img.shape
    if not(hasattr(positions_x, "__iter__")):  positions_x = [positions_x]
    if not(hasattr(positions_y, "__iter__")):  positions_y = [positions_y]

    lines_x = []
    lines_y = []
    for pos in positions_x: lines_x.append([(0, pos), (Nc, pos)])
    for pos in list(positions_y): lines_y.append([(pos, 0), (pos, Nr)])

    for line1, line2, ls in zip(lines_x, lines_y, ["-", "-.", "--"]):
        (line1_xs, line1_ys) = zip(*line1)
        (line2_xs, line2_ys) = zip(*line2)
        ax.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue', linestyle=ls))
        ax.add_line(lines.Line2D(line2_xs, line2_ys, linewidth=2, color='green', linestyle=ls))
    plt.show()





def ims(img, cmap=None, legend=None, nocbar=False, share=True):
    """
    image visualization utility.

    img: 2D numpy.ndarray, or list of 2D numpy.ndarray
        image or list of images
    cmap: string
        Optionnal, name of the colorbar to use.
    legend: string, or list of string
        legend under each image
    nocbar: bool
        if True, no colorbar are displayed. Default is False
    share: bool
        if True, the axis are shared between the images, so that zooming in one image
        will zoom in all the corresponding regions of the other images. Default is True
    """
    try:
        _ = img.shape
        nimg = 1
    except AttributeError:
        nimg = len(img)
    #
    if (nimg <= 2): shp = (1,2)
    elif (nimg <= 4): shp = (2,2)
    elif (nimg <= 6): shp = (2,3)
    elif (nimg <= 9): shp = (3,3)
    else: raise ValueError("too many images")
    #
    plt.figure()
    for i in range(nimg):
        curr = list(shp)
        curr.append(i+1)
        curr = tuple(curr)
        if nimg > 1:
            if i == 0: ax0 = plt.subplot(*curr)
            else:
                if share: plt.subplot(*curr, sharex=ax0, sharey=ax0)
                else: plt.subplot(*curr)
            im = img[i]
            if legend: leg = legend[i]
        else:
            im = img
            if legend: leg = legend
        if cmap:
            plt.imshow(im, cmap=cmap, interpolation="nearest")
        else:
            plt.imshow(im, interpolation="nearest")
        if legend: plt.xlabel(leg)
        if nocbar is False: plt.colorbar()

    plt.show()





import random
import string
import os
import subprocess
import tifffile

def _imagej_open(fname):
    # One file
    if isinstance(fname, str):
        cmd = ['imagej', fname]
    # Multiple files
    if isinstance(fname, list):
        cmd = ['imagej'] + fname
    FNULL = open(os.devnull, 'w')
    process = subprocess.Popen(cmd, stdout=FNULL, stderr=FNULL)
    FNULL.close();
    process.wait()
    return process.returncode


def call_imagej(obj):
    # Open file(s)
    if isinstance(obj, str) or (isinstance(obj, list) and isinstance(obj[0], str)):
        return _imagej_open(obj)
    # Open numpy array(s)
    elif isinstance(obj, np.ndarray) or (isinstance(obj, list) and isinstance(obj[0], np.ndarray)):
        if isinstance(obj, np.ndarray):
            data = obj
            if data.dtype == np.float64: data = data.astype(np.float32)
            fname = '/tmp/' + _randomword(10) + '.tif'
            tifffile.imsave(fname, data)
            return _imagej_open(fname)
        else:
            fname_list = []
            for i, data in enumerate(obj):
                if data.dtype == np.float64: data = data.astype(np.float32)
                fname = '/tmp/' + _randomword(10) + str("_%d.tif" % i)
                fname_list.append(fname)
                tifffile.imsave(fname, data)
            return _imagej_open(fname_list)

    else:
        raise ValueError('Please enter a file name or a numpy array')


def _randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))








