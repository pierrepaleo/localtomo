#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from math import sqrt, pi, log
import matplotlib.pyplot as plt
from tomography import AstraToolbox, clipCircle
from phantoms import phantom, load_lena
from utils import *
from utils import call_imagej as ij
from optim import fista, fista_l1
try:
    from skimage.measure import structural_similarity # compare_ssim in newer versions
    __has_skimage__ = True
except ImportError:
    __has_skimage__ = False



# ------------------------------------------------------------------------------
# Available test cases are: "shepp-logan", "lena", "lena512", "pencil"
TEST_CASE = "lena512"
# Degree of verbosity for displaying images
DISPLAY = 4
# Number of acquired projections
nproj = 400
# Parameters for reconstruction
SIGMA = 4
SPACING = 6
# Other parameters
apodize_rather_than_clip = False
DO_STICK_KNOWNZONE = 0
# ------------------------------------------------------------------------------

# Volume and acquired sinogram
# -----------------------------
if TEST_CASE == "lena": # 256-Lena
    vol = clipCircle(bin2(load_lena()))
    CROP = 60
    OMEGA_R, OMEGA_C = 28, 0
    OMEGA_RADIUS = 14
elif TEST_CASE == "lena512": # 512-Lena
    vol = clipCircle(load_lena())
    vol += 100*ellipse_mask(vol.shape, -90, 190, 24, 18)
    OMEGA_R, OMEGA_C = 56, 0
    OMEGA_RADIUS = 35
    nproj = 800
    N2 = 520
    CROP = 120
elif TEST_CASE == "pencil":
    vol = np.load("data/pencil.npz")["data"]
    vol = bin2(bin2(vol))
    OMEGA_R, OMEGA_C = 5, -5
    OMEGA_RADIUS = 40
    nproj = 800
    N2 = 520
    CROP = 120
else: # default is Shepp-Logan
    vol = phantom(256).astype(np.float32)*250
    OMEGA_R, OMEGA_C = 40, 0
    OMEGA_RADIUS = 20
    CROP = 60
    N2 = 260


tomo0 = AstraToolbox(vol.shape[0], nproj)               # Full tomography geometry
sino0 = tomo0.proj(vol)                                 # Full sinogram (unknown in practice)
sino = np.copy(sino0[:, CROP:-CROP])                    # Acquired (truncated) sinogram
tomo = AstraToolbox(sino.shape[1], sino.shape[0])       # Local tomo geometry
rec_fbp_pad = tomo.fbp(sino/tomo.n_a*pi/2, padding=1)   # Padded FBP
# Do not take the exterior of padded FBP into account
if apodize_rather_than_clip:
    w = compute_window(rec_fbp_pad.shape[0], 50, 50);
    rec_fbp_pad *= w # transition zone should be  > gaussian width
else:
    rec_fbp_pad = clipCircle(rec_fbp_pad)
rec_fbp_pad_reproj = tomo.proj(rec_fbp_pad)


if DISPLAY >= 4: ims(rec_fbp_pad, legend="padded FBP and apodization")
if DISPLAY >= 1: visualize_localtomo_problem(vol, CROP, (OMEGA_R, OMEGA_C), OMEGA_RADIUS)

# Create the extended image
# --------------------------
img_extended = np.zeros((N2, N2))
N = rec_fbp_pad.shape[1]
Delta = (N2-N)//2
img_extended[Delta:-Delta, Delta:-Delta] = np.copy(rec_fbp_pad)
if DISPLAY >= 3: ims(img_extended, legend="Extended image, before placing Gaussians")
tomo = AstraToolbox(img_extended.shape[1], nproj)
# Define Omega for this bigger size
Omega = circle_mask(img_extended.shape, OMEGA_R, OMEGA_C, OMEGA_RADIUS)
Omega = Omega.astype(np.bool)
# Define Omega_g
Omega_g = retrieve_gaussian_components(np.ones_like(img_extended), SIGMA, SPACING, cmask=Omega) # TODO: check that there are no truncation effects !
Omega_g = Omega_g.astype(np.bool)
#~ ims(Omega_g, legend="Omega_g")


# First part of the algorithm: fit the known zone
# ------------------------------------------------
# Define the operator "G". cmask is important to avoid truncation artifacts
G = lambda g : put_gaussians_on_image(img_extended.shape, SIGMA, SPACING, g_coeffs=g, cmask=Omega)
Gadj = lambda x : retrieve_gaussian_components(x, SIGMA, SPACING, cmask=Omega)

volint = np.zeros_like(img_extended); volint[Delta:-Delta, Delta:-Delta] = clipCircle(vol[CROP:-CROP, CROP:-CROP])
err_known = (volint - img_extended)*Omega


_, g0 = fista_l1(err_known, G, Gadj, 250*1e-3, n_it=61)
if DISPLAY >= 3: ims(G(g0), legend="G(g0)")

# Second part of the algorithm: fit the "error" in the whole image, based on the known zone
# -------------------------------------------------------------------------------------------

# Circular mask for slice.
Me = clipCircle(np.ones_like(img_extended))
Me = np.ones_like(img_extended) # CLEARME
# Define the operator "G" with the mask "Me"
G = lambda g : put_gaussians_on_image(img_extended.shape, SIGMA, SPACING, g_coeffs=g, cmask=Me)
Gadj = lambda x : retrieve_gaussian_components(x, SIGMA, SPACING, cmask=Me)
#~ ims(Gadj(img_extended)) # Should be a blured-then-subsampled version of rec_fbp_pad


# Define the operators
Delta = (N2 - N)//2
def op_K(g):
    return crop_sino(tomo.proj(G(g)), Delta)

def op_Kadj(x):
    return Gadj(tomo.backproj(extend_sino(x, Delta)))

# Optimize
sino_to_fit = sino - rec_fbp_pad_reproj
_, gres = fista(sino_to_fit, op_K, op_Kadj, g0, Omega_g, n_it=1501) # <=======
if DISPLAY >= 4: ims(gres)
res = img_extended + G(gres)
if DISPLAY >= 2: ims(res, cmap="gray")

if DO_STICK_KNOWNZONE:
    res[Omega] = np.copy(volint[Omega])


# -------------
# Show results
# -------------

clipCircle2 = lambda x : clip_to_radius(x, Delta - SIGMA)


proposed = clipCircle2(res[Delta:-Delta, Delta:-Delta])
padded_fbp = clipCircle2(rec_fbp_pad)
proposed_diff = clipCircle2((res - volint)[Delta:-Delta, Delta:-Delta])
padded_fbp_diff = clipCircle2(rec_fbp_pad - volint[Delta:-Delta, Delta:-Delta])


# Print metrics
# --------------

ref1 = clipCircle2(volint[Delta:-Delta, Delta:-Delta])
print("-"*79)
print("Results for %s  sigma = %f  spacing = %f  radius = %d  N2 = %d" % (TEST_CASE, SIGMA, SPACING, OMEGA_RADIUS, N2))
if __has_skimage__:
    ssim1 = structural_similarity(ref1.astype(np.float64), padded_fbp.astype(np.float64))
    ssim2 = structural_similarity(ref1.astype(np.float64), proposed.astype(np.float64))
    print("Padded FBP: \t PSNR = %.3f \t SSIM = %f" % (compare_psnr(ref1, padded_fbp), ssim1))
    print("Proposed: \t PSNR = %.3f \t SSIM = %f" % (compare_psnr(ref1, proposed), ssim2))
else:
    print("Padded FBP: \t PSNR = %.3f" % compare_psnr(ref1, padded_fbp))
    print("Proposed: \t PSNR = %.3f" % compare_psnr(ref1, proposed))
print("-"*79)







L = proposed.shape[0]
if DISPLAY >= 4: visualize_lines(proposed, [L//2], [L//2])

# Differences
# -------------
# Mid. line:
plt.figure()
plt.plot(padded_fbp_diff[L//2, :])#, '--')
plt.plot(proposed_diff[L//2, :])
leg = plt.legend([r"$x_0 \, - \, x^\sharp$", r"$\hat{x} \, - \, x^\sharp$"]); leg.draggable();
plt.show()

# Mid. column:
plt.figure()
plt.plot(padded_fbp_diff[:, L//2])#, '--')
plt.plot(proposed_diff[:, L//2])
leg = plt.legend([r"$x_0 \, - \, x^\sharp$", r"$\hat{x} \, - \, x^\sharp$"]); leg.draggable();
plt.show()




# Profiles
# -------------
fbp_full = clipCircle2(tomo0.fbp(sino0/tomo0.n_a*pi/2)[CROP:-CROP, CROP:-CROP])


# Middle line
plt.figure()
l1 = plt.plot(padded_fbp[L//2,:], '-+')
l2 = plt.plot(proposed[L//2, :])
l3 = plt.plot(fbp_full[L//2, :], '--')
l4 = plt.plot(clipCircle2(volint[Delta:-Delta, Delta:-Delta])[L//2, :], '.-.')
for li in [l1, l2, l3, l4]: plt.setp(li, linewidth=2.0)
leg = plt.legend(["padded FBP", "Proposed", "full FBP", "True interior"], loc="best"); leg.draggable();
plt.show()

# Middle column

plt.figure()
l1 = plt.plot(padded_fbp[:, L//2], '-+')
l2 = plt.plot(proposed[:, L//2])
l3 = plt.plot(fbp_full[:, L//2], '--')
l4 = plt.plot(clipCircle2(volint[Delta:-Delta, Delta:-Delta])[:, L//2], '.-.')
for li in [l1, l2, l3, l4]: plt.setp(li, linewidth=2.0)
leg = plt.legend(["padded FBP", "Proposed", "full FBP", "True interior"], loc="best"); leg.draggable();
plt.show()






