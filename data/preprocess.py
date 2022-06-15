import time
import numpy as np
import pandas as pd
import sys
import scipy
import scipy.ndimage.filters
import matplotlib.pyplot as plt

# Start:
# takes two arguments after python constit2img.py .../input.h5 .../output.h5 n_events (for output-data)

#################################
# Settings
#################################

# n_const = n_constit*(px,py,pz,E,p_truthx,p_truthy,p_truthz,E_truth)
# n_constit are sorted by pT, highest pT first,
# n_constit = 200 maximum, but mostly filled with 0

# E or pT?
# problem with panda and safing data if n_crop to large!
#
# included (complib = "blosc", complevel=5) to to_hdf
# included mass

# input/output settings
signal_col = "is_signal_new"
mass_col = "mass"
n_constit = 200
n_crop = 40
batch_size = 1000
max_batches = float(sys.argv[3]) / batch_size
hdf_format = "table"
intensity = "pT"  # or "E" what to use for filling images


# image preprocessing options
Rotate, Flip, Crop, Norm = True, True, True, True  # maxmial preprocessing

# grid settings
xpixels = np.arange(-2.6, 2.6, 0.029) 
ypixels = np.arange(-np.pi, np.pi, 0.035)

# xpixels = np.arange(-np.pi, np.pi, 0.086)
# ypixels = np.arange(-np.pi, np.pi, 0.086)

# check if hottest constituent is close to centre afer shifting
n_warning = 0.7

##################################


# calculate pseudorapidity of pixel entries
def eta(pT, pz):
    small = 1e-10
    
    p = np.sqrt(np.square(pT) + np.square(pz))
    etas = 0.5 * (np.log(np.clip(p + pz, small, None)) - np.log(np.clip(p - pz, small, None)))

    # small_pT = np.abs(pT) < small
    # small_pz = np.abs(pz) < small
    # not_small = ~(small_pT | small_pz)

    # theta = np.arctan(pT[not_small] / pz[not_small])
    # theta[theta < 0] += np.pi

    # etas = np.zeros_like(pT)
    # etas[small_pz] = 0
    # etas[small_pT] = 1e-10
    # etas[not_small] = np.log(np.tan(theta / 2))
    return etas


# calculate phi (in range [-pi,pi]) of pixel entries
def phi(px, py):
    """phis are returned in rad., np.arctan(0,0)=0 -> zero constituents set to -np.pi"""
    phis = np.arctan2(py, px)
    # phis[phis < 0] += 2 * np.pi
    # phis[phis > 2 * np.pi] -= 2 * np.pi
    # phis = phis - np.pi
    return phis


# put eta-phi entries on grid
def orig_image(etas, phis, es):
    """gives always the larger value on the grid, eg. for xpixel = (0,1,2,3,..)  eta=1.3 -> xpixel=2,
    np.argmax gives position of first True value in array
    """
    z = np.zeros((etas.shape[0], len(xpixels), len(ypixels)))
    in_grid = ~(
        (etas < xpixels[0])
        | (etas > xpixels[-1])
        | (phis < ypixels[0])
        | (phis > ypixels[-1])
    )
    xcoords = np.argmax(etas[:, None, :] < xpixels[None, :, None], axis=1)
    ycoords = np.argmax(phis[:, None, :] < ypixels[None, :, None], axis=1)
    ncoords = np.repeat(np.arange(etas.shape[0])[:, None], etas.shape[1], axis=1)
    z[ncoords[in_grid], ycoords[in_grid], xcoords[in_grid]] = es[in_grid]

    return z


# put eta-phi entries on grid
def orig_image2(etas, phis, es):
    """Alternative version of orig_image: Gives the value on grid with minimal distance,
    eg. for xpixel = (0,1,2,3,..) eta=1.3 -> xpixel=1, eta=1.6 ->xpixel=2
    """
    z = np.zeros((etas.shape[0], len(xpixels), len(ypixels)))
    in_grid = ~(
        (etas < xpixels[0])
        | (etas > xpixels[-1])
        | (phis < ypixels[0])
        | (phis > ypixels[-1])
    )
    xcoords = np.argmin(np.abs(etas[:, None, :] - xpixels[None, :, None]), axis=1)
    ycoords = np.argmin(np.abs(phis[:, None, :] - ypixels[None, :, None]), axis=1)
    ncoords = np.repeat(np.arange(etas.shape[0])[:, None], etas.shape[1], axis=1)
    z[ncoords[in_grid], ycoords[in_grid], xcoords[in_grid]] = es[in_grid]

    return z


def print_time(msg):
    print("[%8.2f] %s" % (time.time() - time_start, msg))


def img_mom(x, y, weights, x_power, y_power):
    """returns image momenta for centroid and principal axis"""
    return ((x ** x_power) * (y ** y_power) * weights).sum()


def preprocessing(x, y, weights):
    """(x,y) are the coordinates and weights the corresponding values, shifts
    centroid to origin, rotates image, so that principal axis is vertical,
    flips image, so that most weights lay in (x<0, y>0)-plane.
    Method for calculating principal axis (similar to tensor of inertia):
    https://en.wikipedia.org/wiki/Image_moment
    here: y=phi, phi has modulo 2*np.pi but it's not been taken care of hear,
    so possible issues with calculating the centroid
    -> pre-shifting of events outside of this function solves the problem
    for iamge-data with Delta_phi < 2*np.pi
    """

    # shift
    x_centroid = img_mom(x, y, weights, 1, 0) / weights.sum()
    y_centroid = img_mom(x, y, weights, 0, 1) / weights.sum()
    x = x - x_centroid
    y = y - y_centroid

    # check if shifting worked, there can be problems with modulo variables like phi (y)
    # x and y are sorted after highest weight, 0-comp. gives hottest event
    # for Jet-like Images Centroid should be close to hottest constituen (pT-sorted arrays)
    global n_shift_phi
    global n_shift_eta
    if np.abs(x[0]) > n_warning:
        n_shift_eta += 1
    if np.abs(y[0]) > n_warning:
        n_shift_phi += 1

    if Rotate:
        # covariant matrix, eigenvectors corr. to principal axis
        u11 = img_mom(x, y, weights, 1, 1) / weights.sum()
        u20 = img_mom(x, y, weights, 2, 0) / weights.sum()
        u02 = img_mom(x, y, weights, 0, 2) / weights.sum()
        cov = np.array([[u20, u11], [u11, u02]])

        # Eigenvalues and eigenvectors of covariant matrix
        evals, evecs = np.linalg.eig(cov)

        # sorts the eigenvalues, v1, [::-1] turns array around,
        sort_indices = np.argsort(evals)[::-1]
        e_1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        e_2 = evecs[:, sort_indices[1]]

        # theta to x_asix, arctan2 gives correct angle
        theta = np.arctan2(e_1[0], e_1[1]) # is this correct here? was e_1[0], e_1[1]

        """
        print('e_1 (x,y): ' + str(e_1[0]) + ' ' + str(e_1[1]))
        print('e_2 (x,y): ' + str(e_2[0]) + ' ' + str(e_2[1]))
        print('Evals: ' +str(evals))
        print('theta: ' + str(theta/(np.pi*2)*360))
        """

        # rotation, so that princple axis is vertical
        # anti-clockwise rotation matrix
        rotation = np.matrix(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        transformed_mat = rotation * np.stack([x, y])
        x_rot, y_rot = transformed_mat.A
    else:
        x_rot, y_rot = x, y

    # flipping
    n_flips = 0
    if Flip:
        if weights[x_rot < 0.0].sum() < weights[x_rot > 0.0].sum():
            x_rot = -x_rot
            n_flips += 1
        if weights[y_rot < 0.0].sum() > weights[y_rot > 0.0].sum():
            y_rot = -y_rot
            n_flips += 1

    # print('number of flips: ' + str(n_flips))

    return x_rot, y_rot


def mass(E, px, py, pz):
    mass = np.sqrt(np.maximum(0.0, E ** 2 - px ** 2 - py ** 2 - pz ** 2))
    return mass


def process_batch(start_id):
    print_time(
        "Loading input file (events %i to %i)" % (start_id, start_id + batch_size)
    )
    df = pd.read_hdf(
        sys.argv[1], "table", start=start_id, stop=start_id + batch_size
    )
    if df.shape[0] == 0:
        return False

    print_time("Extracting 4-vectors")
    feat_list = ["E", "PX", "PY", "PZ"]
    cols = [
        "{0}_{1}".format(feature, constit)
        for feature in feat_list
        for constit in range(n_constit)
    ]
    vec4 = np.expand_dims(df[cols], axis=-1).reshape(-1, len(feat_list), n_constit)
    isig = df[signal_col].to_numpy()

    print_time("Calculating pT")
    E = vec4[:, 0, :]
    pxs = vec4[:, 1, :]
    pys = vec4[:, 2, :]
    pzs = vec4[:, 3, :]
    pT = np.sqrt(pxs ** 2 + pys ** 2)

    print_time("Calculating eta")
    etas = eta(pT, pzs)
    print_time("Calculating phi")
    phis = phi(pxs, pys)

    print_time("Calculating the mass")
    E_tot = E.sum(axis=1)
    px_tot = pxs.sum(axis=1)
    py_tot = pys.sum(axis=1)
    pz_tot = pzs.sum(axis=1)
    j_mass = mass(E_tot, px_tot, py_tot, pz_tot)

    # pre-shifting of phi
    phis = (phis.T - phis[:, 0]).T
    phis[phis < -np.pi] += 2 * np.pi
    phis[phis > np.pi] -= 2 * np.pi

    print_time("Preprocessing")
    if intensity == "pT":
        weights = pT
    elif intensity == "E":
        weights = E

    for i in np.arange(0, batch_size):
        etas[i, :], phis[i, :] = preprocessing(etas[i, :], phis[i, :], weights[i, :])

    # using pT instead of energy E
    print_time("Creating images")
    z_ori = orig_image2(etas, phis, pT)
    print_time("Crop and normalize")
    z_new = np.zeros((z_ori.shape[0], n_crop, n_crop))

    for i in range(z_ori.shape[0]):
        if Crop:
            Npix = z_ori[i, :, :].shape
            z_new[i, :, :] = z_ori[i,Npix[0] // 2 - n_crop // 2 : Npix[0] // 2 + n_crop // 2, Npix[1] // 2 - n_crop // 2 : Npix[1] // 2 + n_crop // 2]
        else:
            z_new = z_ori
        if Norm:
            z_sum = z_new[i, :, :].sum()
            if z_sum != 0.0:
                z_new[i, :, :] = z_new[i, :, :] / z_sum
    
    print_time("Reshaping output")
    z_out = z_new.reshape((z_new.shape[0], -1))

    print_time("Creating output dataframe")
    out_cols = (
        ["img_{0}".format(i) for i in range(z_new.shape[1] * z_new.shape[2])]
        + [signal_col]
        + [mass_col]
    )
    df_out = pd.DataFrame(
        data=np.concatenate((z_out, isig[:, np.newaxis], j_mass[:, np.newaxis]), axis=1),
        index=np.arange(start_id, start_id + batch_size),
        columns=out_cols,
    )
    print_time("Writing output file")
    df_out.to_hdf(
        sys.argv[2],
        "table",
        append=(start_id != 0),
        format="table",
        complib="blosc",
        complevel=5,
    )

    return True


# --------------------------------------------------------------
# print Settings
print("----------------------------------------------------")
print("number of x_pixel (phi): " + str(len(xpixels)))
print("number of y_pixel (eta): " + str(len(ypixels)))
assert len(xpixels) == len(ypixels)  # need a square grid
xgrid, ygrid = np.meshgrid(xpixels, ypixels)

print("x_min, x_max (phi): {0:4.2f} {1:4.2f}".format(xpixels.min(), xpixels.max()))
print("y_min, y_max (eta): {0:4.2f} {1:4.2f}".format(ypixels.min(), ypixels.max()))

print("x: step_size (phi): {0:5.3f}".format(xpixels[1] - xpixels[0]))
print("y: step_size (eta): {0:5.3f}".format(ypixels[1] - ypixels[0]))
print("signal_col  = " + signal_col)
print("n_constit   = " + str(n_constit))
print("n_crop      = " + str(n_crop))
print("batch_size  = " + str(batch_size))
print("number of events = " + str(sys.argv[3]))
print("hdf_format  = " + hdf_format)
print(
    "Rotate, Flip, Crop, Norm  = "
    + str(Rotate)
    + " "
    + str(Flip)
    + " "
    + str(Crop)
    + " "
    + str(Norm)
)
print(
    "Output-images: {0}x{0} pixel with range: phi = {1:4.2f}, eta = {2:4.2f}".format(
        n_crop, n_crop * (xpixels[1] - xpixels[0]), n_crop * (ypixels[1] - ypixels[0])
    )
)
print("For intensity of pixel used: " + intensity)
print("---------------------------------------------------")

assert len(xpixels) == len(ypixels)  # need a square grid
xgrid, ygrid = np.meshgrid(xpixels, ypixels)
assert (
    float(sys.argv[3]) >= batch_size
)  # have to modify batch_size to process small number of events

time_start = time.time()
start_id = 0
n_shift_phi = 0  # number of times shifting failed
n_shift_eta = 0

import sys
while process_batch(start_id):
    start_id += batch_size
    if start_id // batch_size == max_batches:
        break

print_time("Shuffling samples")
df = pd.read_hdf(sys.argv[2], "table")
df = df.iloc[np.random.permutation(len(df))]
df.to_hdf(sys.argv[2], "table", format="table", complib="blosc", complevel=5)

if n_shift_eta != 0:
    print_time("Warning: hottest constituent is supposed to be close to origin.")
    print_time(
        "Number of times eta of hottest const. was not close to origin: "
        + str(n_shift_eta)
    )
if n_shift_phi != 0:
    print_time("Warning: hottest constituent is supposed to be close to origin.")
    print_time(
        "Number of times phi of hottest const. was not close to origin: "
        + str(n_shift_phi)
    )

print_time("Finished")
