import base64
import zlib
import numpy as np

# all images are assumed as Numpy RGBA arrays


def smooth(res, sval):
    krn = np.ones(sval) / sval
    f_krn = lambda m: np.convolve(m, krn, mode="same")
    res = np.apply_along_axis(f_krn, axis=1, arr=res)
    res = np.apply_along_axis(f_krn, axis=0, arr=res)
    return res


def linear2srgb(c):
    srgb = np.where(c < 0.0031308, c * 12.92, 1.055 * np.pow(c, 1.0 / 2.4) - 0.055)
    srgb[srgb > 1.0] = 1.0
    srgb[srgb < 0.0] = 0.0
    return srgb


def rgb2hsv(image):
    tmp = image[:, :, :3]
    r = tmp[:, :, 0]
    g = tmp[:, :, 1]
    b = tmp[:, :, 2]

    acmax = tmp.argmax(2)
    cmax = tmp.max(2)
    cmin = tmp.min(2)
    delta = cmax - cmin

    # R G B = 0 1 2
    # prevent zero division
    delta[delta == 0.0] = 1.0
    hr = np.where(acmax == 0, ((g - b) / delta) % 6.0, 0.0)
    hg = np.where(acmax == 1, (b - r) / delta + 2.0, 0.0)
    hb = np.where(acmax == 2, (r - g) / delta + 4.0, 0.0)

    H = np.where(cmax == cmin, 0.0, (hr + hg + hb) / 6.0)
    H[H < 0.0] += 1.0

    # L = (cmax + cmin) / 2.0
    # St = (cmax <= 0.0001) + (cmin >= 0.9999) + (L == 1.0) + (L == 0.0)
    # S = np.where(St, 0.0, (cmax - L) / np.minimum(L, 1.0 - L))

    L = cmax
    St = cmax <= 0.0001
    cmax[St] = 1.0
    S = np.where(St, 0.0, (cmax - cmin) / cmax)

    return np.dstack([H, S, L])


def compress_to_string(img):
    """ Compresses RGBA image array from Numpy into a string """
    if img.shape[0] > 32 or img.shape[1] > 32 or img.shape[0] < 8 or img.shape[1] < 8:
        print("Wrong image size. Required: width = [8, 32], height = [8, 32]")
        return None
    icon = []
    flatdim = img.shape[0] * img.shape[1]
    for v in image.reshape((flatdim, 4)):
        icon.append(int(round(v[0] * 15.0)))
        icon.append(int(round(v[1] * 15.0)))
        icon.append(int(round(v[2] * 7.0)))
        icon.append(int(round(v[3] * 3.0)))

    compressed = zlib.compress(bytes(icon), level=6)
    encoded = base64.b64encode(compressed)

    # don't use "," cause CSV, don't use any base64 characters either as separators
    return repr(img.shape[0]) + "^" + repr(img.shape[1]) + "^" + encoded.decode("utf-8")


def decompress_from_string(inp):
    """ Unpacks a string into a Numpy RGBA image array """
    vals = inp.split("^")
    w, h = int(vals[0]), int(vals[1])
    decoded = base64.b64decode(vals[2])
    uncompressed = zlib.decompress(decoded)
    values = []
    for v in range(len(uncompressed))[::4]:
        values.append(uncompressed[v + 0] / 15)
        values.append(uncompressed[v + 1] / 15)
        values.append(uncompressed[v + 2] / 7)
        values.append(uncompressed[v + 3] / 3)

    return np.array(values).reshape((w, h, 4))
