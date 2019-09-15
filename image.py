import base64
import zlib
import numpy as np


def img_compress(img):
    """ Compresses RGBA image array from Numpy into a string """
    if img.shape[0] > 32 or img.shape[1] > 32 or img.shape[0] < 8 or img.shape[1] < 8:
        print("Wrong image size.")
        return None
    icon = []
    flatdim = img.shape[0] * img.shape[1]
    for v in image.reshape((flatdim, 4)):
        icon.append(int(round(v[0] * 31.0)))
        icon.append(int(round(v[1] * 31.0)))
        icon.append(int(round(v[2] * 31.0)))
        icon.append(int(round(v[3] * 31.0)))

    compressed = zlib.compress(bytes(icon), level=6)
    encoded = base64.b64encode(compressed)
    # print(len(icon), "=>", len(compressed))

    return repr(img.shape[0]) + "," + repr(img.shape[1]) + "," + encoded.decode("utf-8")


def img_decompress(inp):
    """ Unpacks a string into a Numpy RGBA image array """
    vals = inp.split(",")
    w, h = int(vals[0]), int(vals[1])
    decoded = base64.b64decode(vals[2])
    uncompressed = zlib.decompress(decoded)
    values = []
    for v in uncompressed:
        values.append(v / 31)

    return np.array(values).reshape((w, h, 4))
