import base64
import zlib
import numpy as np


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

    return repr(img.shape[0]) + "," + repr(img.shape[1]) + "," + encoded.decode("utf-8")


def decompress_from_string(inp):
    """ Unpacks a string into a Numpy RGBA image array """
    vals = inp.split(",")
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
