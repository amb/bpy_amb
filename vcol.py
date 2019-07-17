import numpy as np


def write_colors(vcol_name, values, mesh):
    """ init values with np.ones((len(values), 4)) """

    if vcol_name not in mesh.vertex_colors:
        mesh.vertex_colors.new(name=vcol_name)

    color_layer = mesh.vertex_colors[vcol_name]
    mesh.vertex_colors[vcol_name].active = True

    print("writing vertex colors for array:", values.shape)

    # write vertex colors
    mloops = np.zeros((len(mesh.loops)), dtype=np.int)
    mesh.loops.foreach_get("vertex_index", mloops)
    color_layer.data.foreach_set("color", values[mloops].flatten())
