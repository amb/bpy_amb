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


def write_colors_bm(vcol_name, values, bm):
    color_layer = bm.loops.layers.color.get(vcol_name)
    if color_layer is None:
        color_layer = bm.loops.layers.color.new(vcol_name)

    for face in bm.faces:
        for loop in face.loops:
            loop[color_layer] = values[loop.vert.index]


def write_face_colors_bm(vcol_name, values, bm):
    color_layer = bm.loops.layers.color.get(vcol_name)
    if color_layer is None:
        color_layer = bm.loops.layers.color.new(vcol_name)

    for face in bm.faces:
        for loop in face.loops:
            loop[color_layer] = values[face.index]


def read_colors_bm(vcol_name, bm):
    color_layer = bm.loops.layers.color.get(vcol_name)
    if color_layer is None:
        return None
    values = np.empty(len(bm.verts))
    for face in bm.faces:
        for loop in face.loops:
            values[loop.vert.index] = loop[color_layer][0]
    return values
