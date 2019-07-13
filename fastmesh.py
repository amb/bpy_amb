"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
import bpy


def write_fast(ve, qu, tr):
    me = bpy.data.meshes.new("testmesh")

    quadcount = len(qu)
    tricount = len(tr)

    me.vertices.add(count=len(ve))

    loopcount = quadcount * 4 + tricount * 3
    facecount = quadcount + tricount

    me.loops.add(loopcount)
    me.polygons.add(facecount)

    face_lengths = np.zeros(facecount, dtype=np.int)
    face_lengths[:tricount] = 3
    face_lengths[tricount:] = 4

    loops = np.concatenate((np.arange(tricount) * 3, np.arange(quadcount) * 4 + tricount * 3))

    v_out = np.concatenate((tr.ravel(), qu.ravel()))

    me.vertices.foreach_set("co", ve.ravel())
    me.polygons.foreach_set("loop_total", face_lengths)
    me.polygons.foreach_set("loop_start", loops)
    me.polygons.foreach_set("vertices", v_out)

    me.update(calc_edges=True)
    # me.validate(verbose=True)

    return me


def read_loops(mesh):
    loops = np.zeros((len(mesh.polygons)), dtype=np.int)
    mesh.polygons.foreach_get("loop_total", loops)
    return loops


def read_loop_starts(mesh):
    loops = np.zeros((len(mesh.polygons)), dtype=np.int)
    mesh.polygons.foreach_get("loop_start", loops)
    return loops


def read_polygon_verts(mesh):
    polys = np.zeros((len(mesh.polygons) * 4), dtype=np.uint32)
    mesh.polygons.foreach_get("vertices", polys)
    return polys


def read_verts(mesh):
    mverts_co = np.zeros((len(mesh.vertices) * 3), dtype=np.float)
    mesh.vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(mesh.vertices), 3))


def read_edges(mesh):
    fastedges = np.zeros((len(mesh.edges) * 2), dtype=np.int)  # [0.0, 0.0] * len(mesh.edges)
    mesh.edges.foreach_get("vertices", fastedges)
    return np.reshape(fastedges, (len(mesh.edges), 2))


def read_norms(mesh):
    mverts_no = np.zeros((len(mesh.vertices) * 3), dtype=np.float)
    mesh.vertices.foreach_get("normal", mverts_no)
    return np.reshape(mverts_no, (len(mesh.vertices), 3))


def write_verts(mesh, mverts_co):
    mesh.vertices.foreach_set("co", mverts_co.ravel())


def write_edges(mesh, fastedges):
    mesh.edges.foreach_set("vertices", fastedges.ravel())


def write_norms(mesh, mverts_no):
    mesh.vertices.foreach_set("normal", mverts_no.ravel())


def safe_bincount(data, weights, dts, conn):
    """
    for i, v in enumerate(data):
        dts[v] += weights[i]
        conn[v] += 1
    return (dts, conn)
    """
    bc = np.bincount(data, weights)
    dts[: len(bc)] += bc
    bc = np.bincount(data)
    conn[: len(bc)] += bc
    return (dts, conn)


def op_fraggle(mesh, thres, n):
    verts = read_verts(mesh)
    edges = read_edges(mesh)
    edge_a, edge_b = edges[:, 0], edges[:, 1]
    # for i in range(len(edge_a)):
    #    fastverts[edge_a[i]] += tvec[i]*thres
    #    fastverts[edge_b[i]] -= tvec[i]*thres
    for _ in range(n):
        tvec = verts[edge_b] - verts[edge_a]
        # tvlen = np.linalg.norm(tvec, axis=1)
        # tvec = (tvec.T / tvlen).T
        verts[edge_a] += tvec * thres
        verts[edge_b] -= tvec * thres
    write_verts(mesh, verts)


def op_smooth_mask(verts, edges, mask, n):
    # for e in edges:
    #    edge_c[e[0]] += 1
    #    edge_c[e[1]] += 1
    edge_c = np.zeros(len(verts), dtype=np.int32)
    e0_u, e0_c = np.unique(edges[:, 0], return_counts=True)
    e1_u, e1_c = np.unique(edges[:, 1], return_counts=True)
    edge_c[e0_u] += e0_c
    edge_c[e1_u] += e1_c
    edge_c = edge_c.T

    new_verts = np.copy(verts)
    new_verts = new_verts.T
    mt1 = (1.0 - mask).T
    mt0 = mask.T
    for _ in range(n):
        # <new vert location> = sum(<connected locations>) / <number of connected locations>
        locs = np.zeros((len(verts), 3), dtype=np.float64)
        nvt = new_verts.T
        np.add.at(locs, edges[:, 0], nvt[edges[:, 1]])
        np.add.at(locs, edges[:, 1], nvt[edges[:, 0]])

        locs = locs.T
        locs /= edge_c
        locs *= mt1

        new_verts *= mt0
        new_verts += locs

    return new_verts.T


def calc_curvature(fastverts, fastedges, fastnorms):
    """ Calculates curvature for specified mesh """
    edge_a, edge_b = fastedges[:, 0], fastedges[:, 1]

    tvec = fastverts[edge_b] - fastverts[edge_a]
    tvlen = np.linalg.norm(tvec, axis=1)

    # normalize vectors
    tvec = (tvec.T / tvlen).T

    # adjust the minimum of what is processed
    edgelength = tvlen * 100
    edgelength = np.where(edgelength < 1, 1.0, edgelength)

    vecsums = np.zeros(fastverts.shape[0], dtype=np.float)
    connections = np.zeros(fastverts.shape[0], dtype=np.int)

    # calculate normal differences to the edge vector in the first edge vertex
    totdot = (np.einsum("ij,ij->i", tvec, fastnorms[edge_a])) / edgelength
    safe_bincount(edge_a, totdot, vecsums, connections)

    # calculate normal differences to the edge vector  in the second edge vertex
    totdot = (np.einsum("ij,ij->i", -tvec, fastnorms[edge_b])) / edgelength
    safe_bincount(edge_b, totdot, vecsums, connections)

    return np.arccos(vecsums / connections) / np.pi


def divnp(a, b):
    if a.shape[1] == 1:
        return a / b
    else:
        for n in range(a.shape[1]):
            a[:, n] /= b
        return a


def mesh_smooth_filter_variable(data, fastverts, fastedges, iterations):
    """ Smooths variables in data [0, 1] over the mesh topology """

    if iterations <= 0:
        return data

    # vert indices of edges
    edge_a, edge_b = fastedges[:, 0], fastedges[:, 1]
    tvlen = np.linalg.norm(fastverts[edge_b] - fastverts[edge_a], axis=1)
    edgelength = np.where(tvlen < 1, 1.0, tvlen)

    data_sums = np.zeros(data.shape, dtype=np.float)
    connections = np.zeros(data.shape[0], dtype=np.float)

    # longer the edge distance to datapoint, less it has influence

    for _ in range(iterations):
        # step 1
        data_sums = np.zeros(data_sums.shape)
        connections = np.zeros(connections.shape)

        per_vert = data[edge_b] / edgelength
        safe_bincount(edge_a, per_vert, data_sums, connections)
        eb_smooth = data_sums / connections

        per_vert = eb_smooth[edge_a] / edgelength
        safe_bincount(edge_b, per_vert, data_sums, connections)

        new_data = data_sums / connections

        # step 2
        data_sums = np.zeros(data_sums.shape)
        connections = np.zeros(connections.shape)

        per_vert = data[edge_a] / edgelength
        safe_bincount(edge_b, per_vert, data_sums, connections)
        ea_smooth = data_sums / connections

        per_vert = ea_smooth[edge_b] / edgelength
        safe_bincount(edge_a, per_vert, data_sums, connections)

        new_data += data_sums / connections
        new_data /= 2.0

        data = new_data

    return new_data
