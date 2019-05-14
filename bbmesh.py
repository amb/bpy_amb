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
import bmesh  # pylint: disable=import-error
from . import fastmesh as afm


def read_bmesh(bmesh):
    bmesh.verts.ensure_lookup_table()
    bmesh.faces.ensure_lookup_table()

    verts = [(i.co[0], i.co[1], i.co[2]) for i in bmesh.verts]
    qu, tr = [], []
    for f in bmesh.faces:
        if len(f.verts) == 4:
            qu.append([])
            for v in f.verts:
                qu[-1].append(v.index)
        if len(f.verts) == 3:
            tr.append([])
            for v in f.verts:
                tr[-1].append(v.index)

    return (np.array(verts), np.array(tr), np.array(qu))


def read_formatted_mesh(me):
    bm = bmesh.new()
    bm.from_mesh(me)

    loops = afm.read_loops(me)
    if np.max(loops) >= 4:
        # Mesh has ngons/quads! Triangulate ...
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")

    nverts, ntris, nquads = read_bmesh(bm)
    bm.free()

    return nverts, ntris, nquads


def get_nonmanifold_verts(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    res = np.zeros((len(bm.verts)), dtype=np.bool)
    for e in bm.edges:
        if len(e.link_faces) < 2:
            res[e.verts[0].index] = True
            res[e.verts[1].index] = True

    bm.free()
    return res


# Mesh connection ops


def traverse_faces(bm, function, start=None, mask=None):
    if mask is not None:
        non_traversed = np.nonzero(mask == False)  # noqa: E712
    else:
        non_traversed = np.ones(len(bm.faces))
        mask = np.zeros(len(bm.faces), dtype=np.bool)

    if start is None:
        others = [bm.faces[non_traversed[0][0]]]
    else:
        others = [bm.faces[start]]

    new_faces = []
    while others != []:
        new_faces.extend(others)
        step = []
        for f in others:
            if mask[f.index] == False:  # noqa: E712
                mask[f.index] = True
                for e in f.edges:
                    if len(e.link_faces) == 2 and function(e):
                        # append the other face
                        lf = e.link_faces
                        step.append(lf[0] if lf[0] != f else lf[1])
        others = step

    return new_faces


def traverse_faces_limit_plane(bm, function, threshold, start=None, mask=None):
    if mask is not None:
        non_traversed = np.nonzero(mask == False)  # noqa: E712
    else:
        non_traversed = np.ones(len(bm.faces))
        mask = np.zeros(len(bm.faces), dtype=np.bool)

    if start is None:
        others = [bm.faces[non_traversed[0][0]]]
    else:
        others = [bm.faces[start]]

    new_faces = []
    while others != []:
        new_faces.extend(others)

        # calc limit plane normal
        u, s, vh = np.linalg.svd(np.array([f.normal for f in new_faces])[:100])
        normal = vh[2, :]

        step = []
        for f in others:
            if mask[f.index] == False:  # noqa: E712
                mask[f.index] = True
                for e in f.edges:
                    if len(e.link_faces) == 2 and function(e) and np.abs(np.dot(normal, f.normal)) < threshold:
                        # append the other face
                        lf = e.link_faces
                        step.append(lf[0] if lf[0] != f else lf[1])
        others = step

    return new_faces


def other_face(e, face):
    if face == e.link_faces[0]:
        return e.link_faces[1]
    else:
        return e.link_faces[0]


def face_faces(face):
    return [other_face(e, face) for e in face.link_edges]


def faces_verts(faces):
    return {v for f in faces for v in f.verts}


def verts_faces(verts):
    return {f for v in verts for f in v.link_faces}


def vert_vert(v):
    return [e.other_vert(v) for e in v.link_edges]


def con_verts(v, tfun):
    return {x for x in vert_vert(v) if tfun(x)}


def con_area(verts, tfun):
    connected = set()
    for v in verts:
        connected |= con_verts(v, tfun)
    return connected


def get_shell(clean_vert, tfun):
    previous = set([clean_vert])
    connected = con_verts(clean_vert, tfun) | previous
    res = []

    while True:
        for v in connected:
            res.append(v)

        previous = connected
        connected = con_area(connected, tfun)
        if len(connected - previous) == 0:
            break

    return res


def mesh_get_shells(bm):
    traversed = np.zeros((len(bm.verts)), dtype=np.bool)
    shells = []

    while np.any(traversed == False):
        location = np.nonzero(traversed == False)[0][0]
        others = [bm.verts[location]]

        shells.append([])
        while others != []:
            shells[-1].extend(others)
            step = []
            for v in others:
                if traversed[v.index] == False:
                    traversed[v.index] = True
                    step.extend([e.other_vert(v) for e in v.link_edges])
            others = step

    return shells


def mesh_get_edge_connection_shells(bm):
    traversed = np.zeros((len(bm.faces)), dtype=np.bool)
    shells = []

    while np.any(traversed == False):
        location = np.nonzero(traversed == False)[0][0]
        others = [bm.faces[location]]

        shells.append([])
        while others != []:
            shells[-1].extend(others)
            step = []
            for f in others:
                if traversed[f.index] == False:
                    traversed[f.index] = True
                    linked_faces = []
                    for e in f.edges:
                        if len(e.link_faces) > 1:
                            linked_faces.append([i for i in e.link_faces if i != f][0])
                    step.extend(linked_faces)
            others = step
    return shells


def bmesh_get_boundary_edgeloops_from_selected(bm):
    edges = [e for e in bm.edges if e.select]
    t_edges = np.ones((len(bm.edges)), dtype=np.bool)
    for e in edges:
        t_edges[e.index] = False
    loops = []

    while np.any(t_edges == False):
        location = np.nonzero(t_edges == False)[0][0]
        others = [bm.edges[location]]

        loops.append([])
        while others != []:
            loops[-1].extend(others)
            step = []
            for e in others:
                if t_edges[e.index] == False:
                    t_edges[e.index] = True
                    step.extend([e for e in e.verts[0].link_edges if not e.is_manifold])
                    step.extend([e for e in e.verts[1].link_edges if not e.is_manifold])
            others = step

    return [list(set(l)) for l in loops]


def bmesh_vertloop_from_edges(edges):
    res = [edges[0]]
    verts = []
    while len(res) < len(edges) and (len(verts) < 3 or verts[-1] != verts[0] and verts[-1] != verts[-2]):
        r = res[-1]

        e0 = [e for e in r.verts[0].link_edges if e != r and e in edges]
        e1 = [e for e in r.verts[1].link_edges if e != r and e in edges]

        if len(e0) > 1 or len(e1) > 1:
            pass
            # print("invalid edge in bmesh_order_edgeloop()")

        if len(e0) == 0:
            # print("not a loop")
            return None

        test = e0[0] not in res
        te = e0[0] if test else e1[0]
        res.append(te)

        # FIXME: hack
        v = r.verts[int(not test)]
        if len(verts) == 0:
            verts.append(v)
        elif verts[-1] != v:
            verts.append(v)

    verts.append(res[-1].other_vert(verts[-1]))
    # print([i.index for i in verts])

    # final sanity check
    if len(verts) != len(list(set(verts))):
        return None
    else:
        return verts


def bmesh_fill_from_loops(bm, loops):
    new_faces = []
    leftover_loops = []
    for l in loops:
        nl = bmesh_vertloop_from_edges(l)
        if nl:
            f = bm.faces.new(nl)
            f.select = True
            f.smooth = True
            new_faces.append(f)
        else:
            leftover_loops.append(l)

    return new_faces, leftover_loops


def bmesh_deselect_all(bm):
    for v in bm.verts:
        v.select = False

    for e in bm.edges:
        e.select = False


class Bmesh_from_edit:
    def __init__(self, mesh):
        self.mesh = mesh

    def __enter__(self):
        self.bm = bmesh.from_edit_mesh(self.mesh)
        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()
        return self.bm

    def __exit__(self, type, value, traceback):
        bmesh.update_edit_mesh(self.mesh)
