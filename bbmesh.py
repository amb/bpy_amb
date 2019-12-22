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

from collections import defaultdict
import numpy as np
import bmesh
from . import fastmesh as afm
from . import bbmesh as abm

# test


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


def traverse_faces(bm, function, start=None, mask=None, mark_face=False):
    if mask is not None:
        non_traversed = (mask == False).argmax(axis=0)  # noqa: E712
        # non_traversed = np.nonzero(mask == False)[0]  # noqa: E712
    else:
        non_traversed = 0
        mask = np.zeros(len(bm.faces), dtype=np.bool)

    if start is None:
        others = [bm.faces[non_traversed]]
    else:
        others = [bm.faces[start]]

    t_edges = np.zeros(len(bm.edges), dtype=np.bool)

    new_faces = []
    while others != []:
        new_faces.extend(others)
        step = []
        for f in others:
            if mask[f.index] == False:  # noqa: E712
                mask[f.index] = True
                for e in f.edges:
                    if t_edges[e.index]:
                        continue

                    t_edges[e.index] = True

                    # traverse only manifold
                    if e.is_contiguous:
                        if function(e, f):
                            # append the other face
                            lf = e.link_faces
                            step.append(lf[0] if lf[0] != f else lf[1])
                        # TODO: figure out if this actually works right
                        elif mark_face:
                            # mark all edges in face as untraversable
                            lf = e.link_faces
                            o = lf[0] if lf[0] != f else lf[1]
                            mask[o.index] = True
                            for f_e in o.edges:
                                t_edges[f_e.index] = True
        others = step

    return new_faces, t_edges


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
                    if (
                        len(e.link_faces) == 2
                        and function(e)
                        and np.abs(np.dot(normal, f.normal)) < threshold
                    ):
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

        # only uniques
        shells[-1] = list(set(shells[-1]))
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
    while len(res) < len(edges) and (
        len(verts) < 3 or verts[-1] != verts[0] and verts[-1] != verts[-2]
    ):
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


def mark_seams_from_charts(bm, new_faces):
    # mark seams from charts
    for chart in new_faces:
        edges = defaultdict(int)
        for f in chart:
            for e in f.edges:
                edges[e.index] += 1

        for k, v in edges.items():
            if v == 1:
                bm.edges[k].seam = True


def bmesh_deselect_all(bm):
    for v in bm.verts:
        v.select = False

    for e in bm.edges:
        e.select = False


def grow_uv_selection(bm, uv_layer, polys, selected, steps):
    """ Grow UV selection """
    # TODO: this is really ad hoc
    connections = defaultdict(set)
    for fi in polys:
        f = bm.faces[fi]
        for l in f.loops:
            l1 = l
            l2 = l.link_loop_next
            uv1 = tuple(l1[uv_layer].uv)
            uv2 = tuple(l2[uv_layer].uv)
            connections[uv1].add(l2)
            connections[uv2].add(l1)

    for _ in range(steps):
        selected = set(selected)
        growth = []
        for s in selected:
            for c in connections[s]:
                growth.append(c)
        growth = set(growth)

        selected = []
        for g in growth:
            selected.append(tuple(g[uv_layer].uv))
            g[uv_layer].select = True


def grow_uv_to_faces(bm, uv_layer):
    """ Grow UV selection to faces """
    for f in bm.faces:
        mark_all = False
        for l in f.loops:
            if l[uv_layer].select:
                mark_all = True
                break

        if mark_all:
            for l in f.loops:
                l[uv_layer].select = True


def edge_same_uv(e0, uv_layer):
    # TODO: nonmanifold mesh

    # if not e0.is_contiguous:
    #     return False

    loop0, loop1 = e0.link_loops

    # ASSUMPTION: loop always goes the same direction (link_loop_next)
    a0, a1 = loop0[uv_layer].uv, loop0.link_loop_next[uv_layer].uv
    b0, b1 = loop1[uv_layer].uv, loop1.link_loop_next[uv_layer].uv

    if (a0 == b0 and a1 == b1) or (a0 == b1 and a1 == b0):
        return True
    else:
        return False


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


def radial_edges(iv):
    # TODO: accept only meshes with no loose edges or verts
    if len(iv.link_edges) == 0:
        raise Exception("ERROR: Invalid source mesh. Loose verts with no edges.")
    for e in iv.link_edges:
        if not e.is_manifold:
            return None
        if len(e.link_faces) == 0:
            raise Exception("ERROR: Invalid source mesh. Wire edges.")
    loop = iv.link_loops[0]
    eg = []
    while True:
        eg.append(loop.edge)
        loop = loop.link_loop_radial_next.link_loop_next
        if loop.edge.verts[0] != iv and loop.edge.verts[1] != iv:
            return None
        if loop.edge == eg[0]:
            break
    return eg


def delaunay_criterion(bm, max_iter=100):
    """ Basically the same as beautify faces """
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    prev_flips = -1
    count = 0
    edge_set = set([e for e in bm.edges])
    while True:
        modified = set([])
        flips = 0

        for e in edge_set:
            f = e.link_faces
            if len(f) == 2 and len(f[0].verts) == 3 and len(f[1].verts) == 3:
                ev0, ev1 = e.verts
                l0 = next(i for i in f[0].loops if i.vert != ev0 and i.vert != ev1)
                l1 = next(i for i in f[1].loops if i.vert != ev0 and i.vert != ev1)
                a0 = l0.calc_angle()
                a1 = l1.calc_angle()
                if a0 + a1 > np.pi:
                    for nev in e.verts:
                        modified.add(nev)
                    ne = bmesh.utils.edge_rotate(e, True)
                    if ne:
                        for nev in ne.verts:
                            modified.add(nev)
                    flips += 1

        count += 1

        if flips == prev_flips or count > max_iter:
            break
        prev_flips = flips

        edge_set.clear()
        for v in modified:
            for e in v.link_edges:
                edge_set.add(e)


def cotan(a, b):
    # va, vb = a, b
    # t = va.x * vb.z - va.z * vb.x
    # return 0.0 if t == 0.0 else (va.x * vb.x + va.z * vb.z) / t
    t = (a.cross(b)).length
    return 0.0 if t == 0.0 else (a.dot(b)) / t


def cotan_weights(bm, s_verts):
    # TODO: accept only meshes with no loose edges or verts
    # radially sorted 1-ring/valence of each vert
    rad_v = {}
    rad_manifold = {}
    for v in s_verts:
        re = abm.radial_edges(v)
        if re is not None:
            rad_v[v] = [e.other_vert(v) for e in re]
            rad_manifold[v] = True
        else:
            rad_v[v] = [e.other_vert(v) for e in v.link_edges]
            rad_manifold[v] = False

        assert len(rad_v[v]) == len(v.link_edges)

    # cotan weights
    # print(">cot_wt ", end="")
    cot_eps = 1e-5
    cot_max = np.cos(cot_eps) / np.sin(cot_eps)
    v_wg = {}
    v_area = {}
    min_area = 1.0e10
    mm = 1 / 3
    non_manifold_verts = []
    for vid, v in enumerate(s_verts):
        # print(vid, end=",")
        wgs = []
        if rad_manifold[v]:
            # manifold vertex valence
            assert len(rad_v) > 0
            rv_v = rad_v[v]
            v_area[v] = mm * sum(f.calc_area() for f in v.link_faces)
            if v_area[v] < min_area:
                min_area = v_area[v]
            totw = 0.0
            for ri, rv in enumerate(rv_v):
                pv = rv_v[(ri - 1) % len(rv_v)]
                nv = rv_v[(ri + 1) % len(rv_v)]
                cv = rv_v[ri]
                v0 = cv.co - v.co
                vb = pv.co - v.co
                va = nv.co - v.co
                cot_a = cotan(v0 - va, -va)
                cot_b = cotan(v0 - vb, -vb)
                wg = cot_a + cot_b
                if wg > cot_max:
                    wg = cot_max
                if wg < -cot_max:
                    wg = -cot_max
                wgs.append(wg)
                totw += wg
            v_wg[v] = [w / totw for w in wgs]
        else:
            # non-manifold vertex valence
            non_manifold_verts.append(v)
            # v_area[v] = sum(f.calc_area() for f in v.link_faces)
            # if v_area[v] <= 0.0:
            #     # no connected faces
            #     v_area[v] = 1.0e-10
            # elif v_area[v] < min_area:
            #     min_area = v_area[v]
            vnum = len(v.link_edges)
            v_wg[v] = [1.0 / vnum for _ in range(vnum)]
            # for ri, rv in enumerate(e.other_vert(v) for e in v.link_edges):

    for v in non_manifold_verts:
        v_area[v] = min_area

    return v_wg, v_area, min_area, rad_v
