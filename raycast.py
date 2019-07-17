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

Created Date: Sunday, July 14th 2019, 1:10:18 pm
Copyright: Tommi Hyppänen
"""

print("Import: raycast.py")

import mathutils.bvhtree as bvht
import mathutils as mu

bvh = None


def init_with_bm(bm):
    global bvh
    bvh = bvht.BVHTree.FromBMesh(bm)


def simple_sample(center, normal):
    component_A = mu.Vector([normal[1], -normal[0], normal[2]])
    component_B = normal.cross(component_A).normalized()
    assert abs(normal.cross(component_A).dot(normal)) < 0.001
    casts = []
    casts.append(bvh.ray_cast(center, normal))
    casts.append(bvh.ray_cast(center, (normal + component_A).normalized()))
    casts.append(bvh.ray_cast(center, (normal - component_A).normalized()))
    casts.append(bvh.ray_cast(center, (normal + component_B).normalized()))
    casts.append(bvh.ray_cast(center, (normal - component_B).normalized()))
    return casts
