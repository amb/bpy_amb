# -*- coding:utf-8 -*-

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Created Date: Thursday, August 8th 2019, 1:31:39 pm
# Copyright: Tommi Hyppänen


import mathutils.bvhtree as bvht
import mathutils as mu

import numpy as np

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


class Raycaster:
    def __init__(self, tris):
        import moderngl

        # in: rays, tris
        # out: distance, u, v, face index

        assert tris.dtype == np.float32

        # print("OpenGL supported version (by Blender):", bgl.glGetString(bgl.GL_VERSION))
        self.ctx = moderngl.create_context(require=430)
        assert self.ctx.version_code >= 430
        # print(
        #     "Compute max work group size:",
        #     ctx.info["GL_MAX_COMPUTE_WORK_GROUP_SIZE"],
        #     end="\n\n",
        # )

        self.shader_text = """
        #version 430
        #define TILE_WIDTH 8
        #define TILE_HEIGHT 8

        const ivec2 tileSize = ivec2(TILE_WIDTH, TILE_HEIGHT);

        layout(local_size_x=TILE_WIDTH, local_size_y=TILE_HEIGHT) in;

        layout(binding=0) writeonly buffer out_0 { vec4 outp[]; };

        layout(binding=1) readonly buffer Tris { float tris[]; };
        layout(binding=2) readonly buffer Rays { float rays[]; };

        uniform uint img_size;
        uniform uint tris_size;

        vec3 tri_isec2(vec3 ro, vec3 rd, vec3 v0, vec3 v1, vec3 v2) {
            vec3 v1v0 = v1 - v0;
            vec3 v2v0 = v2 - v0;
            vec3 rov0 = ro - v0;
            vec3 n = cross(v1v0, v2v0);
            vec3 q = cross(rov0, rd);
            float rdn = dot(rd, n);
            //if (rdn == 0.0) return -1.0;
            float d = 1.0 / rdn;
            float u = d * (dot(-q, v2v0));
            float v = d * (dot(q, v1v0));
            float t = d * (dot(-n, rov0));
            if (u < 0.0 || u > 1.0 || v < 0.0 || u + v > 1.0) t = -1.0;
            return vec3(t, u, v);
        }

        void main() {
            uint tx = gl_GlobalInvocationID.x;
            uint ty = gl_GlobalInvocationID.y;
            uint loc = tx + ty * img_size;

            const float maxdist = 10000.0;

            float outc = 0.0;

            vec3 orig = vec3(rays[loc*6+0], rays[loc*6+1], rays[loc*6+2]);
            vec3 dir =  vec3(rays[loc*6+3], rays[loc*6+4], rays[loc*6+5]);

            uint face = -1;
            vec4 res = vec4(maxdist, 0.0, 0.0, 0.0);
            vec3 normal = vec3(0.0, 0.0, 0.0);
            for (uint i=0; i<tris_size; i++) {
                vec3 v0 = vec3(tris[i*9+0], tris[i*9+1], tris[i*9+2]);
                vec3 v1 = vec3(tris[i*9+3], tris[i*9+4], tris[i*9+5]);
                vec3 v2 = vec3(tris[i*9+6], tris[i*9+7], tris[i*9+8]);
                vec3 rc = tri_isec2(orig, dir, v0, v1, v2);
                if (rc[0] > 0.0 && rc[0] < res[0]) {
                    res[0] = rc[0];
                    res[1] = rc[1];
                    res[2] = rc[2];
                    res[3] = float(i);

                    // calc face normal
                    // vec3 va = v0-v1;
                    // vec3 vb = v0-v2;
                    // normal = normalize(cross(va, vb));
                }
            }

            // can only handle 16777217 polys because of floating point conversion error

            // distance, u, v, index
            outp[loc] = res;
        }
        """

        self.tile_size = 8
        self.shader = self.ctx.compute_shader(self.shader_text)

        tri_buffer = self.ctx.buffer(tris)
        tri_buffer.bind_to_storage_buffer(1)
        assert len(tris) < 16777217
        self.shader.get("tris_size", -1).value = len(tris)

    def cast(self, rays):
        assert rays.dtype == np.float32

        # tile_size = 8
        # w, h
        # lrays = len(rays)
        #
        # constraints:
        # lrays//w < h
        # w % 8 == 0

        # is this padding really necessary?
        w = (int(np.sqrt(len(rays) - 1)) // self.tile_size) * self.tile_size + self.tile_size

        assert w * w >= len(rays) and w % self.tile_size == 0

        out_np = np.zeros(shape=(w, w, 4), dtype=np.float32)
        out_buffer = self.ctx.buffer(out_np)
        out_buffer.bind_to_storage_buffer(0)

        rays_np = np.zeros(shape=(w * w, 2, 3), dtype=np.float32)
        rays_np[: rays.shape[0], :, :] = rays
        ray_buffer = self.ctx.buffer(rays)
        ray_buffer.bind_to_storage_buffer(2)

        self.shader.get("img_size", -1).value = w
        self.shader.run(group_x=w // self.tile_size, group_y=w // self.tile_size)

        res = np.frombuffer(out_buffer.read(), dtype=np.float32).reshape((w * w, 4))
        res = res[: rays.shape[0], :]
        return res
