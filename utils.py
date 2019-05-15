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


import cProfile
import pstats
import io
import numpy as np


# TODO: with aut.set_mode("OBJECT")  # OBJECT, EDIT ...


def profiling_start():
    # profiling
    pr = cProfile.Profile()
    pr.enable()
    return pr


def profiling_end(pr):
    # end profile, print results
    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats(sortby).print_stats(20)
    print(s.getvalue())


def rotate_object(obj, q, point):
    R = q.to_matrix().to_4x4()
    T = mu.Matrix.Translation(point)
    M = T @ R @ T.inverted()
    obj.location = M @ obj.location
    obj.rotation_euler.rotate(M)


def kmeans_pp(data, charts, steps):
    """ kmeans++ """
    # 1. Choose one center uniformly at random from among the data points.
    bmf = []
    bmf.append(np.random.randint(len(data)))

    for _ in range(charts):
        # 2. For each data point x, compute D(x), the distance between x and the nearest
        #    center that has already been chosen.
        dist = []
        for ict in bmf:
            dist.append(np.sum(np.abs(data[ict] - data), axis=1))
        dist = np.min(np.array(dist, dtype=np.float), axis=0)

        # 3. Choose one new data point at random as a new center, using a weighted probability
        #    distribution where a point x is chosen with probability proportional to D(x)^2.
        dist **= 2

        # TODO: check the math here
        dist *= np.random.random()
        bmf.append(np.argmax(dist))

    # 4. Repeat Steps 2 and 3 until k centers have been chosen.
    # 5. Now that the initial centers have been chosen, proceed using standard k-means clustering.

    # start init
    centers = np.array([data[i] for i in bmf], dtype=np.float)
    cluster_pick = np.zeros((data.shape[0]), dtype=np.uint)
    dist = np.empty((len(centers), data.shape[0]), dtype=np.float)

    # N steps
    for st_i in range(steps):
        # manhattan distances to cluster centers
        for ict, ct in enumerate(centers):
            dist[ict] = np.sum(np.abs(centers[ict] - data), axis=1)
        cluster_pick = np.argmin(dist, axis=0)
        cl_counts = np.bincount(cluster_pick)

        # pad end
        if cl_counts.shape[0] < centers.shape[0]:
            cl_counts = np.resize(cl_counts, (centers.shape[0],))

        # recalc centers
        np.add.at(centers, cluster_pick, data)
        for ict, ct in enumerate(centers):
            if cl_counts[ict] > 0:
                centers[ict] /= cl_counts[ict]

    return centers, cl_counts, cluster_pick
