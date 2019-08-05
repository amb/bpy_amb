import numpy as np
import mathutils as mu


def rotate_object(obj, q, point):
    R = q.to_matrix().to_4x4()
    T = mu.Matrix.Translation(point)
    M = T @ R @ T.inverted()
    obj.location = M @ obj.location
    obj.rotation_euler.rotate(M)


def kmeans_pp(data, charts, steps, pp=True):
    """ kmeans++ """
    # 1. Choose one center uniformly at random from among the data points.
    bmf = []
    bmf.append(np.random.randint(len(data)))

    if pp:
        for _ in range(charts):
            # 2. For each data point x, compute D(x), the distance between x and the nearest
            #    center that has already been chosen.
            # TODO: do this entirely in numpy
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
        # 5. Now that the initial centers have been chosen, proceed using standard
        #    k-means clustering.
    else:
        # faster init, with random variables, no plus plus portion of the algorithm applied
        for _ in range(charts):
            bmf.append(np.random.randint(len(data)))

    # start init
    centers = np.array([data[i] for i in bmf], dtype=np.float)
    cluster_pick = np.empty((data.shape[0]), dtype=np.uint)
    dist = np.empty((len(centers), data.shape[0]), dtype=np.float)

    # N steps
    for st_i in range(steps):
        # manhattan distances to cluster centers
        for ict, ct in enumerate(centers):
            dist[ict] = np.sum(np.abs(centers[ict] - data), axis=1)
        cluster_pick = np.argmin(dist, axis=0)

        # move centers to mean
        for ict, ct in enumerate(centers):
            centers[ict] = np.mean(data[cluster_pick == ict], axis=0)

    # return centers, cl_counts, cluster_pick
    return centers, cluster_pick
