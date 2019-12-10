import numpy as np
import mathutils as mu


def tri_ori(p1, p2, p3):
    # skip colinearity test
    return (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1]) > 0


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


# https://github.com/chrisjmccormick/dbscan
# MIT-license
"""
This is a simple implementation of DBSCAN intended to explain the algorithm.
@author: Chris McCormick
"""


def MyDBSCAN(D, eps, MinPts):
    """
    Cluster the dataset `D` using the DBSCAN algorithm.

    MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.

    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """

    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.
    labels = [0] * len(D)

    # C is the ID of the current cluster.
    C = 0

    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the
    # cluster growth is all handled by the 'expandCluster' routine.

    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    for P in range(0, len(D)):

        # Only points that have not already been claimed can be picked as new
        # seed points.
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
            continue

        # Find all of P's neighboring points.
        NeighborPts = regionQuery(D, P, eps)

        # If the number is below MinPts, this point is noise.
        # This is the only condition under which a point is labeled
        # NOISE--when it's not a valid seed point. A NOISE point may later
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to
        # something else).
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        # Otherwise, if there are at least MinPts nearby, use this point as the
        # seed for a new cluster.
        else:
            C += 1
            growCluster(D, labels, P, NeighborPts, C, eps, MinPts)

    # All data has been clustered!
    return labels


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    Grow a new cluster with label `C` from the seed point `P`.

    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.

    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # Assign the cluster label to the seed point.
    labels[P] = C

    # Look at each neighbor of P (neighbors are referred to as Pn).
    # NeighborPts will be used as a FIFO queue of points to search--that is, it
    # will grow as we discover new branch points for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original
    # dataset.
    i = 0
    while i < len(NeighborPts):

        # Get the next point from the queue.
        Pn = NeighborPts[i]

        # If Pn was labelled NOISE during the seed search, then we
        # know it's not a branch point (it doesn't have enough neighbors), so
        # make it a leaf point of cluster C and move on.
        if labels[Pn] == -1:
            labels[Pn] = C

        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C

            # Find all the neighbors of Pn
            PnNeighborPts = regionQuery(D, Pn, eps)

            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched.
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            # else:
            # Do nothing
            # NeighborPts = NeighborPts

        # Advance to the next point in the FIFO queue.
        i += 1

    # We've finished growing cluster C!


def regionQuery(D, P, eps):
    """
    Find all points in dataset `D` within distance `eps` of point `P`.

    This function calculates the distance between a point P and every other
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    neighbors = []

    # For each point in the dataset...
    for Pn in range(0, len(D)):

        # If the distance is below the threshold, add it to the neighbors list.
        if np.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)

    return neighbors
