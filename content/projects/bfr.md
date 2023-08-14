---
title: "BFR"
url: /projects/bfr
type: posts
desc: [
    "<b>code</b>: <a href=\"https://github.com/kali-v/tiny-lab/blob/main/bfr.ipynb\"> here </a>"
]
---

## Experiments

The following is a from-scratch implementation of BFR and k-means and a performance comparison. The code also contains functions for generating datasets that have distinct clusters and artificial noise in them. All experiments and conductions were done on these datasets and might vary on different ones.

**Initialization** - Regarding the initialization phase of the BFR algorithm, some sources suggest clustering the first batch using an in-memory algorithm and creating a Discard Set (DS) from sufficiently large clusters. Even though this method has the advantage of a variable number of clusters, I found this method to have worse SSE and Silhouette scores compared to a method that clusters a subset to a fixed number of clusters from the first batch and creates DS from this clustering. While this method yields a fixed number of clusters, it generally produces better results, provided that the number of clusters (K) is chosen correctly.

**Dimensionality** - BFR was designed to cluster data in a high-dimensional space, so the results are much better for higher dimensions datasets. When comparing BFR with k-means on the 2D dataset, BFR was able to achieve only 30% of the k-means SSE score and about 80% of the Silhouette score. When dealing with 10+ dimensional datasets, the BFR was able to have scores similar to k-means.

**Granularity of k-means** - Implemented k-means has an ending condition defined as a number of samples that changed cluster must be under 'KEND'. Variable 'KEND' plays a significant role not only in speed but also final result. With smaller 'KEND', the algorithm runs longer but results in more distinct clusters. Because BFR calls k-means on smaller batches, we can achieve decent speed even with smaller 'KEND'. On bigger datasets, BFR can achieve several times speed-up compared to k-means.

**Dataset size** - BFR was designed to handle big datasets, so the performance on small datasets is not that great.

**Batch size** - Bigger batch sizes lead to better results, but the difference starts to diminish with bigger datasets. Bigger batch sizes also increase the wall-clock time.




```python
# %pip install seaborn==0.12.1 matplotlib==3.6.0 scipy==1.9.1 sklearn==1.1.2 numpy==1.23.3
```

## Dataset

generating datasets to play with.


```python
from scipy.stats.qmc import Sobol
import random
import numpy as np
import math

def sample_sobol(d, n):
    sampler = Sobol(d)
    return sampler.random(n)


def sample_random(d, n):
    samples = []
    for i in range(n):
        sample = []
        for s in range(d):
            sample.append(random.random())
        samples.append(sample)

    return np.array(samples)


def euclidean_dist(a, b):
    return np.sqrt(((a - b)**2).sum())


def generate_dataset(d, n, dsens):
    """
    First sample from quasi random Sobol
    then choose centers and remove samples far from any center
    then sample random points to create noise 
    """
    samples = sample_sobol(d, n*10)

    nclusters = random.randint(8, 11)
    clusters = []
    while len(clusters) != nclusters:
        cr = np.array([random.random() for _ in range(d)])
        distinct = True
        for cluster in clusters:
            if euclidean_dist(cr, cluster) < 0.25:
                distinct = False
                break
        if distinct:
            clusters.append(cr)

    _samples = []
    for cluster in clusters:
        for sample in samples:
            append = True
            for s in range(d):
                sen = random.random()/random.randint(*dsens)*d
                if math.sqrt((cluster[s] - sample[s])**2) > sen:
                    append = False
                    break

            if append:
                _samples.append(sample)
            if len(_samples) >= (int)(n * 0.985):
                break

    _samples = np.array([*_samples, *sample_random(d, (int)(n*0.015))])
    return _samples
```

## Plotting

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_samples(samples, cluster_ids):
    sns.jointplot(x=samples[:, 0], y=samples[:, 1], hue=cluster_ids, s=5)
    plt.show()


def plot_with_centers(centers):
    # just for drawing centers as points in plot
    for i in range(len(centers)):
        _samples = np.append(_samples, [centers[i]], axis=0)
        _cluster_ids = np.append(_cluster_ids, K + 1)
    plot_samples(_samples, _cluster_ids)

```

# kmeans + BFR

```python
from sklearn import metrics
import time

def kmeans(data_set, centers):
    dims = len(data_set[0])
    cluster_ids = []
    clusters = [[] for _ in range(len(centers))]

    for i, point in enumerate(data_set):
        min_dis = euclidean_dist(point, centers[0])
        min_index = 0
        for j, center in enumerate(centers[1:]):
            dis = euclidean_dist(point, center)
            if dis < min_dis:
                min_dis = dis
                min_index = j + 1
        clusters[min_index].append(data_set[i])
        cluster_ids.append(min_index)

    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            sums = [0 for _ in range(dims)]

            for point in cluster:
                for d in range(dims):
                    sums[d] += point[d]

            for j in range(dims):
                sums[j] /= len(cluster)

            centers[i] = sums

    return centers, clusters, cluster_ids


def kmeans_fit(samples, K):
    """
    kmeans that ends if less than KEND samples changed cluster
    """
    centers = random.sample(list(samples), K)
    cluster_ids = []

    it = 0
    while True:
        centers, clusters, _cluster_ids = kmeans(samples, centers)
        diff = sum([0 if _cluster_ids[i] == cluster_ids[i] else 1 for i in range(len(cluster_ids))])
        cluster_ids = _cluster_ids

        if diff < KEND and it != 0:
            return clusters, _cluster_ids, centers
            break

        it += 1


def std_dev(points: np.array):
    if len(points) == 1:
        return 1
    u = points.mean()
    return np.sqrt(((points - u)**2).sum()/len(points))


def load_batch(samples, i):
    return samples[i * BS:(i+1) * BS]


def mahal_dist(point, summary):
    xc = (point - (summary[1]/summary[0]))
    var = summary[2] / summary[0] - (summary[1] / summary[0])**2
    stddev = np.sqrt(var) + 1e-7  # just to be sure we don't divide by zero
    return np.sqrt(((xc/stddev)**2).sum())


def is_cluster_small(cluster, nosamples):
    return len(cluster) < SMALL_CLUSTER_RATIO * nosamples


def is_cluster_smallv2(cluster):
    return len(cluster) < SMALL_CLUSTER


def is_cluster_big(cluster, nosamples):
    return len(cluster) > BIG_CLUSTER_RATIO * nosamples


def is_cluster_bigv2(cluster):
    return len(cluster) > BIG_CLUSTER


def init_summary(cluster):
    # return summary for cluster

    dims = len(cluster[0])
    n = len(cluster)
    _sum = np.zeros(dims)
    _sumq = np.zeros(dims)

    for point in cluster:
        _sum += point
        _sumq += point**2

    return [n, _sum, _sumq]


def update_summary(summary, cluster):
    # update summary with cluster
    summary[0] += len(cluster)
    for point in cluster:
        summary[1] += point
        summary[2] += point**2
    return summary


def var_comb_cluster(a, b):
    # compute variance for combined a, b clusters
    n = a[0] + b[0]
    s = a[1] + b[1]
    sq = a[2] + b[2]

    return (sq/n) - (s/n)**2


def merge_cs(CS):
    if len(CS) > 1:
        for i in range(len(CS)):
            j = i
            while j < len(CS):
                if (CS_MERGE_TRESH > var_comb_cluster(CS[i], CS[j])).all():
                    # merge CS
                    CS[i][0] += CS[j][0]
                    CS[i][1] += CS[j][1]
                    CS[i][2] += CS[j][2]
                    del CS[j]
                    j -= 1
                j += 1


def init_bfr(_samples, DS, CS, RS):
    rest_points = []

    if True:
        d = (int)(len(_samples)/10)
        subset = _samples[0:d]
        clusters, cluster_ids, _ = kmeans_fit(subset, K)
        for cluster in clusters:
            DS.append(init_summary(cluster))
        rest_points = _samples[d:]
    else:
        # slightly different initialization
        # hard to find the right hyperparams, but variable n of clusters
        clusters, cluster_ids, _ = kmeans_fit(_samples, KK)
        for i, cluster in enumerate(clusters):
            if is_cluster_bigv2(cluster):
                DS.append(init_summary(cluster))
            else:
                rest_points.extend(cluster)

    if len(rest_points) > 0:
        clusters, cluster_ids, _ = kmeans_fit(rest_points, min(KK, len(rest_points)))
        for cluster in clusters:
            if is_cluster_smallv2(cluster):
                RS.extend(cluster)
            else:
                CS.append(init_summary(cluster))


def merge_all(DS, CS, RS):
    if len(DS) > 0:
        # merge points into the closest DS
        for sample in RS:
            min_dist = float("inf")
            for dsi, ds in enumerate(DS):
                dist = mahal_dist(sample, ds)
                if dist < min_dist:
                    min_dist = dist
                    minin = dsi
            DS[minin] = update_summary(DS[minin], [sample])

        # merge cs into the closest DS
        for cs in CS:
            min_dist = float("inf")
            for dsi, ds in enumerate(DS):
                dist = ((ds[1]/ds[0] - cs[1]/cs[0])**2).sum()
                if dist < min_dist:
                    min_dist = dist
                    minin = dsi

            DS[minin][0] += cs[0]
            DS[minin][1] += cs[1]
            DS[minin][2] += cs[2]


def cluster_rs(CS, RS):
    if len(RS) > 0:
        clusters, cluster_ids, _ = kmeans_fit(RS, min(KK, len(RS)))
        RS = []
        for cluster in clusters:
            if is_cluster_smallv2(cluster):
                RS.extend(cluster)
            else:
                CS.append(init_summary(cluster))


def classify_samples(samples, DS, CS, RS):
    ndims = len(samples[0])
    for sample in samples:
        is_merged = False
        for i, summary in enumerate(DS):
            if mahal_dist(sample, summary) < math.sqrt(ndims):
                DS[i] = update_summary(DS[i], sample)
                is_merged = True
                break
        if not is_merged:
            for i, summary in enumerate(CS):
                if mahal_dist(sample, summary) < math.sqrt(ndims):
                    CS[i] = update_summary(CS[i], sample)
                    is_merged = True
                    break
        if not is_merged:
            RS.append(sample)


def bfr(samples, debugging=False):
    DS = []
    CS = []
    RS = []

    _samples = load_batch(samples, 0)
    init_bfr(_samples, DS, CS, RS)

    it = 1
    _samples = load_batch(samples, it)
    while len(_samples) > 0:
        classify_samples(_samples, DS, CS, RS)
        cluster_rs(CS, RS)
        merge_cs(CS)

        it += 1
        _samples = load_batch(samples, it)

    merge_all(DS, CS, RS)

    clss = []
    if debugging and len(DS) > 0:
        # assign samples to clusters
        clusters = [[] for _ in range(len(DS))]
        for sample in samples:
            min_dist = float('inf')
            minin = 0
            for dsi, ds in enumerate(DS):
                dist = mahal_dist(sample, ds)
                if dist < min_dist:
                    min_dist = dist
                    minin = dsi

            clss.append(minin)
            clusters[minin].append(sample)

        for i in range(len(clusters)):
            clusters[i] = np.array(clusters[i])

        centers = [d[1]/d[0] for d in DS]

        return clusters, clss, centers

```

```python

def visualize(samples, RS, CS, DS):
    colors = []
    for sample in samples:
        if np.isin(sample, RS).any():
            colors.append(1)
        else:
            min_dist = float('inf')
            colors.append(-1)
            idx = len(colors) - 1
            for csi, cs in enumerate(CS):
                dist = mahal_dist(sample, cs)
                if dist < min_dist:
                    min_dist = dist
                    colors[idx] = 2
            for dsi, ds in enumerate(DS):
                dist = mahal_dist(sample, ds)
                if dist < min_dist:
                    min_dist = dist
                    colors[idx] = 3

    print("RS", len(RS))
    print("CS", len(CS))
    print("DS", len(DS))

    plot_samples(samples, colors)

    # plot_samples(_samples, cluster_ids)


def sse(clusters, centers):
    _sse = 0
    for i, centroid in enumerate(centers):
        for sample in clusters[i]:
            for d in range(len(centroid)):
                _sse += (sample[d] - centroid[d]) ** 2
    return _sse


def test_bfr_kmeans(n):
    bfr_time = bfr_sse = bfr_sil = 0
    kmeans_time = kmeans_sse = kmeans_sil = 0

    for i in range(n):
        print(f"test {i+1}")
        samples = generate_dataset(TEST_DIM, N, TEST_DSENS)

        st = time.time()
        clusters, clss, centers = bfr(samples, True)
        bfr_time += time.time() - st
        bfr_sse += sse(clusters, centers)
        bfr_sil += metrics.silhouette_score(samples, clss) + 1

        st = time.time()
        clusters, clss, centers = kmeans_fit(samples, K)
        kmeans_time += time.time() - st
        kmeans_sse += sse(clusters, centers)
        kmeans_sil += metrics.silhouette_score(samples, clss) + 1

    print("bfr time", bfr_time/n)
    print("bfr sse", bfr_sse/n)
    print("bfr sil", bfr_sil/n)
    print("----")
    print("kmeans time", kmeans_time/n)
    print("kmeans sse", kmeans_sse/n)
    print("kmeans sil", kmeans_sil/n)
    print("----")
    # bfr better with 1+
    print("bfr/kmeans perf time", kmeans_time/bfr_time)
    print("bfr/kmeans perf sse", kmeans_sse/bfr_sse)
    print("bfr/kmeans perf sil", bfr_sil/kmeans_sil)


def bfr_dataset(path):
    samples = np.genfromtxt(path, delimiter=',')

    st = time.time()
    clusters, clss, centers = kmeans_fit(samples, K)
    # clusters, clss, centers = bfr(samples, True)

    print(time.time() - st)
    print(len(samples))
    print(len(clusters))
    print(sse(clusters, centers))
    print(metrics.silhouette_score(samples, clss))
```

```python
TEST_DIM = 15
TEST_DSENS = [6, 13]

N = 40000  # number of samples
BS = 20000  # BFR batch size
KEND = 500  # BFR kmeans ends if less than KEND samples changed cluster

K = 8
KK = 8  # K for the inner kmeans in bfr

BIG_CLUSTER_RATIO = 0.05
BIG_CLUSTER = 100

SMALL_CLUSTER = 5
SMALL_CLUSTER_RATIO = 0.01

CS_MERGE_TRESH = 0.02


if __name__ == "__main__":
    # samples = generate_dataset(2, 1000, [7, 9])
    # plot_samples(samples, None)

    num_tests = 2
    test_bfr_kmeans(num_tests)

    # bfr_dataset('2d-dataset.csv')
    # bfr_dataset('13d-dataset.csv')

```
