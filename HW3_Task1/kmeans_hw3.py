import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def load_data(data_path, label_path=None):
    X = pd.read_csv(data_path, header=None).values.astype(float)
    y = None
    if label_path:
        y = pd.read_csv(label_path, header=None).values.ravel()
    return X, y


def euclidean_distance_matrix(X, C):
    XX = np.sum(X**2, axis=1)[:, None]
    CC = np.sum(C**2, axis=1)[None, :]
    XC = X.dot(C.T)
    d2 = XX + CC - 2 * XC
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def one_minus_cosine_distance_matrix(X, C, eps=1e-12):
    X_norm = np.linalg.norm(X, axis=1)[:, None]
    C_norm = np.linalg.norm(C, axis=1)[None, :]
    dot = X.dot(C.T)
    denom = np.maximum(X_norm * C_norm, eps)
    cos_sim = dot / denom
    return 1.0 - cos_sim


def one_minus_generalized_jaccard_distance_matrix(X, C, eps=1e-12):
    n, d = X.shape
    k = C.shape[0]
    dist = np.zeros((n, k))
    Xp = np.maximum(X, 0.0)
    Cp = np.maximum(C, 0.0)
    for j in range(k):
        c = Cp[j]
        mins = np.minimum(Xp, c[None, :]).sum(axis=1)
        maxs = np.maximum(Xp, c[None, :]).sum(axis=1)
        sim = np.zeros_like(mins)
        mask = maxs > eps
        sim[mask] = mins[mask] / maxs[mask]
        dist[:, j] = 1.0 - sim
    return dist


class KMeansScratch:
    def __init__(self, n_clusters=3, metric='euclidean', max_iter=500, tol=0.0, random_state=None):
        self.k = int(n_clusters)
        self.metric = metric.lower()
        self.max_iter = int(max_iter)
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.sse_history = []

    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        indices = rng.choice(n, self.k, replace=False)
        return X[indices].astype(float)

    def _distance_matrix(self, X, C):
        if self.metric == 'euclidean':
            return euclidean_distance_matrix(X, C)
        elif self.metric == 'cosine':
            return one_minus_cosine_distance_matrix(X, C)
        elif self.metric == 'jaccard':
            return one_minus_generalized_jaccard_distance_matrix(X, C)
        else:
            raise ValueError(f"Unknown metric {self.metric}")

    def _compute_sse(self, X, labels, C):
        dist_mat = self._distance_matrix(X, C)
        distances = dist_mat[np.arange(X.shape[0]), labels]
        return float(np.sum(distances ** 2))

    def fit(self, X, stop_criterion='centroid', verbose=False):
        """
        stop_criterion: 'centroid' | 'sse_increase' | 'max_iter'
        returns: dictionary with sse history, iterations, time, final labels, centroids
        """
        start_time = time.time()
        n, d = X.shape
        C = self._init_centroids(X)
        prev_centroids = None
        prev_sse = None
        sse_history = []
        labels = np.zeros(n, dtype=int)

        for it in range(1, self.max_iter + 1):
            dist_mat = self._distance_matrix(X, C)
            labels = np.argmin(dist_mat, axis=1)

            new_C = np.zeros_like(C)
            for j in range(self.k):
                members = X[labels == j]
                if len(members) == 0:
                    idx = np.random.randint(0, n)
                    new_C[j] = X[idx]
                else:
                    new_C[j] = members.mean(axis=0)

            C = new_C
            sse = self._compute_sse(X, labels, C)
            sse_history.append(sse)

            if stop_criterion == 'centroid':
                if prev_centroids is not None:
                    centroid_shift = np.linalg.norm(C - prev_centroids)
                    if centroid_shift <= self.tol:
                        if verbose:
                            print(f"[iter {it}] centroid shift {centroid_shift:.6f} <= tol -> stop")
                        break
            elif stop_criterion == 'sse_increase':
                if prev_sse is not None and sse > prev_sse + 1e-12:
                    if verbose:
                        print(f"[iter {it}] SSE increased from {prev_sse:.6f} to {sse:.6f} -> stop")
                    break

            prev_centroids = C.copy()
            prev_sse = sse

        total_time = time.time() - start_time
        self.centroids = C
        self.labels_ = labels
        self.sse_history = sse_history

        summary = {
            'sse_history': sse_history,
            'iterations': len(sse_history),
            'time': total_time,
            'labels': labels,
            'centroids': C
        }
        return summary

    def predict(self, X):
        dist = self._distance_matrix(X, self.centroids)
        return np.argmin(dist, axis=1)


def label_clusters_majority(labels_pred, y_true):
    mapping = {}
    for cluster in np.unique(labels_pred):
        members = y_true[labels_pred == cluster]
        if len(members) == 0:
            mapping[cluster] = None
        else:
            most_common = Counter(members).most_common(1)[0][0]
            mapping[cluster] = most_common
    pred_labels = np.array([mapping[c] for c in labels_pred])
    accuracy = np.mean(pred_labels == y_true)
    return mapping, accuracy, pred_labels


def run_experiment(data_path, label_path, k, metric, stop, max_iter, random_state=None, verbose=False):
    X, y = load_data(data_path, label_path)
    km = KMeansScratch(n_clusters=k, metric=metric, max_iter=max_iter, random_state=random_state)
    result = km.fit(X, stop_criterion=stop, verbose=verbose)
    sse = result['sse_history'][-1] if result['sse_history'] else None
    iters = result['iterations']
    t = result['time']
    mapping, acc, pred_labels = label_clusters_majority(result['labels'], y)
    return {
        'metric': metric,
        'stop': stop,
        'sse_history': result['sse_history'],
        'final_sse': sse,
        'iterations': iters,
        'time_s': t,
        'accuracy': float(acc),
        'cluster_label_map': mapping
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to data CSV (no header). Rows=instances, cols=features.')
    p.add_argument('--labels', required=True, help='Path to labels CSV (one column).')
    p.add_argument('--k', required=True, type=int, help='Number of clusters (K).')
    p.add_argument('--metric', choices=['euclidean', 'cosine', 'jaccard'], default='euclidean')
    p.add_argument('--stop', choices=['centroid', 'sse_increase', 'max_iter'], default='centroid',
                   help='Stopping criterion')
    p.add_argument('--max_iter', type=int, default=500)
    p.add_argument('--random_state', type=int, default=0)
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    out = run_experiment(
        data_path=args.data,
        label_path=args.labels,
        k=args.k,
        metric=args.metric,
        stop=args.stop,
        max_iter=args.max_iter,
        random_state=args.random_state,
        verbose=args.verbose
    )
    print("=== KMeans from-scratch results ===")
    print(f"Metric: {out['metric']}")
    print(f"Stopping criterion: {out['stop']}")
    print(f"K: {args.k}")
    print(f"Iterations: {out['iterations']}")
    print(f"Time (s): {out['time_s']:.4f}")
    print(f"Final SSE: {out['final_sse']:.6f}")
    print(f"Accuracy (majority label): {out['accuracy']:.6f}")
    print("Cluster -> assigned label mapping:")
    for c, lab in sorted(out['cluster_label_map'].items()):
        print(f"  Cluster {c} -> label {lab}")
    print("SSE history (last 10 iterations):")
    print(out['sse_history'][-10:])

if __name__ == '__main__':
    main()
