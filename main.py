import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
 
def weisfeiler_lehman_hash(node_labels, adj, h=3):
    """Weisfeiler-Lehman graph isomorphism test / kernel features."""
    labels = list(node_labels)
    all_labels = [set(labels)]
    for _ in range(h):
        new_labels = []
        for i, label in enumerate(labels):
            neighbors = sorted([labels[j] for j in range(len(labels)) if adj[i][j]])
            new_label = hash((label, tuple(neighbors))) % (10**6)
            new_labels.append(new_label)
        labels = new_labels
        all_labels.append(set(labels))
    return all_labels
 
def wl_kernel(g1_labels, g1_adj, g2_labels, g2_adj, h=3):
    """WL subtree kernel between two graphs."""
    h1 = weisfeiler_lehman_hash(g1_labels, g1_adj, h)
    h2 = weisfeiler_lehman_hash(g2_labels, g2_adj, h)
    k = 0
    for l1, l2 in zip(h1, h2):
        k += len(l1 & l2)
    return k
 
def random_walk_kernel(A1, A2, steps=3, lam=0.1):
    """Random walk kernel (simplified)."""
    n1, n2 = len(A1), len(A2)
    A1_norm = A1 / (A1.sum(1, keepdims=True) + 1e-8)
    A2_norm = A2 / (A2.sum(1, keepdims=True) + 1e-8)
    k = 0
    rw1 = np.eye(n1); rw2 = np.eye(n2)
    for t in range(steps):
        rw1 = rw1 @ A1_norm; rw2 = rw2 @ A2_norm
        k += lam**t * np.sum(rw1[:min(n1,n2),:min(n1,n2)] *
                              rw2[:min(n1,n2),:min(n1,n2)])
    return k
 
# Simulate 20 small molecular graphs
np.random.seed(42)
graphs = []
targets = []
for i in range(20):
    n = np.random.randint(4, 9)
    adj = (np.random.rand(n,n) > 0.6).astype(float)
    adj = ((adj + adj.T) > 0).astype(float); np.fill_diagonal(adj, 0)
    labels = np.random.randint(0, 4, n).tolist()
    graphs.append((labels, adj)); targets.append(np.random.rand())
 
K = np.array([[wl_kernel(g1[0],g1[1], g2[0],g2[1]) for g2 in graphs] for g1 in graphs], float)
K /= np.sqrt(np.outer(np.diag(K), np.diag(K)) + 1e-8)
print(f"WL Kernel matrix shape: {K.shape}")
print(f"Kernel matrix diagonal (self-similarity): {K.diagonal()[:5].round(2)}")
print(f"Sample off-diagonal: {K[0,1]:.3f}")
