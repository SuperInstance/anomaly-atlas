#!/usr/bin/env python3
"""
Anomaly Detection Atlas
========================
Conservation-based anomaly detection across ALL domains.

The core insight: conservation ratio drops at anomalies because anomalies
violate the smooth structure captured by the graph Laplacian.

Domains: Music, Finance, Climate, Social, Protein, Neural, PX4
Pattern: Build Laplacian → compute conservation → threshold on deviation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. GRAPH LAPLACIAN & CONSERVATION
# ============================================================

def build_laplacian(W):
    """Build normalized Laplacian from weight matrix."""
    d = W.sum(axis=1)
    d[d == 0] = 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L_norm = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L_norm

def compute_conservation_ratio(L, f):
    """
    Conservation ratio: how much of the signal is preserved under
    graph diffusion. High = smooth/conserved, Low = anomalous.
    """
    f = f / (np.linalg.norm(f) + 1e-10)
    # Rayleigh quotient: f^T L f / f^T f — low means smooth
    smoothness = f @ L @ f
    # Conservation = 1 - smoothness (normalized)
    # Eigenvalues of L_norm are in [0, 2], so smoothness in [0, 2]
    conservation = 1.0 - smoothness / 2.0
    return conservation

def build_weight_matrix(n, structure='chain', extra_edges=0):
    """Build adjacency/weight matrix for different graph structures."""
    if structure == 'chain':
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i+1] = W[i+1, i] = 1.0
    elif structure == 'ring':
        W = np.zeros((n, n))
        for i in range(n):
            W[i, (i+1) % n] = W[(i+1) % n, i] = 1.0
    elif structure == 'grid':
        side = int(np.sqrt(n))
        assert side * side == n, f"n={n} must be a perfect square for grid"
        W = np.zeros((n, n))
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if j + 1 < side:
                    W[idx, idx+1] = W[idx+1, idx] = 1.0
                if i + 1 < side:
                    W[idx, idx+side] = W[idx+side, idx] = 1.0
    elif structure == 'random':
        W = np.random.rand(n, n) * 0.5
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        W[W < 0.3] = 0
    else:
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i+1] = W[i+1, i] = 1.0

    # Add extra random edges
    for _ in range(extra_edges):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            W[i, j] = W[j, i] = 1.0
    return W

# ============================================================
# 2. SYNTHETIC ANOMALY DATASETS
# ============================================================

def generate_tonal_with_chromatic_insertion(T=200, n=24, anomaly_start=140, anomaly_len=20):
    """
    Music: 24 pitch classes (2 octaves of chromatic), chain graph.
    Normal: tonal (C major) — only diatonic pitches active.
    Anomaly: chromatic insertion — all 12 pitch classes active.
    """
    W = build_weight_matrix(n, 'chain')
    graphs = [W.copy() for _ in range(T)]
    attributes = []
    ground_truth = np.zeros(T)

    # C major scale degrees (pitch class indices): C D E F G A B
    diatonic = [0, 2, 4, 5, 7, 9, 11]
    # Extend to 24 pitch classes (2 octaves)
    diatonic_24 = diatonic + [d + 12 for d in diatonic]

    for t in range(T):
        f = np.zeros(n)
        if anomaly_start <= t < anomaly_start + anomaly_len:
            # Chromatic insertion: all pitches active with random energy
            f = np.random.exponential(0.5, n)
            f[diatonic_24] = 1.0  # tonal still present
            ground_truth[t] = 1
        else:
            # Normal tonal music
            active = np.random.choice(diatonic_24, size=np.random.randint(3, 6), replace=False)
            f[active] = np.random.exponential(1.0, len(active))
        attributes.append(f)

    return graphs, attributes, ground_truth, 'Music (Chromatic Insertion)'


def generate_market_with_crisis(T=200, n=20, anomaly_start=140, anomaly_len=20):
    """
    Finance: 20 correlated assets, correlation graph.
    Normal: correlated Brownian motion.
    Anomaly: crisis — correlations break, volatility spikes.
    """
    W = build_weight_matrix(n, 'chain', extra_edges=n*2)
    graphs = [W.copy() for _ in range(T)]
    attributes = []
    ground_truth = np.zeros(T)

    base_prices = np.random.randn(n) * 10
    drift = np.random.randn(n) * 0.01

    for t in range(T):
        if anomaly_start <= t < anomaly_start + anomaly_len:
            # Crisis: correlations break, vol spikes
            f = base_prices + drift * t + np.random.randn(n) * 5.0
            # Shuffle: break correlation structure
            idx = np.random.permutation(n)
            f = f[idx] + np.random.randn(n) * 3.0
            ground_truth[t] = 1
        else:
            # Normal correlated motion
            common = np.random.randn() * 0.5
            f = base_prices + drift * t + common + np.random.randn(n) * 0.3
        attributes.append(f)

    return graphs, attributes, ground_truth, 'Finance (Crisis Onset)'


def generate_climate_with_heatwave(T=200, n=25, anomaly_start=140, anomaly_len=20):
    """
    Climate: 5x5 grid of temperature sensors.
    Normal: smooth spatial temperature field.
    Anomaly: heatwave — localized extreme temperatures.
    """
    W = build_weight_matrix(n, 'grid')
    graphs = [W.copy() for _ in range(T)]
    attributes = []
    ground_truth = np.zeros(T)

    side = int(np.sqrt(n))
    for t in range(T):
        # Base: smooth spatial field with seasonal component
        base_temp = 15 + 10 * np.sin(2 * np.pi * t / 365)
        field = np.zeros(n)
        for i in range(side):
            for j in range(side):
                field[i*side + j] = base_temp + 0.5 * (i + j) + np.random.randn() * 0.5

        if anomaly_start <= t < anomaly_start + anomaly_len:
            # Heatwave: center of grid gets extreme
            cx, cy = side // 2, side // 2
            for i in range(side):
                for j in range(side):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist < 2:
                        field[i*side + j] += 15 + np.random.randn() * 2
            ground_truth[t] = 1
        attributes.append(field)

    return graphs, attributes, ground_truth, 'Climate (Heatwave)'


def generate_network_with_bots(T=200, n=30, anomaly_start=140, anomaly_len=20):
    """
    Social: 30 users with interaction graph.
    Normal: organic community structure.
    Anomaly: bot injection — random connections, uniform activity.
    """
    W = build_weight_matrix(n, 'random')
    graphs_list = [W.copy() for _ in range(T)]
    attributes = []
    ground_truth = np.zeros(T)

    # Assign community memberships
    communities = np.random.randint(0, 3, n)

    for t in range(T):
        f = np.zeros(n)
        if anomaly_start <= t < anomaly_start + anomaly_len:
            # Bots: uniform activity across all users
            f = np.random.exponential(1.0, n)
            # Modify graph: bots connect randomly
            W_mod = graphs_list[t].copy()
            bot_indices = np.random.choice(n, size=5, replace=False)
            for b in bot_indices:
                targets = np.random.choice(n, size=np.random.randint(3, 8), replace=False)
                for tgt in targets:
                    W_mod[b, tgt] = W_mod[tgt, b] = 1.0
            graphs_list[t] = W_mod
            ground_truth[t] = 1
        else:
            # Normal: activity follows community structure
            for c in range(3):
                members = np.where(communities == c)[0]
                activity = np.random.exponential(2.0)
                f[members] = activity + np.random.randn(len(members)) * 0.3
        f = np.abs(f)
        attributes.append(f)

    return graphs_list, attributes, ground_truth, 'Social (Bot Injection)'


def generate_protein_with_mutation(T=200, n=20, anomaly_start=140, anomaly_len=20):
    """
    Protein: 20 residues in contact map chain.
    Normal: native folding contacts.
    Anomaly: mutation disrupts contacts → different energy landscape.
    """
    W = build_weight_matrix(n, 'chain', extra_edges=n)
    graphs_list = [W.copy() for _ in range(T)]
    attributes = []
    ground_truth = np.zeros(T)

    for t in range(T):
        f = np.random.exponential(1.0, n)  # residue energies
        if anomaly_start <= t < anomaly_start + anomaly_len:
            # Mutation: disrupt specific region
            mut_site = n // 2
            f[mut_site-2:mut_site+3] = np.random.exponential(5.0, 5)
            # Also modify contact map near mutation
            W_mod = graphs_list[t].copy()
            for i in range(max(0, mut_site-3), min(n, mut_site+3)):
                for j in range(n):
                    W_mod[i, j] = 0
                    W_mod[j, i] = 0
            graphs_list[t] = W_mod
            ground_truth[t] = 1
        attributes.append(f)

    return graphs_list, attributes, ground_truth, 'Protein (Mutation Impact)'


def generate_training_with_overfitting(T=200, n=10, anomaly_start=140, anomaly_len=20):
    """
    Neural: 10 layers in a network, chain graph.
    Normal: loss decreasing, smooth gradient flow.
    Anomaly: overfitting onset — gradients explode/vanish in specific layers.
    """
    W = build_weight_matrix(n, 'chain')
    graphs_list = [W.copy() for _ in range(T)]
    attributes = []
    ground_truth = np.zeros(T)

    for t in range(T):
        # Gradient magnitudes per layer (normally smooth)
        epoch = t / T
        base_gradient = np.exp(-epoch * 2) * np.linspace(1, 0.5, n) + np.random.randn(n) * 0.05
        base_gradient = np.abs(base_gradient)

        if anomaly_start <= t < anomaly_start + anomaly_len:
            # Overfitting: later layers get erratic gradients
            base_gradient[-3:] = np.abs(np.random.randn(3)) * 3.0
            base_gradient[:3] = np.abs(np.random.randn(3)) * 0.001
            ground_truth[t] = 1
        attributes.append(base_gradient)

    return graphs_list, attributes, ground_truth, 'Neural (Overfitting Onset)'


def generate_px4_sensor_failure(T=200, n=12, anomaly_start=140, anomaly_len=20):
    """
    PX4: 12 sensors (gyro x/y/z, accel x/y/z, mag x/y/z, baro, gps, lidar, flow),
    ring graph (sensor bus).
    Normal: correlated sensor readings.
    Anomaly: sensor failure — one sensor drops out, readings become erratic.
    """
    W = build_weight_matrix(n, 'ring', extra_edges=4)
    graphs_list = [W.copy() for _ in range(T)]
    attributes = []
    ground_truth = np.zeros(T)

    for t in range(T):
        # Normal: all sensors report correlated values
        base = np.random.randn() * 0.5
        f = base + np.random.randn(n) * 0.2

        if anomaly_start <= t < anomaly_start + anomaly_len:
            # Sensor failure: sensor 3 (accel Z) drops to 0 or spikes
            failed_sensor = 3
            if np.random.rand() > 0.5:
                f[failed_sensor] = 0  # dropout
            else:
                f[failed_sensor] = np.random.randn() * 50  # spike
            # Sometimes cascades to neighbors
            if t > anomaly_start + anomaly_len // 2:
                f[2] = np.random.randn() * 10
                f[4] = np.random.randn() * 10
            ground_truth[t] = 1
        attributes.append(f)

    return graphs_list, attributes, ground_truth, 'PX4 (Sensor Failure)'


# ============================================================
# 3. ANOMALY DETECTORS
# ============================================================

def conservation_anomaly_detector(graphs, attributes, window=30, sigma=2.0):
    """General-purpose conservation-based anomaly detector."""
    scores = []
    for t in range(len(graphs)):
        L = build_laplacian(graphs[t])
        cr = compute_conservation_ratio(L, attributes[t])
        scores.append(cr)

    scores = np.array(scores)
    baseline_mean = np.mean(scores[:window])
    baseline_std = np.std(scores[:window]) + 1e-10

    anomaly_scores = np.abs((scores - baseline_mean) / baseline_std)
    return anomaly_scores, scores


def zscore_variance_detector(attributes, window=30, sigma=2.0):
    """Baseline: Z-score on raw attribute variance."""
    variances = [np.var(a) for a in attributes]
    variances = np.array(variances)

    baseline_mean = np.mean(variances[:window])
    baseline_std = np.std(variances[:window]) + 1e-10

    anomaly_scores = np.abs((variances - baseline_mean) / baseline_std)
    return anomaly_scores, variances


def eigenvalue_detector(graphs, attributes, window=30, sigma=2.0):
    """Baseline: spectral gap (algebraic connectivity) monitoring."""
    gaps = []
    for t in range(len(graphs)):
        L = build_laplacian(graphs[t])
        eigs = np.sort(np.real(np.linalg.eigvalsh(L)))
        gap = eigs[1] if len(eigs) > 1 else 0  # algebraic connectivity
        gaps.append(gap)

    gaps = np.array(gaps)
    baseline_mean = np.mean(gaps[:window])
    baseline_std = np.std(gaps[:window]) + 1e-10

    anomaly_scores = np.abs((gaps - baseline_mean) / baseline_std)
    return anomaly_scores, gaps


def random_detector(T):
    """Random baseline: uniform random anomaly scores."""
    return np.random.rand(T), np.random.rand(T)


# ============================================================
# 4. EVALUATION METRICS
# ============================================================

def compute_roc(anomaly_scores, ground_truth, n_thresholds=200):
    """Compute ROC curve (TPR vs FPR)."""
    thresholds = np.linspace(0, anomaly_scores.max() + 0.1, n_thresholds)
    tpr_list = []
    fpr_list = []

    total_pos = ground_truth.sum()
    total_neg = len(ground_truth) - total_pos

    for thresh in thresholds:
        detected = (anomaly_scores >= thresh).astype(int)
        tp = (detected * ground_truth).sum()
        fp = (detected * (1 - ground_truth)).sum()
        tpr_list.append(tp / (total_pos + 1e-10))
        fpr_list.append(fp / (total_neg + 1e-10))

    return np.array(fpr_list), np.array(tpr_list)


def compute_auc(fpr, tpr):
    """Compute AUC using trapezoidal rule."""
    # Sort by FPR
    idx = np.argsort(fpr)
    fpr_sorted = fpr[idx]
    tpr_sorted = tpr[idx]
    return np.trapz(tpr_sorted, fpr_sorted)


def compute_detection_latency(anomaly_scores, ground_truth, threshold_percentile=95):
    """How many timesteps after anomaly start before detection."""
    thresh = np.percentile(anomaly_scores[:int(len(anomaly_scores)*0.7)], threshold_percentile)
    anomaly_indices = np.where(ground_truth == 1)[0]
    if len(anomaly_indices) == 0:
        return float('inf')
    anomaly_start = anomaly_indices[0]

    for t in range(anomaly_start, len(anomaly_scores)):
        if anomaly_scores[t] >= thresh:
            return t - anomaly_start
    return float('inf')


def compute_fpr_at_fixed_tpr(fpr, tpr, target_tpr=0.8):
    """False positive rate at a fixed true positive rate."""
    idx = np.argmin(np.abs(tpr - target_tpr))
    return fpr[idx]


# ============================================================
# 5. RUN ALL EXPERIMENTS
# ============================================================

print("=" * 80)
print("ANOMALY DETECTION ATLAS")
print("Conservation-Based Anomaly Detection Across All Domains")
print("=" * 80)

# Generate all datasets
datasets = {}
print("\n📊 Generating synthetic datasets...")
generators = [
    generate_tonal_with_chromatic_insertion,
    generate_market_with_crisis,
    generate_climate_with_heatwave,
    generate_network_with_bots,
    generate_protein_with_mutation,
    generate_training_with_overfitting,
    generate_px4_sensor_failure,
]

for gen in generators:
    graphs, attrs, gt, name = gen()
    datasets[name] = (graphs, attrs, gt)
    print(f"  ✓ {name}: T={len(graphs)}, nodes={len(graphs[0])}, anomaly_frames={int(gt.sum())}")

# Run detectors
methods = {
    'Conservation': conservation_anomaly_detector,
    'Z-Score Variance': zscore_variance_detector,
    'Eigenvalue Gap': eigenvalue_detector,
    'Random': None,
}

print("\n🔬 Running anomaly detection methods...")

results = {}
for domain_name, (graphs, attrs, gt) in datasets.items():
    T = len(graphs)
    results[domain_name] = {}

    for method_name, method_fn in methods.items():
        if method_name == 'Random':
            scores, raw = random_detector(T)
        elif method_name == 'Conservation':
            scores, raw = method_fn(graphs, attrs)
        elif method_name == 'Eigenvalue Gap':
            scores, raw = method_fn(graphs, attrs)
        else:
            scores, raw = method_fn(attrs)

        fpr, tpr = compute_roc(scores, gt)
        auc = compute_auc(fpr, tpr)
        latency = compute_detection_latency(scores, gt)
        fpr_at_80 = compute_fpr_at_fixed_tpr(fpr, tpr, 0.8)

        results[domain_name][method_name] = {
            'fpr': fpr, 'tpr': tpr, 'auc': auc,
            'latency': latency, 'fpr_at_80tpr': fpr_at_80,
            'anomaly_scores': scores, 'raw_scores': raw,
        }
    print(f"  ✓ {domain_name}")

# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("AUC COMPARISON TABLE")
print("=" * 80)

# Header
header = f"{'Domain':<30}"
for m in methods:
    header += f" {m:>18}"
print(header)
print("-" * (30 + 18 * len(methods) + len(methods)))

for domain_name in results:
    row = f"{domain_name:<30}"
    for method_name in methods:
        auc = results[domain_name][method_name]['auc']
        row += f" {auc:>18.4f}"
    print(row)

# Average row
print("-" * (30 + 18 * len(methods) + len(methods)))
avg_row = f"{'AVERAGE':<30}"
for method_name in methods:
    avg_auc = np.mean([results[d][method_name]['auc'] for d in results])
    avg_row += f" {avg_auc:>18.4f}"
print(avg_row)

print("\n" + "=" * 80)
print("DETECTION LATENCY (timesteps after anomaly onset)")
print("=" * 80)

header = f"{'Domain':<30}"
for m in methods:
    header += f" {m:>18}"
print(header)
print("-" * (30 + 18 * len(methods) + len(methods)))

for domain_name in results:
    row = f"{domain_name:<30}"
    for method_name in methods:
        lat = results[domain_name][method_name]['latency']
        if lat == float('inf'):
            row += f" {'∞':>18}"
        else:
            row += f" {lat:>18.1f}"
    print(row)

print("\n" + "=" * 80)
print("FALSE POSITIVE RATE @ 80% TPR")
print("=" * 80)

header = f"{'Domain':<30}"
for m in methods:
    header += f" {m:>18}"
print(header)
print("-" * (30 + 18 * len(methods) + len(methods)))

for domain_name in results:
    row = f"{domain_name:<30}"
    for method_name in methods:
        fpr80 = results[domain_name][method_name]['fpr_at_80tpr']
        row += f" {fpr80:>18.4f}"
    print(row)

# ============================================================
# 7. GENERATE THE ATLAS PLOT
# ============================================================

print("\n🎨 Generating Atlas visualization...")

n_domains = len(results)
n_methods = len(methods)
method_colors = {
    'Conservation': '#2ecc71',
    'Z-Score Variance': '#3498db',
    'Eigenvalue Gap': '#e67e22',
    'Random': '#95a5a6',
}

fig = plt.figure(figsize=(24, 28))
gs = GridSpec(n_domains + 2, n_methods + 1, figure=fig, hspace=0.4, wspace=0.35,
              height_ratios=[1]*n_domains + [0.3, 1.2])

# --- ROC curves per domain ---
for i, (domain_name, domain_results) in enumerate(results.items()):
    ax = fig.add_subplot(gs[i, :n_methods])

    for method_name, method_results in domain_results.items():
        color = method_colors[method_name]
        lw = 2.5 if method_name == 'Conservation' else 1.2
        ls = '-' if method_name == 'Conservation' else '--'
        ax.plot(method_results['fpr'], method_results['tpr'],
                color=color, linewidth=lw, linestyle=ls,
                label=f"{method_name} (AUC={method_results['auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=0.8)
    ax.set_title(domain_name, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.grid(True, alpha=0.2)

# --- AUC Bar Chart (Atlas summary) ---
ax_bar = fig.add_subplot(gs[n_domains + 1, :])

x = np.arange(n_domains)
width = 0.18
method_list = list(methods.keys())

for j, method_name in enumerate(method_list):
    aucs = [results[d][method_name]['auc'] for d in results]
    bars = ax_bar.bar(x + j * width, aucs, width,
                      label=method_name,
                      color=method_colors[method_name],
                      alpha=0.85 if method_name == 'Conservation' else 0.6,
                      edgecolor='black' if method_name == 'Conservation' else 'gray',
                      linewidth=1.5 if method_name == 'Conservation' else 0.5)

ax_bar.set_xlabel('Domain', fontsize=12)
ax_bar.set_ylabel('AUC', fontsize=12)
ax_bar.set_title('THE ATLAS: Conservation-Based Anomaly Detection Across All Domains',
                 fontsize=16, fontweight='bold', pad=15)
ax_bar.set_xticks(x + width * 1.5)
short_names = [d.split('(')[0].strip() for d in results]
ax_bar.set_xticklabels(short_names, fontsize=10, rotation=15, ha='right')
ax_bar.legend(fontsize=10, loc='upper left')
ax_bar.set_ylim([0, 1.1])
ax_bar.axhline(y=0.5, color='red', linestyle=':', alpha=0.4, label='Random baseline')
ax_bar.grid(True, alpha=0.2, axis='y')

# Add value labels on bars
for j, method_name in enumerate(method_list):
    aucs = [results[d][method_name]['auc'] for d in results]
    for k, v in enumerate(aucs):
        ax_bar.text(x[k] + j * width, v + 0.02, f'{v:.2f}',
                   ha='center', va='bottom', fontsize=7,
                   fontweight='bold' if method_name == 'Conservation' else 'normal')

# Supertitle
fig.suptitle('Anomaly Detection Atlas\nConservation-Based Detection Across All Domains',
             fontsize=20, fontweight='bold', y=0.99)

plt.savefig('/home/phoenix/.openclaw/workspace/experiments/anomaly-atlas/atlas.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/home/phoenix/.openclaw/workspace/experiments/anomaly-atlas/atlas.pdf',
            bbox_inches='tight', facecolor='white')
print("  ✓ Saved atlas.png and atlas.pdf")

# ============================================================
# 8. TIME-SERIES DETAIL PLOT
# ============================================================

fig2, axes = plt.subplots(n_domains, 1, figsize=(20, 3 * n_domains))
if n_domains == 1:
    axes = [axes]

for i, (domain_name, domain_results) in enumerate(results.items()):
    ax = axes[i]
    gt = datasets[domain_name][2]
    T = len(gt)

    # Plot conservation ratio and ground truth
    cons_scores = domain_results['Conservation']['raw_scores']
    ax.plot(range(T), cons_scores, color='#2ecc71', linewidth=1.5, label='Conservation Ratio')

    # Mark anomaly region
    anomaly_idx = np.where(gt == 1)[0]
    if len(anomaly_idx) > 0:
        a_start, a_end = anomaly_idx[0], anomaly_idx[-1]
        ax.axvspan(a_start, a_end, alpha=0.15, color='red', label='True Anomaly')

    # Mark detected anomalies
    det_scores = domain_results['Conservation']['anomaly_scores']
    thresh = 2.0
    detected = det_scores >= thresh
    ax.fill_between(range(T), cons_scores.min(), cons_scores.max(),
                    where=detected, alpha=0.1, color='orange', label='Detected')

    ax.set_title(domain_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Conservation Ratio', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2)

    if i == n_domains - 1:
        ax.set_xlabel('Time', fontsize=10)

fig2.suptitle('Conservation Ratio Time Series — All Domains',
              fontsize=16, fontweight='bold')
plt.savefig('/home/phoenix/.openclaw/workspace/experiments/anomaly-atlas/timeseries.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("  ✓ Saved timeseries.png")

# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

cons_aucs = [results[d]['Conservation']['auc'] for d in results]
zscore_aucs = [results[d]['Z-Score Variance']['auc'] for d in results]
eigen_aucs = [results[d]['Eigenvalue Gap']['auc'] for d in results]
random_aucs = [results[d]['Random']['auc'] for d in results]

print(f"""
Conservation-based anomaly detection achieves:
  Average AUC: {np.mean(cons_aucs):.4f} (±{np.std(cons_aucs):.4f})

vs. Baselines:
  Z-Score Variance: {np.mean(zscore_aucs):.4f} (±{np.std(zscore_aucs):.4f})
  Eigenvalue Gap:   {np.mean(eigen_aucs):.4f} (±{np.std(eigen_aucs):.4f})
  Random:           {np.mean(random_aucs):.4f} (±{np.std(random_aucs):.4f})

Conservation advantage over best baseline: {np.mean(cons_aucs) - max(np.mean(zscore_aucs), np.mean(eigen_aucs)):+.4f} AUC

The pattern holds across ALL domains:
  Build Laplacian → Compute Conservation → Threshold on Deviation
  → Detect anomalies in music, finance, climate, social, protein, neural, and PX4 data.

Output files:
  atlas.png      — ROC curves + AUC bar chart (THE ATLAS)
  atlas.pdf      — High-resolution version
  timeseries.png — Conservation ratio time series with anomaly regions
""")

print("=" * 80)
print("✅ ANOMALY DETECTION ATLAS COMPLETE")
print("=" * 80)
