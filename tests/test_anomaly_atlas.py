"""Tests for Anomaly Detection Atlas."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from anomaly_atlas import (
    build_laplacian, compute_conservation_ratio, build_weight_matrix,
    conservation_anomaly_detector, zscore_variance_detector,
    compute_roc, compute_auc, compute_detection_latency,
    generate_tonal_with_chromatic_insertion,
    generate_market_with_crisis,
    generate_climate_with_heatwave,
    generate_network_with_bots,
    generate_protein_with_mutation,
    generate_training_with_overfitting,
    generate_px4_sensor_failure,
)


class TestLaplacian:
    def test_build_laplacian_chain(self):
        W = build_weight_matrix(4, 'chain')
        L = build_laplacian(W)
        assert L.shape == (4, 4)
        # Normalized Laplacian diagonal should be close to 1 for non-isolated nodes
        assert np.allclose(np.diag(L), np.array([1.0, 1.0, 1.0, 1.0]), atol=0.01)

    def test_build_laplacian_symmetric(self):
        W = build_weight_matrix(6, 'ring')
        L = build_laplacian(W)
        assert np.allclose(L, L.T)

    def test_build_laplacian_psd(self):
        W = build_weight_matrix(5, 'chain', extra_edges=2)
        L = build_laplacian(W)
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-10)  # PSD


class TestConservationRatio:
    def test_smooth_signal_high_conservation(self):
        n = 10
        W = build_weight_matrix(n, 'chain')
        L = build_laplacian(W)
        # Smooth signal: linear gradient
        f = np.linspace(1, 0, n)
        cr = compute_conservation_ratio(L, f)
        assert cr > 0.5  # Smooth should be high conservation

    def test_noisy_signal_lower_conservation(self):
        n = 10
        W = build_weight_matrix(n, 'chain')
        L = build_laplacian(W)
        # Noisy signal
        np.random.seed(42)
        f = np.random.randn(n)
        cr_noisy = compute_conservation_ratio(L, f)
        f_smooth = np.linspace(1, 0, n)
        cr_smooth = compute_conservation_ratio(L, f_smooth)
        assert cr_smooth >= cr_noisy


class TestWeightMatrix:
    def test_chain_structure(self):
        W = build_weight_matrix(5, 'chain')
        assert W.shape == (5, 5)
        assert W[0, 1] == 1.0
        assert W[0, 2] == 0.0  # Not connected

    def test_ring_structure(self):
        W = build_weight_matrix(5, 'ring')
        assert W[0, 4] == 1.0  # Wrap-around
        assert W[4, 0] == 1.0

    def test_grid_structure(self):
        W = build_weight_matrix(9, 'grid')  # 3x3 grid
        assert W.shape == (9, 9)
        # Corner node should have 2 connections
        assert np.sum(W[0] > 0) == 2

    def test_symmetric(self):
        W = build_weight_matrix(8, 'random')
        assert np.allclose(W, W.T)

    def test_extra_edges(self):
        W = build_weight_matrix(5, 'chain', extra_edges=3)
        # More non-zero entries than plain chain
        W_plain = build_weight_matrix(5, 'chain')
        assert np.sum(W > 0) >= np.sum(W_plain > 0)


class TestGenerators:
    @pytest.mark.parametrize("gen", [
        generate_tonal_with_chromatic_insertion,
        generate_market_with_crisis,
        generate_climate_with_heatwave,
        generate_network_with_bots,
        generate_protein_with_mutation,
        generate_training_with_overfitting,
        generate_px4_sensor_failure,
    ])
    def test_generator_output(self, gen):
        graphs, attrs, gt, name = gen()
        assert len(graphs) == len(attrs) == len(gt)
        assert gt.sum() > 0  # Has anomalies
        assert isinstance(name, str)

    def test_generators_have_anomaly_region(self):
        for gen in [generate_tonal_with_chromatic_insertion, generate_market_with_crisis]:
            _, _, gt, _ = gen()
            # Anomaly should be in later portion
            assert gt[:100].sum() < gt.sum()


class TestDetectors:
    def test_conservation_detector(self):
        graphs, attrs, gt, _ = generate_px4_sensor_failure()
        scores, raw = conservation_anomaly_detector(graphs, attrs)
        assert len(scores) == len(gt)
        assert np.all(scores >= 0)

    def test_zscore_detector(self):
        graphs, attrs, gt, _ = generate_market_with_crisis()
        scores, raw = zscore_variance_detector(attrs)
        assert len(scores) == len(gt)


class TestMetrics:
    def test_compute_roc(self):
        np.random.seed(42)
        scores = np.random.rand(200)
        gt = np.zeros(200)
        gt[140:160] = 1
        fpr, tpr = compute_roc(scores, gt)
        assert len(fpr) == len(tpr)
        assert len(fpr) > 0  # Has thresholds

    def test_compute_auc_perfect(self):
        fpr = np.array([0, 0, 1])
        tpr = np.array([0, 1, 1])
        auc = compute_auc(fpr, tpr)
        assert auc == 1.0

    def test_compute_auc_random(self):
        fpr = np.array([0, 0.5, 1])
        tpr = np.array([0, 0.5, 1])
        auc = compute_auc(fpr, tpr)
        assert abs(auc - 0.5) < 0.01

    def test_detection_latency(self):
        scores = np.zeros(200)
        scores[150] = 100  # Detect at 150
        gt = np.zeros(200)
        gt[140:160] = 1
        latency = compute_detection_latency(scores, gt, threshold_percentile=95)
        assert isinstance(latency, (int, float, np.integer, np.floating))
