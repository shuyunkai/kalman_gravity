"""核心算法单元测试：Parker闭环、KF平滑、VGG、挠曲响应、评估指标。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def test_parker_roundtrip():
    """验证 Parker 正演→反演闭环一致性。"""
    from kalman_gravity import KalmanBathymetryInverter

    n = 64
    inverter = KalmanBathymetryInverter(grid_shape=(n, n), region_size_deg=2.0, lat_center=0)
    np.random.seed(42)
    bathy = np.random.randn(n, n) * 500 - 3000

    grav = inverter.gravity_forward(bathy, d=3000, drho=1700)
    inverter.apply_flexural_isostasy(Te=10e3)
    bathy_rec = inverter.gravity_to_bathymetry_linear(
        grav, d=3000, drho=1700, k_min=2 * np.pi / 160000, k_max=2 * np.pi / 2000)

    corr = np.corrcoef(bathy.ravel(), bathy_rec.ravel())[0, 1]
    assert corr > 0.5, f"Parker roundtrip correlation too low: {corr:.3f}"
    print(f"  PASS: Parker roundtrip r={corr:.3f}")


def test_kalman_gravity_smoothing():
    """验证 KF 重力场重建输出有效。"""
    from kalman_gravity import KalmanGravityReconstructor

    n_lr, upscale = 32, 2
    n_hr = n_lr * upscale
    np.random.seed(42)
    xx, yy = np.meshgrid(np.linspace(0, 1, n_lr), np.linspace(0, 1, n_lr))
    grav_lr = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) * 50
               + np.random.randn(n_lr, n_lr) * 2)

    recon = KalmanGravityReconstructor(
        grid_shape=(n_hr, n_hr), region_size_deg=2.0, lr_grid_shape=(n_lr, n_lr))
    recon.initialize(gravity_lr=grav_lr, process_noise_std=2.0,
                     sat_noise_std=3.0, ship_noise_std=1.0)
    grav_hr = recon.run(n_iterations=3, smooth_sigma=0.8, verbose=False)

    assert grav_hr.shape == (n_hr, n_hr), f"Wrong HR shape: {grav_hr.shape}"
    assert np.all(np.isfinite(grav_hr)), "Non-finite values in output"
    assert abs(grav_hr.mean() - grav_lr.mean()) < 3 * grav_lr.std(), "Output mean drifts too far"
    print(f"  PASS: KF gravity smoothing shape={grav_hr.shape}")


def test_vgg_computation():
    """验证频域 VGG 计算结果有效。"""
    from utils import compute_vgg

    n = 64
    np.random.seed(42)
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    grav = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) * 50
    vgg = compute_vgg(grav, dx_deg=2.0 / n, lat_center=0)

    assert vgg.shape == grav.shape, "VGG shape mismatch"
    assert np.all(np.isfinite(vgg)), "Non-finite VGG values"
    assert vgg.std() > 0, "VGG is flat"
    print(f"  PASS: VGG computation std={vgg.std():.2f}")


def test_flexural_response():
    """验证挠曲均衡响应函数单调有界。"""
    from kalman_gravity import KalmanBathymetryInverter

    n = 64
    inverter = KalmanBathymetryInverter(grid_shape=(n, n), region_size_deg=2.0, lat_center=0)
    inverter.apply_flexural_isostasy(Te=10e3)

    phi = inverter.flexural_phi
    assert phi.min() >= 0, f"Phi min {phi.min():.3f} < 0"
    assert phi.max() <= 1.0, f"Phi max {phi.max():.3f} > 1"
    assert phi[0, 0] < 0.5, f"DC flexural response should be < 0.5: {phi[0,0]:.3f}"
    assert phi[-1, -1] > phi[0, 0] * 1.5, \
        f"High-k ({phi[-1,-1]:.3f}) should exceed DC ({phi[0,0]:.3f})"
    print(f"  PASS: Flexural response phi(0)={phi[0,0]:.3f}, phi(high)={phi[-1,-1]:.3f}")


def test_metrics_calculation():
    """验证评估指标计算结果合理。"""
    from utils import compute_metrics

    np.random.seed(42)
    truth = np.random.randn(100) * 10 + 50
    pred = truth + np.random.randn(100) * 3
    metrics = compute_metrics(pred, truth)

    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "Correlation" in metrics
    assert metrics["RMSE"] > 0
    assert metrics["Correlation"] > 0.5
    print(f"  PASS: Metrics RMSE={metrics['RMSE']:.2f}, r={metrics['Correlation']:.3f}")


if __name__ == "__main__":
    print("Running core algorithm tests...")
    print("=" * 50)
    tests = [test_parker_roundtrip, test_kalman_gravity_smoothing,
             test_vgg_computation, test_flexural_response, test_metrics_calculation]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
