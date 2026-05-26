"""工具函数：评估指标计算、VGG计算、频域滤波、数据插值、可视化。"""
import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")

fm._load_fontmanager(try_read_cache=False)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def compute_metrics(pred, truth, mask=None):
    """计算预测值与真值之间的多项评估指标。"""
    if mask is not None:
        pred = pred[mask]
        truth = truth[mask]
    else:
        pred = pred.ravel()
        truth = truth.ravel()

    diff = pred - truth
    valid = ~(np.isnan(diff) | np.isinf(diff))
    diff = diff[valid]
    pred_v = pred[valid]
    truth_v = truth[valid]

    n = len(diff)
    if n == 0:
        return {}

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    mre = np.mean(np.abs(diff / (np.maximum(np.abs(truth_v), 1e-6)))) * 100
    std = np.std(diff)
    mean_bias = np.mean(diff)
    corr = np.corrcoef(pred_v, truth_v)[0, 1] if n > 1 else 0.0

    signal_power = np.var(truth_v)
    noise_power = np.var(diff)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    mu_x, mu_y = pred_v.mean(), truth_v.mean()
    sig_x, sig_y = pred_v.std(), truth_v.std()
    sig_xy = np.mean((pred_v - mu_x) * (truth_v - mu_y))
    c1, c2 = 1e-4, 9e-4
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x ** 2 + sig_y ** 2 + c2)
    )

    return {
        "RMSE": rmse, "MAE": mae, "MRE_%": mre, "STD": std,
        "Bias": mean_bias, "Correlation": corr,
        "SNR_dB": snr, "SSIM": ssim, "N": n,
    }


def print_metrics(metrics, title="Metrics"):
    """格式化打印评估指标。"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if v > 100:
                print(f"  {k:15s}: {v:12.2f}")
            else:
                print(f"  {k:15s}: {v:12.4f}")
        else:
            print(f"  {k:15s}: {v!s:>12}")
    print(f"{'='*50}")


def compute_vgg(gravity, dx_deg, lat_center=0):
    """频域垂直重力梯度: VGG = dg/dz = F^{-1}[|k|*F[g]]。含纬度校正。"""
    ny, nx = gravity.shape
    cos_lat = np.cos(np.radians(lat_center))
    dx_m = dx_deg * 111320 * cos_lat
    dy_m = dx_deg * 111320
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx_m)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy_m)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    k[0, 0] = 1e-10

    G_fft = np.fft.fft2(gravity)
    VGG_fft = G_fft * k
    vgg = np.real(np.fft.ifft2(VGG_fft))
    return vgg


def bandpass_filter(data, dx_km, lam_min=None, lam_max=None):
    """频域带通滤波：保留 lam_min~lam_max 波长范围的信号。"""
    ny, nx = data.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx_km)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dx_km)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    filt = np.ones_like(k)
    if lam_min is not None:
        k_highpass = 2 * np.pi / lam_min
        filt[k < k_highpass] = 0
    if lam_max is not None:
        k_lowpass = 2 * np.pi / lam_max
        filt[k > k_lowpass] = 0

    data_fft = np.fft.fft2(data)
    filtered = np.real(np.fft.ifft2(data_fft * filt))
    return filtered


def interpolate_nans(data, method="nearest"):
    """用最近邻插值填充数据中的 NaN 值。"""
    mask = np.isnan(data)
    if not mask.any():
        return data
    yy, xx = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij")
    points = np.column_stack([yy[~mask], xx[~mask]])
    values = data[~mask]
    grid_points = np.column_stack([yy[mask], xx[mask]])
    data_filled = data.copy()
    data_filled[mask] = griddata(points, values, grid_points, method=method)
    return data_filled


def plot_gravity_field(data_list, titles, save_path=None, figsize=(18, 5)):
    """并排绘制多个重力场图。"""
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (data, title) in enumerate(zip(data_list, titles)):
        ax = axes[i]
        im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto", origin="lower")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (格网点)")
        ax.set_ylabel("Y (格网点)")
        plt.colorbar(im, ax=ax, label="mGal", shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图保存至: {save_path}")
    plt.close()


def plot_bathymetry(data_list, titles, save_path=None, figsize=(18, 5)):
    """并排绘制多个水深图。"""
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (data, title) in enumerate(zip(data_list, titles)):
        ax = axes[i]
        im = ax.imshow(data, cmap="Blues_r", aspect="auto", origin="lower")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (格网点)")
        ax.set_ylabel("Y (格网点)")
        plt.colorbar(im, ax=ax, label="深度 (m)", shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图保存至: {save_path}")
    plt.close()


def plot_residual(pred, truth, title="Residual", save_path=None, vmin=-50, vmax=50):
    """绘制预测值与真值的残差图。"""
    residual = pred - truth
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(residual, cmap="RdBu_r", aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X (格网点)")
    ax.set_ylabel("Y (格网点)")
    plt.colorbar(im, ax=ax, label="mGal / m", shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"残差图保存至: {save_path}")
    plt.close()


def plot_scatter_compare(pred, truth, ax=None, title="预测值 vs 真实值", alpha=0.3):
    """绘制预测值与真值的散点对比图。"""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))
    mask = ~(np.isnan(pred) | np.isnan(truth))
    p, t = pred[mask].ravel(), truth[mask].ravel()
    ax.scatter(t, p, alpha=alpha, s=2)
    rng = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(rng, rng, "r--", lw=1.5, label="y=x")
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.set_title(title)
    ax.legend()
    rmse = np.sqrt(np.mean((p - t) ** 2))
    corr = np.corrcoef(p, t)[0, 1]
    ax.text(0.05, 0.95, f"RMSE={rmse:.2f}\nR={corr:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    return ax


def plot_error_histogram(pred, truth, title="Error Distribution", save_path=None):
    """绘制误差分布直方图（含正态拟合曲线）。"""
    diff = (pred - truth).ravel()
    diff = diff[~np.isnan(diff)]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(diff, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", lw=1.5, label="零线")
    ax.axvline(x=diff.mean(), color="orange", linestyle="-", lw=1.5,
               label=f"均值={diff.mean():.2f}")
    mu, std = diff.mean(), diff.std()
    x_vals = np.linspace(diff.min(), diff.max(), 200)
    y_vals = np.exp(-(x_vals - mu) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
    ax.plot(x_vals, y_vals, "k-", lw=1.5, label=f"N({mu:.2f}, {std:.2f})")
    ax.set_xlabel("误差")
    ax.set_ylabel("概率密度")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
