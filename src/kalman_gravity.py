"""核心算法: 卡尔曼滤波重力场重建.

状态变量 x: 高分辨率格网 (ny_hr, nx_hr)
观测1 z_sat: 低分辨率卫星重力 (ny_lr, nx_lr), 观测模型为块均值
观测2 z_ship: 稀疏船载重力, 观测模型为点采样
状态转移: 空间高斯平滑, 模拟重力场连续性先验
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")


class KalmanGravityReconstructor:
    """标准卡尔曼滤波：融合卫星(1')和船载(15'')重力数据，重建高分辨率重力场。

    状态变量 x: 高分辨率格网 (ny*upscale, nx*upscale)
    观测1 z_sat: 低分辨率卫星重力 (ny, nx)，观测模型为块均值
    观测2 z_ship: 稀疏船载重力，观测模型为点采样
    状态转移: 空间高斯平滑，模拟重力场连续性先验
    """

    def __init__(self, grid_shape, region_size_deg, lr_grid_shape=None,
                 localize=False, adaptive=False, corr_px=3.0, adaptive_ship=True):
        self.grid_shape = grid_shape
        self.ny, self.nx = grid_shape
        self.n_state = self.ny * self.nx
        self.region_size_deg = region_size_deg
        self.localize = localize
        self.adaptive = adaptive
        self.corr_px = float(corr_px)
        self.adaptive_ship = adaptive_ship

        if lr_grid_shape is None:
            self.lr_ny = max(1, int(self.ny / 4))
            self.lr_nx = max(1, int(self.nx / 4))
        else:
            self.lr_ny, self.lr_nx = lr_grid_shape

        # 4x4 块内平均相关系数（用于块观测的局部化增益）
        by = self.ny // self.lr_ny
        bx = self.nx // self.lr_nx
        yy, xx = np.mgrid[0:by, 0:bx]
        dd = np.hypot(yy - (by - 1) / 2.0, xx - (bx - 1) / 2.0)
        LL = np.exp(-0.5 * (dd / self.corr_px) ** 2)
        m = by * bx
        self.block_rho = float((LL.sum() - m) / max(m * (m - 1), 1))

        self.x = None
        self._initialized = False

    def initialize(self, gravity_lr, gravity_ship=None,
                   process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                   ship_train_mask=None):
        """初始化滤波器：上采样卫星数据作为初值，设定噪声参数。

        默认值即本项目实际使用的配置 (Q=4, R_sat=9, R_ship=1 mGal²)，与
        BathyPriorKF / three_method_compare 保持一致；原默认值 (1.0/2.0/0.5)
        与实跑值不同，直接调用会得到另一套方法。

        ship_train_mask: 船载训练/验证拆分掩码 (True=用于约束, False=保留验证).
                         避免船载数据既做初始状态又被重新观测的循环。
        """
        from scipy.ndimage import zoom
        zoom_y = self.ny / gravity_lr.shape[0]
        zoom_x = self.nx / gravity_lr.shape[1]
        x0 = zoom(gravity_lr, (zoom_y, zoom_x), order=1)

        if gravity_ship is not None:
            if ship_train_mask is not None:
                train = ship_train_mask & ~np.isnan(gravity_ship)
                x0[train] = gravity_ship[train]
            else:
                valid = ~np.isnan(gravity_ship)
                x0[valid] = gravity_ship[valid]

        self.x = x0.ravel()
        init_var = (process_noise_std ** 2) * np.ones(self.n_state)
        self.P_diag = init_var.copy()
        self.Q_diag = (process_noise_std ** 2) * np.ones(self.n_state)
        self.R_sat_val = sat_noise_std ** 2
        self.R_ship_val = ship_noise_std ** 2
        self.gravity_lr = gravity_lr
        self.gravity_ship = gravity_ship
        self.ship_train_mask = ship_train_mask
        self._initialized = True
        self._iter_count = 0

        print(f"卡尔曼滤波器初始化完成. 状态维度: {self.n_state}")
        print(f"  过程噪声 std: {process_noise_std:.2f} mGal")
        print(f"  卫星观测噪声 std: {sat_noise_std:.2f} mGal")
        print(f"  船载观测噪声 std: {ship_noise_std:.2f} mGal")

    def predict(self, smooth_sigma=1.0):
        """预测步：空间高斯平滑 + 协方差膨胀（模拟过程噪声）。"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化滤波器")

        x_2d = self.x.reshape(self.ny, self.nx)
        x_pred_2d = gaussian_filter(x_2d, sigma=smooth_sigma)
        self.x = x_pred_2d.ravel()

        factor = 1.0 + smooth_sigma * 0.1
        self.P_diag = self.P_diag * factor + self.Q_diag
        positive = self.P_diag[self.P_diag > 0]
        cap = 100.0 * (np.median(positive) + 1e-10) if positive.size > 0 else 1e10
        self.P_diag = np.minimum(self.P_diag, cap)

    def update_satellite(self):
        """更新步-卫星观测：用低分辨率格网约束高分辨率重建结果。

        每个低分辨率像元对应一块高分辨率像元，约束块均值接近卫星观测值。
        NOTE: 使用块均值 P 和标量近似 Kalman gain，导致块内 P 均一化。
        这是高维状态下的计算效率近似，块内各格点的空间不确定性信息会丢失。
        """
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化滤波器")

        x_2d = self.x.reshape(self.ny, self.nx)
        P_2d = self.P_diag.reshape(self.ny, self.nx)
        lr_ny, lr_nx = self.gravity_lr.shape
        hr_per_lr_y = self.ny // lr_ny
        hr_per_lr_x = self.nx // lr_nx

        if (self.localize and hr_per_lr_y * lr_ny == self.ny
                and hr_per_lr_x * lr_nx == self.nx):
            m = hr_per_lr_y * hr_per_lr_x
            fac = 1.0 + (m - 1) * self.block_rho
            X = x_2d.reshape(lr_ny, hr_per_lr_y, lr_nx, hr_per_lr_x)
            Pb = P_2d.reshape(lr_ny, hr_per_lr_y, lr_nx, hr_per_lr_x)
            P_bar = Pb.mean(axis=(1, 3))
            S = P_bar * fac / m + self.R_sat_val
            innovation = self.gravity_lr - X.mean(axis=(1, 3))
            K = Pb * fac / m / S[:, None, :, None]
            X += K * innovation[:, None, :, None]
            Pb *= (1.0 - K * fac / m)
            self.x = x_2d.ravel()
            self.P_diag = P_2d.ravel()
            return

        if hr_per_lr_y * lr_ny == self.ny and hr_per_lr_x * lr_nx == self.nx:
            # 矢量化路径: (ny, nx) -> (lr_ny, hy, lr_nx, hx)，一次算完全部块。
            # 20°x16° 区域是 1,152,000 个块, 原来的双重 Python 循环是这个流程最大的
            # 性能瓶颈(约占总耗时绝大部分), 等价改写后为个位数次 numpy 归约。
            X = x_2d.reshape(lr_ny, hr_per_lr_y, lr_nx, hr_per_lr_x)
            Pb = P_2d.reshape(lr_ny, hr_per_lr_y, lr_nx, hr_per_lr_x)
            z_pred = X.mean(axis=(1, 3))
            P_bar = Pb.mean(axis=(1, 3))
            K = P_bar / (P_bar + self.R_sat_val)
            innovation = self.gravity_lr - z_pred
            X += (K * innovation)[:, None, :, None]
            Pb[...] = ((1.0 - K) * P_bar)[:, None, :, None]
            self.x = x_2d.ravel()
            self.P_diag = P_2d.ravel()
            return

        for i in range(lr_ny):
            for j in range(lr_nx):
                i0 = i * hr_per_lr_y
                i1 = min((i + 1) * hr_per_lr_y, self.ny)
                j0 = j * hr_per_lr_x
                j1 = min((j + 1) * hr_per_lr_x, self.nx)

                block = x_2d[i0:i1, j0:j1]
                block_flat = block.ravel()
                z = self.gravity_lr[i, j]

                z_pred = block_flat.mean()
                P_block = P_2d[i0:i1, j0:j1].mean()
                K = P_block / (P_block + self.R_sat_val)
                innovation = z - z_pred
                block_flat_updated = block_flat + K * innovation
                x_2d[i0:i1, j0:j1] = block_flat_updated.reshape(block.shape)
                P_2d[i0:i1, j0:j1] = (1 - K) * P_block

        self.x = x_2d.ravel()
        self.P_diag = P_2d.ravel()  # explicit: P_2d is a view, ensure P_diag updated

    def update_shipborne(self, max_points_per_update=None, prop_radius=3,
                         prop_fraction=0.3, chunk_size=20000):
        """更新步-船载观测：逐点卡尔曼更新 + 高斯邻域传播。

        船载重力在每个测点上高精度观测，更新后通过邻域传播补偿对角协方差
        忽略的空间相关性。

        max_points_per_update:
            None（默认）= 使用**全部**有效观测点。
            原实现默认 5000，每轮随机抽 5000 个像素，6 轮下来只有约 37% 的训练
            像素被直接观测过；再叠加"先格网化、每格网只留一个点"的损失，实际进入
            滤波的原始船载测量值不足 7%。数据量极大确需限流时再传正整数。
        chunk_size:
            邻域传播分批累加的批大小，仅用于限制峰值内存，不影响结果。

        说明：本实现与原逐点循环在数学上完全等价（含边缘点窗口裁剪后重新归一化），
        只是改为向量化 + np.bincount 累加，因此可以负担全量观测点。
        """
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化滤波器")
        if self.gravity_ship is None:
            return

        valid_mask = ~np.isnan(self.gravity_ship)
        if self.ship_train_mask is not None:
            valid_mask = valid_mask & self.ship_train_mask
        valid_indices = np.where(valid_mask.ravel())[0]
        if len(valid_indices) == 0:
            print("  无有效船载观测点, 跳过更新")
            return

        if max_points_per_update is not None and len(valid_indices) > max_points_per_update:
            seed = 42 + self._iter_count
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(valid_indices), max_points_per_update, replace=False)
            obs_indices = np.sort(valid_indices[chosen])
        else:
            obs_indices = valid_indices
        self._iter_count += 1

        x_2d = self.x.reshape(self.ny, self.nx)
        P_2d = self.P_diag.reshape(self.ny, self.nx)
        ship_flat = self.gravity_ship.ravel()

        # ---- 1. 逐点标量卡尔曼更新（矢量化；观测像素互不重复，顺序无关）----
        ii, jj = np.divmod(obs_indices, self.nx)

        if self.localize:
            self._update_ship_local(obs_indices, ii, jj, x_2d, P_2d,
                                    ship_flat, valid_mask)
            self.x = x_2d.ravel()
            self.P_diag = P_2d.ravel()
            return

        P_ii = P_2d[ii, jj]
        K = P_ii / (P_ii + self.R_ship_val)
        innov = ship_flat[obs_indices] - x_2d[ii, jj]
        x_2d[ii, jj] += K * innov
        P_2d[ii, jj] = (1.0 - K) * P_ii

        # ---- 2. 高斯邻域传播（矢量化 + bincount 累加）----
        r = prop_radius
        offs = np.arange(-r, r + 1)
        dyy, dxx = np.meshgrid(offs, offs, indexing="ij")
        dist = np.sqrt(dxx.astype(float) ** 2 + dyy.astype(float) ** 2)
        w_base = np.exp(-0.5 * (dist / (r / 2.0)) ** 2)      # (2r+1, 2r+1)

        n_obs = obs_indices.size
        win = (2 * r + 1) ** 2
        flat_i = np.empty(n_obs * win, dtype=np.int64)
        flat_c = np.empty(n_obs * win, dtype=np.float64)
        flat_w = np.empty(n_obs * win, dtype=np.float64)
        pos = 0
        for s in range(0, n_obs, chunk_size):
            sl = slice(s, min(s + chunk_size, n_obs))
            ti = ii[sl, None, None] + dyy[None, :, :]
            tj = jj[sl, None, None] + dxx[None, :, :]
            inside = (ti >= 0) & (ti < self.ny) & (tj >= 0) & (tj < self.nx)
            # 边缘点的窗口被裁剪后按原实现重新归一化
            S = np.where(inside, w_base[None, :, :], 0.0).sum(axis=(1, 2))
            cw = w_base[None, :, :] / np.maximum(S, 1e-10)[:, None, None]
            m = int(inside.sum())
            contrib = (K[sl] * innov[sl])[:, None, None] * cw      # (m, W, W)
            flat_i[pos:pos + m] = (ti[inside] * self.nx + tj[inside])
            flat_c[pos:pos + m] = contrib[inside]
            flat_w[pos:pos + m] = cw[inside]
            pos += m
        flat_i, flat_c, flat_w = flat_i[:pos], flat_c[:pos], flat_w[:pos]

        big = self.ny * self.nx
        correction_field = np.bincount(flat_i, weights=flat_c,
                                       minlength=big).reshape(self.ny, self.nx)
        weight_field = np.bincount(flat_i, weights=flat_w,
                                   minlength=big).reshape(self.ny, self.nx)

        has_weight = weight_field > 0
        no_ship = ~valid_mask
        mask_prop = has_weight & no_ship
        if mask_prop.sum() > 0:
            blend = correction_field[mask_prop] / np.maximum(weight_field[mask_prop], 1e-10)
            x_2d[mask_prop] += prop_fraction * blend
            info_gain = prop_fraction * np.minimum(weight_field[mask_prop], 1.0)
            P_2d[mask_prop] = P_2d[mask_prop] * np.maximum(1.0 - info_gain, 0.1)

        self.x = x_2d.ravel()
        self.P_diag = P_2d.ravel()

    def _update_ship_local(self, obs_indices, ii, jj, x_2d, P_2d,
                           ship_flat, valid_mask):
        """局部化船测更新：邻居增益由 P∘L 推出，替代“更新后扩散”。"""
        R = self.R_ship_val
        P_i = P_2d[ii, jj].copy()
        K = P_i / (P_i + R)
        innov = ship_flat[obs_indices] - x_2d[ii, jj]
        x_2d[ii, jj] += K * innov
        P_2d[ii, jj] = (1.0 - K) * P_i

        r = min(int(np.ceil(3.0 * self.corr_px)), 40)
        offs = np.arange(-r, r + 1)
        dyy, dxx = np.meshgrid(offs, offs, indexing="ij")
        L = np.exp(-0.5 * (np.hypot(dxx, dyy) / self.corr_px) ** 2)
        L[r, r] = 0.0

        n = len(obs_indices)
        win = (2 * r + 1) ** 2
        big = self.ny * self.nx
        cs = np.zeros(big)
        ws = np.zeros(big)
        cnt = np.zeros(big)
        chunk = max(1, min(20000, int(2.0e7 / win)))
        for s in range(0, n, chunk):
            sl = slice(s, min(s + chunk, n))
            ti = ii[sl, None, None] + dyy
            tj = jj[sl, None, None] + dxx
            inside = (ti >= 0) & (ti < self.ny) & (tj >= 0) & (tj < self.nx)
            Pj = np.zeros(ti.shape)
            Pj[inside] = P_2d[ti[inside], tj[inside]]
            g = Pj * L[None] / (P_i[sl, None, None] + R)
            c = innov[sl, None, None] * g
            inf = L[None] * P_i[sl, None, None] / (P_i[sl, None, None] + R)
            idx = ti[inside] * self.nx + tj[inside]
            cs += np.bincount(idx, weights=c[inside], minlength=big)
            ws += np.bincount(idx, weights=inf[inside], minlength=big)
            cnt += np.bincount(idx, minlength=big)

        cs = cs.reshape(self.ny, self.nx)
        ws = ws.reshape(self.ny, self.nx)
        cnt = cnt.reshape(self.ny, self.nx)
        mask = (cnt > 0) & ~valid_mask
        if mask.any():
            x_2d[mask] += cs[mask] / np.maximum(cnt[mask], 1)
            P_2d[mask] *= np.clip(1.0 - ws[mask] / np.maximum(cnt[mask], 1),
                                  0.1, 1.0)

    def smooth_backward(self, x_history, x_pred_history, P_diag_history, smooth_sigma=0.8):
        """RTS 反向平滑：利用前向滤波历史状态和协方差，获得更优估计。"""
        factor = 1.0 + smooth_sigma * 0.1
        n_steps = len(x_history)
        x_smooth = x_history[-1].copy()
        P_smooth = P_diag_history[-1].copy()

        for k in range(n_steps - 2, -1, -1):
            P_pred = P_diag_history[k] * factor + self.Q_diag
            C_k = P_diag_history[k] / np.maximum(P_pred, 1e-10)
            x_smooth = x_history[k] + C_k * (x_smooth - x_pred_history[k + 1])
            P_smooth = P_diag_history[k] + C_k ** 2 * (P_smooth - P_pred)
            P_smooth = np.maximum(P_smooth, 0)

        return x_smooth, P_smooth

    def run(self, n_iterations=6, smooth_sigma=4.0, verbose=True):
        """运行迭代预测-更新循环 + RTS 反向平滑，返回重建的高分辨率重力场。

        默认值即本项目实际使用的配置 (6 轮, σ=4 像元≈1.85 km)，与
        BathyPriorKF / three_method_compare 一致；原默认 (5 轮, σ=0.8)
        会得到另一套方法。
        """
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化滤波器")

        self._iter_count = 0
        x_history = []
        x_pred_history = []
        p_history = []

        for it in range(n_iterations):
            self.predict(smooth_sigma=smooth_sigma)
            x_pred_history.append(self.x.copy())

            if self.adaptive:
                x2 = self.x.reshape(self.ny, self.nx)
                lr = self.gravity_lr
                by = self.ny // lr.shape[0]; bx = self.nx // lr.shape[1]
                bm = x2.reshape(lr.shape[0], by, lr.shape[1], bx).mean(axis=(1, 3))
                d = lr - bm
                P2 = self.P_diag.reshape(self.ny, self.nx)
                Pb = P2.reshape(lr.shape[0], by, lr.shape[1], bx).mean(axis=(1, 3))
                r_emp = float(np.var(d) - np.mean(Pb))
                self.R_sat_val = float(np.clip(
                    0.7 * self.R_sat_val + 0.3 * max(r_emp, 1.0), 1.0, 100.0))
                if self.adaptive_ship and self.gravity_ship is not None:
                    v = (~np.isnan(self.gravity_ship)) & self.ship_train_mask
                    if v.any():
                        res = self.gravity_ship[v] - x2[v]
                        r_emp = float(np.var(res) - np.mean(P2[v]))
                        self.R_ship_val = float(np.clip(
                            0.7 * self.R_ship_val + 0.3 * max(r_emp, 0.25),
                            0.25, 25.0))
                if verbose:
                    print(f"    adaptive: R_sat={self.R_sat_val:.2f}, "
                          f"R_ship={self.R_ship_val:.2f}")

            self.update_satellite()
            if self.gravity_ship is not None:
                self.update_shipborne()

            x_history.append(self.x.copy())
            p_history.append(self.P_diag.copy())

            if verbose:
                x_2d = self.x.reshape(self.ny, self.nx)
                print(f"  迭代 {it+1}/{n_iterations}: mean={x_2d.mean():.2f}, std={x_2d.std():.2f}, "
                      f"range=[{x_2d.min():.1f}, {x_2d.max():.1f}]")

        x_final, P_smooth = self.smooth_backward(x_history, x_pred_history, p_history, smooth_sigma=smooth_sigma)
        self._P_final = P_smooth.reshape(self.ny, self.nx)
        return x_final.reshape(self.ny, self.nx)
