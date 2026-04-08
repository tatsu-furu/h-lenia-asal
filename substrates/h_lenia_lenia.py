# -*- coding: utf-8 -*-
"""
H-Lenia コア Lenia 計算関数 (v64 - 高速化版)
- 初期配置ループを Python for から jax.lax.scan に変更
- これにより、大量の個体配置時のコンパイル時間増大とConstant Folding警告を解消
"""
import jax
import jax.numpy as jnp
import jax.image
import jax.scipy.ndimage
import numpy as np    
import functools

# --- 基本関数 ---
@jax.jit
def bell(x, m, s):
    return jnp.exp(-((x - m) / s) ** 2 / 2.0)

@jax.jit
def growth(U, m, s):
    return bell(U, m, s) * 2.0 - 1.0

# --- JAX対応 回転関数 ---
def rotate_jax(image, angle_deg):
    angle_rad = jnp.deg2rad(angle_deg)
    h, w, c = image.shape
    
    # 座標グリッド
    y, x = jnp.mgrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = y - cy
    x = x - cx
    
    cos_a = jnp.cos(-angle_rad)
    sin_a = jnp.sin(-angle_rad)
    
    src_x = cos_a * x - sin_a * y + cx
    src_y = sin_a * x + cos_a * y + cy
    
    coordinates = jnp.stack([src_y, src_x])
    
    def map_channel(ch_idx):
        return jax.scipy.ndimage.map_coordinates(
            image[..., ch_idx], coordinates, order=1, mode='constant', cval=0.0
        )
    
    rotated = jax.vmap(map_channel)(jnp.arange(c))
    return jnp.transpose(rotated, (1, 2, 0))

# --- ワールド初期化 ---
def initialize_world(param, size, scale, num_individuals, key):
    kernels = param.get("kernels", [])
    if not kernels and "b" in param:
        kernels = [{
            "b": param["b"], 
            "m": param.get("m"), 
            "s": param.get("s"), 
            "h": param.get("h", 1.0), 
            "r": 1.0, 
            "c0": 0, 
            "c1": 0
        }]
    
    cells = param.get("cells", [])
    num_cells = len(cells)

    if not kernels:
        m, s, h = jnp.array([[]]), jnp.array([[]]), jnp.array([[]])
        c0 = tuple()
        c1 = tuple(() for _ in range(max(1, num_cells)))
        num_channels = max(1, num_cells)
    else:
        m, s, h = [jnp.stack([k[p] for k in kernels])[None, None, ...] for p in ('m', 's', 'h')]
        c0 = tuple(k['c0'] for k in kernels)
        num_channels = max(1, num_cells)
        c1 = tuple(tuple(i for i, k in enumerate(kernels) if k.get('c1') == c) for c in range(num_channels))

    cells_resized = []
    C = None 
    pattern_h, pattern_w = 0, 0
    
    if cells and scale > 0:
        for cell in cells:
            cell_arr = jnp.array(cell)
            new_shape = (int(cell_arr.shape[0] * scale), int(cell_arr.shape[1] * scale))
            resized = jax.image.resize(cell_arr, new_shape, method="nearest")
            cells_resized.append(resized)
        if cells_resized:
            C = jnp.dstack(cells_resized)
            pattern_h, pattern_w, num_channels_from_C = C.shape
            num_channels = max(num_channels, num_channels_from_C)
            if num_channels_from_C > num_cells and kernels:
                c1 = tuple(tuple(i for i, k in enumerate(kernels) if k.get('c1') == c) for c in range(num_channels))

    R_scaled = param["R"] * scale if scale > 0 else param["R"]
    A = jnp.zeros([size, size, num_channels], dtype=jnp.float32)

    # --- ワールドに初期個体を配置 (scanによる高速化版) ---
    if num_individuals > 0 and C is not None and pattern_h > 0 and pattern_w > 0:
        key_pos, key_angle = jax.random.split(key)
        rand_pos = jax.random.uniform(key_pos, shape=(num_individuals, 2), minval=0.0, maxval=float(size))
        angles = jax.random.uniform(key_angle, (num_individuals,), minval=0, maxval=360.0)

        # パディング作成
        pad_h = pattern_h
        pad_w = pattern_w
        A_padded = jnp.pad(A, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

        # ループ処理の中身を関数として定義
        def scan_body(carry_A, inputs):
            angle, pos_raw = inputs
            
            # 回転
            rotated_C = rotate_jax(C, angle)
            
            # 位置計算
            pos_x = pos_raw[0].astype(jnp.int32)
            pos_y = pos_raw[1].astype(jnp.int32)
            offset_x = pos_x + pad_h - pattern_h // 2
            offset_y = pos_y + pad_w - pattern_w // 2
            
            # 切り出し -> 加算 -> 書き戻し
            current_patch = jax.lax.dynamic_slice(
                carry_A, (offset_x, offset_y, 0), (pattern_h, pattern_w, num_channels)
            )
            added_patch = current_patch + rotated_C
            new_A = jax.lax.dynamic_update_slice(
                carry_A, added_patch, (offset_x, offset_y, 0)
            )
            
            return new_A, None # 第2戻り値はscanの蓄積結果（今回は不要なのでNone）

        # jax.lax.scan で高速にループ実行
        scan_inputs = (angles, rand_pos)
        A_padded, _ = jax.lax.scan(scan_body, A_padded, scan_inputs)

        # 切り抜き & クリップ
        A = A_padded[pad_h : pad_h + size, pad_w : pad_w + size, :]
        A = jnp.clip(A, 0.0, 1.0)

    T_val = param.get("T", 10.0)
    return A, m, s, h, c0, c1, R_scaled, num_channels, T_val

# --- カーネル計算 ---
def compute_kernel(param, R, size):
    mid = size // 2
    Y, X = jnp.meshgrid(jnp.arange(-mid, size - mid), jnp.arange(-mid, size - mid), indexing='ij')
    D_grid = jnp.sqrt(X**2 + Y**2)
    
    kernels_fft = []
    kernels = param.get("kernels", [])
    if not kernels and "b" in param:
        kernels = [{
            "b": param["b"], 
            "m": param.get("m"), 
            "s": param.get("s"), 
            "h": param.get("h", 1.0), 
            "r": 1.0
        }]
    num_kernels = len(kernels)

    if num_kernels == 0:
        return jnp.zeros((size, size, 1), dtype=jnp.complex64)

    for i in range(num_kernels):
        k = param["kernels"][i]
        b_arr = jnp.array(k.get('b', []), dtype=jnp.float32)
        len_b = len(b_arr)

        if len_b == 0:
            K_single = jnp.zeros((size, size))
        else:
            Ds = D_grid / (R * k.get('r', 1) / len_b + 1e-9)
            indices = jnp.minimum(Ds.astype(int), len_b - 1)
            K_single = (Ds < len_b) * b_arr[indices] * bell(Ds % 1.0, 0.5, 0.15)
        kernels_fft.append(K_single)

    K = jnp.dstack(kernels_fft)
    K_sum = jnp.sum(K, axis=(0, 1), keepdims=True)
    nK = K / (K_sum + 1e-9) 
    fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0, 1)), axes=(0, 1))
    return fK.astype(jnp.complex64)

# --- Lenia 更新ステップ ---
@functools.partial(jax.jit, static_argnames=("num_channels",))
def _lenia_update_step(A, fK, m, s, h, c0, c1, T, num_channels):
    fA = jnp.fft.fft2(A, axes=(0, 1)).astype(jnp.complex64)
    c0_array = jnp.array(c0)
    fAk = fA[:, :, c0_array]
    fK_filtered = fK
    U = jnp.real(jnp.fft.ifft2(fK_filtered * fAk, axes=(0, 1)))
    G = growth(U, m, s) * h
    H_list = []
    for c in range(num_channels):
        kernel_indices = jnp.array(c1[c])
        H_list.append(jnp.sum(G[:, :, kernel_indices], axis=-1, where=kernel_indices.size > 0, initial=0.0))
    H = jnp.stack(H_list, axis=-1)
    safe_T = jnp.where(T <= 1e-6, 1e-6, T)
    A_new = jnp.clip(A + (1.0 / safe_T) * H, 0.0, 1.0);
    return A_new.astype(jnp.float32)