# -*- coding: utf-8 -*-
"""
H-Lenia シミュレータ コア関数
- JAXによる高速化（JITコンパイル）を前提として記述
- 層間のリサイズ処理と、相互作用の計算ロジックを定義
"""
import jax, jax.numpy as jnp, functools

# JAXのJIT(Just-In-Time)コンパイル機能を使って、この関数を高速化する
# static_argnamesは、コンパイルの際に定数として扱う引数を指定し、最適化を促進する
@functools.partial(jax.jit, static_argnames=("target_shape", "method"))
def resize_matrix(matrix, target_shape, method="linear"):
    """JAXを使って行列を指定された形状にリサイズする汎用関数"""
    
    # 入力をJAXが扱える配列形式に変換
    matrix,target_shape = jnp.asarray(matrix),tuple(target_shape)
    
    # 入力が2D（グレースケール画像など）の場合、一時的に3Dに変換して処理する
    if matrix.ndim == 2: 
        matrix_expanded, iw2d = matrix[...,None], True
    else: 
        matrix_expanded, iw2d = matrix, False
        
    # JAXの画像リサイズ関数を呼び出す
    new_shape = target_shape + matrix_expanded.shape[2:]
    resized = jax.image.resize(matrix_expanded, new_shape, method=method)
    
    # 入力が2Dだった場合は、次元を元に戻して返す
    return resized[...,0] if iw2d else resized

# JITコンパイルで高速化
@functools.partial(jax.jit, static_argnames=("target_shape",))
def average_pooling_downsample(matrix, target_shape):
    """平均プーリング法によるダウンサンプリング（画像の縮小）を行う"""
    
    target_shape = tuple(target_shape)
    # 縮小率（縦・横それぞれ何ピクセルを1ピクセルにまとめるか）を計算
    fy,fx = matrix.shape[0]//target_shape[0], matrix.shape[1]//target_shape[1]
    
    # 元の行列を、プーリングするブロックごとに次元を分割して変形する
    # 例：(100, 100) -> (50, 50) に縮小する場合、(50, 2, 50, 2) の形に変形
    new_shape = (target_shape[0],fy,target_shape[1],fx)+matrix.shape[2:]
    
    # ブロックごと（分割した次元）の平均値を計算することで、平均プーリングを実現
    return matrix.reshape(new_shape).mean(axis=(1,3))

# JITコンパイルで高速化。interaction_typeなどは定数として扱う
@functools.partial(jax.jit, static_argnames=("interaction_type", "repulsion_mode_is_1_minus_abs", "downsample_method"))
def compute_interaction(x0, x, k, interaction_type: int, repulsion_mode_is_1_minus_abs: bool, downsample_method: str):
    """
    2つの層（x0:作用元, x:作用先）間の相互作用の力場を計算するメイン関数
    """
    x0, x = jnp.asarray(x0), jnp.asarray(x)
    
    # --- ステップ1: 層間の解像度を合わせる ---
    # 作用元の層(x0)が、作用先の層(x)より高解像度の場合 -> ダウンサンプリング
    if x0.shape[:2] > x.shape[:2]:
        if downsample_method == "linear": 
            resized_x0 = resize_matrix(x0, x.shape[:2], method="linear") # 線形補間
        elif downsample_method == "average": 
            resized_x0 = average_pooling_downsample(x0, x.shape[:2]) # 平均プーリング
    # 作用元の層(x0)が、作用先の層(x)より低解像度の場合 -> アップサンプリング
    elif x0.shape[:2] < x.shape[:2]: 
        resized_x0 = resize_matrix(x0, x.shape[:2], method="nearest") # 最近傍補間
    # 解像度が同じ場合
    else: 
        resized_x0 = x0
        
    # --- ステップ2: 相互作用の効果(E)を計算 ---
    # 2層間の状態値の差分(デルタ)を計算
    delta, abs_delta = resized_x0 - x, jnp.abs(resized_x0 - x)
    
    # 斥力(Repulsion)の計算方法を2種類準備
    # モード1: 論文で定義した 1 - |delta| ベースの計算
    base_repulse_effect_1mA = jnp.where(delta > 0, -(1.0 - abs_delta), (1.0 - abs_delta))
    # モード2: 単純な線形の斥力 (-delta)
    base_repulse_effect_linear = -delta
    
    # Configで設定された斥力モードを選択
    base_repulse_E = jnp.where(repulsion_mode_is_1_minus_abs, base_repulse_effect_1mA, base_repulse_effect_linear)
    
    # JAXのswitch文(JITコンパイルと相性が良い)を使い、相互作用タイプに応じて最終的な効果(E)を決定
    # interaction_typeが0なら引力(A), 1なら斥力(R), 2なら相互作用なし(N)
    base_E = jax.lax.switch(
        interaction_type, 
        [lambda d: d,                 # 0 (Attraction): deltaをそのまま使う
         lambda d: base_repulse_E,    # 1 (Repulsion): 計算済みの斥力を使う
         lambda d: jnp.zeros_like(d)  # 2 (None): ゼロ（影響なし）
        ], 
        delta
    )
    
    # --- ステップ3: 最終的な力場を計算 ---
    # 効果(E)に相互作用強度(k)を掛けて、基本的な力(F)を計算
    base_force_F = k * base_E
    
    # 境界減衰(damping)を適用し、最終的な力場を計算
    # 状態値が[0, 1]の範囲を超えないように、力にブレーキをかける
    dampened_force = base_force_F * jnp.where(
        base_force_F > 0,      # もし力がプラス（状態値を増やそうとする）なら...
        1.0 - x,               # ...1に近づくほどブレーキが強くなる (1-x)
        jnp.where(
            base_force_F < 0,  # もし力がマイナス（状態値を減らそうとする）なら...
            x,                 # ...0に近づくほどブレーキが強くなる (x)
            0.0                # もし力がゼロなら、ブレーキもゼロ
        )
    )
    
    # データ型をfloat32に指定して返す
    return dampened_force.astype(jnp.float32)