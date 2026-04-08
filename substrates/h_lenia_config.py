# -*- coding: utf-8 -*-
"""
H-Lenia 設定クラス (v52 - 修正最終版)
- このファイルでシミュレーションの全パラメータを管理します
"""
import numpy as np
from . import pattern as pt 
from datetime import datetime
import itertools 
import os

class SimulationConfig:
    """3層H-Leniaシミュレーションの全設定を管理するクラス"""
    
    def __init__(self):
        """シミュレーションの基本パラメータをここで定義します"""
        
        # --- シミュレーション基本設定 ---
        self.n_anim = 500  # 生成する動画の総フレーム数 (full length, memory efficient because we use 'final' sampling)
        self.steps_per_frame = 16  # 動画の1フレームあたり、内部で何ステップ計算を進めるか
        self.gpu_chunk_size = 100  # GPU使用時に一度に計算するフレーム数の上限 (メモリ制限対策)
        self.dpi = 100  # 出力する動画の解像度 (dots per inch)

        # --- 階層設定 (3層: L1, L2, L3) ---
        # 各層の相対的なスケールを定義。
        # 例: scale_1=4, scale_2=2, scale_3=1 の場合、L1が最も高解像度なボトム層になる
        self.scale_1 = 16  # L1 のスケール
        self.scale_2 = 4  # L2 のスケール
        self.scale_3 = 1  # L3 のスケール
        
        # 最も低解像度な層(スケールが1の層)の、一辺のピクセル数
        self.base_size = 96
        # 最も高解像度な層に配置する初期個体数の基準値
        self.base_number_of_individuals = int(os.environ.get('LENIA_BASE_POP', 9))
        print(f"DEBUG: base_number_of_individuals set to {self.base_number_of_individuals}")

        # --- 各層のLeniaパラメータ ---
        # pattern.py で定義されているLeniaの種別名を指定
        self.pattern_name1 = 'aquarium'
        self.pattern_name2 = 'aquarium'
        self.pattern_name3 = 'aquarium'

        # --- ダウンサンプリング手法 ---
        # 高解像度層から低解像度層へ情報を伝える際の計算方法
        self.downsample_method = "average"
        print(f"Downsample Method: {self.downsample_method}")

        # --- 相互作用設定 ---
        # これから実行する実験の相互作用タイプをリストで指定する
        self.selected_interactions_thesis = [
            'R/AR/A',  # bottom(Middleから受ける)/Middle(Topから受ける, Bottomから受ける)/Top(Middleから受ける)
        ]

        # --- 斥力計算モード ---
        self.repulsion_mode = '1_minus_abs'
        print(f"Repulsion Mode: {self.repulsion_mode}")

        # --- 相互作用設定 (Notation順) ---
        # 論文表記: R(0.6)/A(0.5)R(0.9)/A(0.4)
        # 物理的意味: 上層は逃げ(0.6)、中層は上を追う(0.5)が下からは逃げる(0.9)、下層は中層を追う(0.4)
        
        # 1. Top(L3) <--- Middle(L2) : [R]epulsion (0.6)
        self.k23_values = [0.6]  # L3がL2から受ける
        
        # 2. Middle(L2) <--- Top(L3) : [A]ttraction (0.5)
        self.k32_values = [0.5]  # L2がL3から受ける
        
        # 3. Middle(L2) <--- Bottom(L1) : [R]epulsion (0.9)
        self.k12_values = [0.9]  # L2がL1から受ける
        
        # 4. Bottom(L1) <--- Middle(L2) : [A]ttraction (0.4)
        self.k21_values = [0.4]  # L1がL2から受ける

        # --- 出力設定 ---
        self.output_dir_base = "/mnt/e/H-Lenia"
        self.device_type = "gpu"

        self.max_parallel_animations = os.cpu_count()

        # 【重要】上記の基本パラメータから、実際のシミュレーションで使う値を計算する
        # この呼び出しは、計算に必要な変数が全て定義された後に置く必要があります
        self._calculate_derived_params()

    def _calculate_derived_params(self):
        """基本パラメータから、各層の具体的なサイズや個体数などを自動計算する"""
        
        # 1. 基準となる最大スケールを動的に決定
        max_scale = max(self.scale_1, self.scale_2, self.scale_3)

        # L1 の設定
        self.size_1 = self.base_size * self.scale_1
        # もしL1が最大スケールなら、初期個体を配置する
        if self.scale_1 == max_scale:
            self.num_individuals_1 = self.base_number_of_individuals * (self.scale_1 ** 2)
        else:
            self.num_individuals_1 = 0
        try: 
            self.param_1 = pt.pattern[self.pattern_name1]
        except KeyError: raise ValueError(f"Pattern '{self.pattern_name1}' not found")

        # L2 の設定
        self.size_2 = self.base_size * self.scale_2
        if self.scale_2 == max_scale:
            self.num_individuals_2 = self.base_number_of_individuals * (self.scale_2 ** 2)
        else:
            self.num_individuals_2 = 0
        try: 
            self.param_2 = pt.pattern[self.pattern_name2].copy(); self.param_2['R'] *= 1.5
            self.param_2['R'] = self.param_2['R'] * 2.0 # Increase L2 Radius (Inhibition Range)
        except KeyError: raise ValueError(f"Pattern '{self.pattern_name2}' not found")

        # L3 の設定
        self.size_3 = self.base_size * self.scale_3
        if self.scale_3 == max_scale:
            self.num_individuals_3 = self.base_number_of_individuals * (self.scale_3 ** 2)
        else:
            self.num_individuals_3 = 0
        try: 
            self.param_3 = pt.pattern[self.pattern_name3].copy(); self.param_3['R'] *= 2.0
            self.param_3['R'] = self.param_3['R'] * 3.0 # Increase L3 Radius (Apex Inhibition Range)
        except KeyError: raise ValueError(f"Pattern '{self.pattern_name3}' not found")
        
        # 2. 時間スケールの計算式を修正
        self.time_scale_1 = max(1, max_scale // self.scale_1)
        self.time_scale_2 = max(1, max_scale // self.scale_2)
        self.time_scale_3 = max(1, max_scale // self.scale_3)
        
    def get_timestamped_basedir(self):
        """実行日時を含む、ユニークなベースディレクトリ名を取得する"""
        return f"{self.output_dir_base}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # --- ヘルパー関数群 ---
    def _char_to_type(self, char):
        if char == 'A': return 0
        elif char == 'R': return 1
        return 2

    def translate_thesis_to_code(self, thesis_notation):
        """
        論文表記 'Top/Middle/Bottom' を、コードのタプル (t12, t21, t23, t32) に変換する。
        L1, L2, L3のスケール値に基づき、Top/Middle/Bottomを動的に割り当てる。
        """
        scales = [(self.scale_1, 1), (self.scale_2, 2), (self.scale_3, 3)]
        scales.sort(key=lambda x: x[0]) # スケール値でソート (小さい=Top)
        
        top_idx = scales[0][1]
        mid_idx = scales[1][1]
        bot_idx = scales[2][1]

        parts = thesis_notation.split('/')
        if len(parts) != 3 or len(parts[1]) != 2: 
            raise ValueError(f"Invalid thesis notation: {thesis_notation}")

        # 論文表記と層間の対応関係
        # parts[0]     -> Top層がMiddle層から受ける影響
        # parts[1][0]  -> Middle層がTop層から受ける影響
        # parts[1][1]  -> Middle層がBottom層から受ける影響
        # parts[2]     -> Bottom層がMiddle層から受ける影響
        
        interactions = {}
        interactions[f'{mid_idx}{top_idx}'] = self._char_to_type(parts[0])      # Mid -> Top
        interactions[f'{top_idx}{mid_idx}'] = self._char_to_type(parts[1][0])   # Top -> Mid
        interactions[f'{bot_idx}{mid_idx}'] = self._char_to_type(parts[1][1])   # Bot -> Mid
        interactions[f'{mid_idx}{bot_idx}'] = self._char_to_type(parts[2])      # Mid -> Bot
        
        # シミュレータが要求する順番 (t12, t21, t23, t32) で返す
        t12 = interactions.get('12', 2)
        t21 = interactions.get('21', 2)
        t23 = interactions.get('23', 2)
        t32 = interactions.get('32', 2)

        return (t12, t21, t23, t32)

    def interaction_type_to_str(self, interaction_type):
        if interaction_type == 0: return 'A'
        elif interaction_type == 1: return 'R'
        return 'N'