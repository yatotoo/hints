#!/usr/bin/env python3
"""
interaction_extraction.py - Higher-Order Interaction Extraction
================================================================
This module implements the Kramers-Moyal coefficient method to extract
pairwise (A), three-way (C), and four-way (E) interaction tensors from
multivariate time series data, following Eq. (3) in the paper.

Author: Stability Analysis Pipeline
Date: 2024
License: MIT

Command-line Arguments:
-----------------------
--input_file : str
    Path to trajectory file (.npy format)
--output_dir : str
    Directory for saving extracted tensors (default: './outputs')
--order : int
    Maximum order of interactions (default: 3, up to cubic terms)
--tau : float
    Time lag for Kramers-Moyal estimation (default: 0.01)
--window_size : int
    Number of data points per window (default: 50000)
--overlap : float
    Window overlap fraction (default: 0.95)
--normalize : bool
    Whether to normalize data to zero mean unit variance (default: True)
--verbose : bool
    Print detailed progress information (default: True)
"""

import argparse
import numpy as np
from scipy import linalg
from typing import Tuple, Optional, Dict, List
import os
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def _solve_with_ridge_and_standardize(X, y, lam=1e-4, intercept_col=0):
    """
    对除截距列外的所有列做标准化（均值0、方差1），y不动。
    拟合 ridge，然后把系数反变换回原始量纲。
    """
    import numpy as np

    # 拆出截距列与其余列
    X0 = X[:, [intercept_col]]                    # (T,1)
    Xp = np.delete(X, intercept_col, axis=1)      # (T, P)

    # 仅对 Xp 标准化
    mu = Xp.mean(axis=0)
    sd = Xp.std(axis=0)
    sd[sd == 0] = 1.0
    Xp_std = (Xp - mu) / sd

    # 重新拼回标准化后的设计矩阵
    X_std = np.concatenate([X0, Xp_std], axis=1)

    # 岭回归闭式解（可换成 sklearn Ridge）
    # (X^T X + lam I)^{-1} X^T y
    P = X_std.shape[1]
    I = np.eye(P); I[intercept_col, intercept_col] = 0.0  # 不惩罚截距
    beta_std = np.linalg.solve(X_std.T @ X_std + lam * I, X_std.T @ y)

    # 把标准化系数反变换回原尺度
    beta = np.zeros_like(beta_std)
    beta[intercept_col] = beta_std[intercept_col] - np.sum((mu / sd) * beta_std[1:])
    beta[1:] = beta_std[1:] / sd

    return beta


class InteractionExtractor:
    """
    Extract interaction tensors A, C, E from multivariate time series
    using the Kramers-Moyal coefficient method
    """

    def __init__(self, order: int = 3, tau: float = 0.01):
        """
        Initialize the extractor

        Parameters:
        -----------
        order : int
            Maximum order of interactions (3 for cubic terms)
        tau : float
            Time lag for finite difference approximation
        """
        self.order = order
        self.tau = tau
        self.A = None  # Pairwise interactions
        self.C = None  # Three-way interactions
        self.E = None  # Four-way interactions
        self.alpha = None  # Constant drift terms

    def extract(self, trajectory: np.ndarray,
                normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        提取交互张量
        """
        # 归一化
        if normalize:
            mean = trajectory.mean(axis=0)
            std = trajectory.std(axis=0)
            std[std < 1e-10] = 1.0
            trajectory = (trajectory - mean) / std

        T, N = trajectory.shape

        # 初始化张量
        self.alpha = np.zeros(N)
        self.A = np.zeros((N, N))
        self.C = np.zeros((N, N, N))
        self.E = np.zeros((N, N, N, N))

        # 对每个变量拟合
        for i in tqdm(range(N), desc="Extracting interactions"):
            X, y = self._build_linear_system(trajectory, i)

            # 使用Ridge回归（更稳定）
            # coeffs = _solve_with_ridge_and_standardize(X, y, lam=1e-4, intercept_col=0)
            from scipy.linalg import lstsq
            coeffs, _, _, _ = lstsq(X, y)
            # 提取系数
            self._extract_coefficients(coeffs, i, N)

        # ========================
        # 关键：最后统一除以τ
        # ========================
        self.alpha = self.alpha / self.tau
        self.A = self.A / self.tau
        self.C = self.C / self.tau
        self.E = self.E / self.tau

        return {
            'alpha': self.alpha,
            'A': self.A,
            'C': self.C,
            'E': self.E
        }

    def _normalize(self, trajectory: np.ndarray) -> np.ndarray:
        """Normalize to zero mean unit variance"""
        mean = np.mean(trajectory, axis=0)
        std = np.std(trajectory, axis=0)
        # Avoid division by zero
        std[std < 1e-10] = 1.0
        return (trajectory - mean) / std

    def _build_linear_system(self, trajectory: np.ndarray,
                             var_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建线性系统，严格遵循论文补充材料公式(S3)

        核心思想：
        1. y_i(t,τ) = x_i(t+τ) - x_i(t)  （不除以τ）
        2. 设计矩阵X包含：[1, x_1, ..., x_N, x_1², x_1*x_2, ..., x_N², x_1³, ...]
        3. 拟合得到系数 α̃, φ̃, ψ̃, γ̃
        4. 最后在extract()中除以τ得到 α, A, C, E

        Parameters:
        -----------
        trajectory : np.ndarray
            时间序列数据，形状 (T, N)
        var_idx : int
            当前变量的索引 i

        Returns:
        --------
        X : np.ndarray
            设计矩阵，形状 (T-1, n_features)
        y : np.ndarray
            响应向量 y_i(t,τ)，形状 (T-1,)
        """
        T, N = trajectory.shape
        T_eff = T - 1  # 有效时间点数

        # ========================
        # 步骤1：计算 y_i(t,τ)
        # ========================
        # 注意：这里不除以τ！
        y = trajectory[1:, var_idx] - trajectory[:-1, var_idx]

        # ========================
        # 步骤2：构建设计矩阵 X
        # ========================
        # X的每一行对应时刻t的状态 x(t)
        # X的列对应不同阶数的单项式

        X_parts = []

        # --------------------
        # 2.1 常数项（截距）
        # --------------------
        X_const = np.ones((T_eff, 1))
        X_parts.append(X_const)

        # --------------------
        # 2.2 一阶项：x_j(t)
        # --------------------
        # 取时刻t的状态（不是t+τ）
        X_linear = trajectory[:-1, :]  # 形状 (T_eff, N)
        X_parts.append(X_linear)

        # --------------------
        # 2.3 二阶项：x_j(t) * x_k(t)
        # --------------------
        if self.order >= 2:
            # 生成所有 (j,k) 组合，包括 j=k 的情况
            # 顺序：x_1*x_1, x_1*x_2, ..., x_1*x_N,
            #       x_2*x_1, x_2*x_2, ..., x_2*x_N,
            #       ...
            #       x_N*x_1, x_N*x_2, ..., x_N*x_N
            X_quad = []
            for j in range(N):
                for k in range(N):
                    X_quad.append(X_linear[:, j] * X_linear[:, k])
            X_quad = np.column_stack(X_quad)  # 形状 (T_eff, N*N)
            X_parts.append(X_quad)

        # --------------------
        # 2.4 三阶项：x_j(t) * x_k(t) * x_l(t)
        # --------------------
        if self.order >= 3:
            # 生成所有 (j,k,l) 组合
            # 顺序：x_1*x_1*x_1, x_1*x_1*x_2, ..., x_N*x_N*x_N
            X_cubic = []
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        X_cubic.append(X_linear[:, j] * X_linear[:, k] * X_linear[:, l])
            X_cubic = np.column_stack(X_cubic)  # 形状 (T_eff, N*N*N)
            X_parts.append(X_cubic)

        # --------------------
        # 2.5 拼接所有列
        # --------------------
        X = np.hstack(X_parts)

        # ========================
        # 步骤3：返回 (X, y)
        # ========================
        # 注意：
        # - y 是原始差分（未除以τ）
        # - X 的列对应所有单项式
        # - 拟合得到的系数是 α̃, φ̃, ψ̃, γ̃
        # - 在 extract() 函数中会除以τ得到 α, A, C, E

        return X, y

    def _solve_linear_system(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve the linear system using least squares

        Parameters:
        -----------
        X : np.ndarray
            Design matrix
        y : np.ndarray
            Response vectorFVextract()

        Returns:
        --------
        coeffs : np.ndarray
            Estimated coefficients
        """
        # Use lstsq for numerical stability
        coeffs, _, _, _ = linalg.lstsq(X, y)
        return coeffs

    def _extract_coefficients(self, coeffs: np.ndarray,
                              var_idx: int, N: int):
        """
        从拟合系数中提取张量元素

        重要：
        - 输入的 coeffs 是 α̃, φ̃, ψ̃, γ̃（未除以τ）
        - 直接赋值，不做任何对称化或除法
        - 最后在 extract() 中统一除以τ

        Parameters:
        -----------
        coeffs : np.ndarray
            线性回归得到的系数
        var_idx : int
            当前变量索引 i
        N : int
            系统维度
        """
        idx = 0

        # ========================
        # 1. 常数项 α̃_i
        # ========================
        self.alpha[var_idx] = coeffs[idx]
        idx += 1

        # ========================
        # 2. 线性项 φ̃_ij (对应A矩阵)
        # ========================
        for j in range(N):
            self.A[var_idx, j] = coeffs[idx]
            idx += 1

        # ========================
        # 3. 二阶项 ψ̃_ijk (对应C张量)
        # ========================
        if self.order >= 2:
            for j in range(N):
                for k in range(N):
                    # 关键：直接赋值，不除以2，不对称化
                    self.C[var_idx, j, k] = coeffs[idx]
                    idx += 1

        # ========================
        # 4. 三阶项 γ̃_ijkl (对应E张量)
        # ========================
        if self.order >= 3:
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        # 关键：直接赋值，不除以6，不对称化
                        self.E[var_idx, j, k, l] = coeffs[idx]
                        idx += 1

        # ========================
        # 注意事项：
        # ========================
        # 1. 这里不做对称化！
        #    - 因为设计矩阵X已经包含了所有(j,k)和(j,k,l)组合
        #    - 例如 x_1*x_2 和 x_2*x_1 是两个不同的列
        #    - 拟合会自动学到 C_ijk 和 C_ikj 可能不同
        #
        # 2. 如果系统确实对称，拟合结果会自然接近对称
        #    - 例如 Lorenz-96 有旋转对称性
        #    - C_ijk ≈ C_ikj 会自动满足
        #
        # 3. 最后除以τ在 extract() 函数中统一完成



def analyze_interactions(tensors: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Analyze extracted interaction tensors

    Parameters:
    -----------
    tensors : dict
        Dictionary containing interaction tensors

    Returns:
    --------
    dict
        Summary statistics
    """
    stats = {}

    # Alpha (drift) statistics
    alpha = tensors['alpha']
    stats['alpha_mean'] = float(np.mean(alpha))
    stats['alpha_std'] = float(np.std(alpha))
    stats['alpha_max'] = float(np.max(np.abs(alpha)))

    # A matrix (pairwise) statistics
    A = tensors['A']
    stats['A_mean'] = float(np.mean(A))
    stats['A_std'] = float(np.std(A))
    stats['A_max'] = float(np.max(np.abs(A)))
    stats['A_diagonal_mean'] = float(np.mean(np.diag(A)))
    stats['A_offdiagonal_mean'] = float(np.mean(A[~np.eye(A.shape[0], dtype=bool)]))

    # C tensor (three-way) statistics
    C = tensors['C']
    stats['C_mean'] = float(np.mean(C))
    stats['C_std'] = float(np.std(C))
    stats['C_max'] = float(np.max(np.abs(C)))
    stats['C_nonzero_fraction'] = float(np.mean(np.abs(C) > 1e-10))

    # E tensor (four-way) statistics
    E = tensors['E']
    stats['E_mean'] = float(np.mean(E))
    stats['E_std'] = float(np.std(E))
    stats['E_max'] = float(np.max(np.abs(E)))
    stats['E_nonzero_fraction'] = float(np.mean(np.abs(E) > 1e-10))

    return stats


def sliding_window_extraction(trajectory: np.ndarray,
                              window_size: int = 50000,
                              overlap: float = 0.95,
                              **kwargs) -> List[Dict[str, np.ndarray]]:
    """
    Extract interactions using sliding windows

    Parameters:
    -----------
    trajectory : np.ndarray
        Full time series
    window_size : int
        Size of each window
    overlap : float
        Overlap fraction between windows

    Returns:
    --------
    list
        List of tensor dictionaries for each window
    """
    T, N = trajectory.shape
    stride = int(window_size * (1 - overlap))

    results = []
    extractor = InteractionExtractor(**kwargs)

    for start in tqdm(range(0, T - window_size + 1, stride),
                      desc="Processing windows"):
        end = start + window_size
        window_data = trajectory[start:end]

        tensors = extractor.extract(window_data)
        results.append(tensors)

    return results


def main():
    """Main function for command-line execution"""

    parser = argparse.ArgumentParser(
        description='Extract interaction tensors from time series')
    parser.add_argument('--input_file', type=str,
                        default='./outputs/lorenz96_trajectory.npy',
                        help='Path to trajectory file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--order', type=int, default=3,
                        help='Maximum interaction order')
    parser.add_argument('--tau', type=float, default=0.01,
                        help='Time lag for Kramers-Moyal estimation')
    parser.add_argument('--window_size', type=int, default=50000,
                        help='Window size for sliding analysis')
    parser.add_argument('--overlap', type=float, default=0.95,
                        help='Window overlap fraction')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize data to zero mean unit variance')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')

    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory from {args.input_file}")
    trajectory = np.load(args.input_file)
    print(f"Trajectory shape: {trajectory.shape}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract interactions
    print("\nExtracting interaction tensors...")
    extractor = InteractionExtractor(order=args.order, tau=args.tau)
    tensors = extractor.extract(trajectory, normalize=args.normalize)

    # Save tensors
    for name, tensor in tensors.items():
        output_file = os.path.join(args.output_dir, f'{name}_tensor.npy')
        np.save(output_file, tensor)
        print(f"Saved {name} tensor to {output_file}")

    # Compute and save statistics
    stats = analyze_interactions(tensors)
    stats_file = os.path.join(args.output_dir, 'interaction_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\nInteraction Statistics:")
    print(f"  Alpha: mean={stats['alpha_mean']:.3e}, max={stats['alpha_max']:.3e}")
    print(f"  A matrix: mean={stats['A_mean']:.3e}, diagonal={stats['A_diagonal_mean']:.3e}")
    print(f"  C tensor: mean={stats['C_mean']:.3e}, nonzero={stats['C_nonzero_fraction']:.1%}")
    print(f"  E tensor: mean={stats['E_mean']:.3e}, nonzero={stats['E_nonzero_fraction']:.1%}")

    # Optional: sliding window analysis
    if args.verbose:
        print("\n[Optional] Running sliding window analysis...")
        window_results = sliding_window_extraction(
            trajectory,
            window_size=args.window_size,
            overlap=args.overlap,
            order=args.order,
            tau=args.tau
        )
        print(f"Extracted tensors for {len(window_results)} windows")

        # Save window results
        window_file = os.path.join(args.output_dir, 'window_tensors.npy')
        np.save(window_file, window_results)
        print(f"Saved window results to {window_file}")

    return tensors


if __name__ == '__main__':
    main()