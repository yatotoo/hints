## =====================================================================
##  em_transmitter_stability.py - 电磁发射机稳定性分析
##  基于HiNTS框架，适配CSV数据
## =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
import os
import warnings

warnings.filterwarnings('ignore')

# ===== 导入您的完整提取器 ===== #
from interaction_extraction import InteractionExtractor


class EMTransmitterStabilityAnalyzer:
    """
    电磁发射机稳定性分析器

    基于HiNTS框架（Buffoni et al., 2022）
    适配电力电子系统的实测时间序列数据

    主要简化：
    1. 不寻找固定点（系统一直在运行，无固定点）
    2. 直接用A矩阵作为Jacobian（Order=1时）
    3. 或用 J_eff = A + 2C·x̄（Order=2时，x̄为数据均值）
    """

    def __init__(self, order=1, tau=1.0):
        """
        Parameters:
        -----------
        order : int
            交互阶数
            1: 仅线性 (dx/dt = α + Ax)
            2: 含二阶 (dx/dt = α + Ax + Cx²)
            3: 含三阶 (dx/dt = α + Ax + Cx² + Ex³)
        tau : float
            时间步长（对应采样间隔）
        """
        self.order = order
        self.tau = tau
        self.extractor = InteractionExtractor(order=order, tau=tau)

        # 电磁发射机的标准变量
        self.analysis_vars = ['Toff', 'U', 'I', 'Freq', 'T1', 'T2', 'T3', 'AC', 'R', 'L']

    def analyze_file(self, file_path, output_dir=None, save_figures=True, verbose=True):
        """
        分析单个CSV文件

        完整流程：
        1. 读取CSV → 标准化
        2. 提取张量（α, A, C, E）
        3. 计算有效Jacobian
        4. 特征值分解
        5. 稳定性判断
        6. 特征向量贡献分析
        7. 可视化
        """

        # ==================== 1. 数据准备 ==================== #
        if output_dir is None:
            file_name = os.path.basename(file_path).replace('.csv', '')
            output_dir = f'outputs/{file_name}'

        if save_figures:
            os.makedirs(f'{output_dir}/figures', exist_ok=True)
            os.makedirs(f'{output_dir}/data', exist_ok=True)

        if verbose:
            print("=" * 70)
            print(f"电磁发射机稳定性分析")
            print("=" * 70)
            print(f"文件: {os.path.basename(file_path)}")
            print(f"阶数: Order={self.order}")
            print(f"τ: {self.tau}")

        # 读取数据
        df = pd.read_csv(file_path)

        # 检查变量
        available_vars = [v for v in self.analysis_vars if v in df.columns]
        if len(available_vars) < len(self.analysis_vars):
            missing = set(self.analysis_vars) - set(available_vars)
            if verbose:
                print(f"  ⚠️  缺少变量: {missing}")

        # 提取数据矩阵
        data = df[available_vars].values

        # 处理缺失值
        if np.isnan(data).any():
            valid_rows = ~np.isnan(data).any(axis=1)
            data = data[valid_rows]

        n_samples, n_vars = data.shape

        if verbose:
            print(f"\n数据:")
            print(f"  样本数: {n_samples:,}")
            print(f"  变量数: {n_vars}")
            print(f"  变量: {available_vars}")

        # ==================== 2. 提取交互张量 ==================== #
        if verbose:
            print(f"\n提取交互张量...")

        # 使用完整的HiNTS提取器
        tensors = self.extractor.extract(
            trajectory=data,
            normalize=True  # 内部自动标准化
        )

        alpha = tensors['alpha']
        A = tensors['A']
        C = tensors['C']
        E = tensors['E']

        if verbose:
            print(f"  ✓ α: {alpha.shape}")
            print(f"  ✓ A: {A.shape}, 范围[{A.min():.3f}, {A.max():.3f}]")
            if self.order >= 2:
                print(f"  ✓ C: {C.shape}, ||C||={np.linalg.norm(C):.3e}")
            if self.order >= 3:
                print(f"  ✓ E: {E.shape}, ||E||={np.linalg.norm(E):.3e}")

        # ==================== 3. 计算有效Jacobian ==================== #
        if verbose:
            print(f"\n计算有效Jacobian...")


        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0) + 1e-10
        data_normalized = (data - data_mean) / data_std
        x_working_point = data_normalized.mean(axis=0)

        if verbose:
            print(f"  工作点范数: ||x̄|| = {np.linalg.norm(x_working_point):.6f}")

        # 根据Order计算有效Jacobian
        if self.order == 1:
            # Order=1: J_eff = A
            J_eff = A.copy()
            if verbose:
                print(f"  J_eff = A (线性近似)")

        elif self.order == 2:
            # Order=2: J_eff = A + (C + C^T) · x̄
            # 按照论文 Eq.(4): ∂F_i/∂x_j = A_ij + (C_ijk + C_ikj)x_k
            J_eff = A.copy()

            for i in range(n_vars):
                for j in range(n_vars):
                    for k in range(n_vars):
                        # 对称化的贡献
                        J_eff[i, j] += (C[i, j, k] + C[i, k, j]) * x_working_point[k]

            if verbose:
                print(f"  J_eff = A + (C+C^T)·x̄")
                print(f"  二阶修正: ||J_eff - A|| = {np.linalg.norm(J_eff - A):.3e}")

        elif self.order == 3:
            # Order=3: 包含三阶项的贡献
            J_eff = A.copy()

            # 二阶贡献
            for i in range(n_vars):
                for j in range(n_vars):
                    for k in range(n_vars):
                        J_eff[i, j] += (C[i, j, k] + C[i, k, j]) * x_working_point[k]

            # 三阶贡献（按论文补充材料）
            for i in range(n_vars):
                for j in range(n_vars):
                    for k in range(n_vars):
                        for l in range(n_vars):
                            J_eff[i, j] += (E[i, j, k, l] +
                                            E[i, k, j, l] +
                                            E[i, k, l, j])  * x_working_point[k] * x_working_point[l]

            if verbose:
                print(f"  J_eff = A + (C+C^T)·x̄ + (E+...)·x̄²")
                print(f"  三阶修正: ||J_eff - A|| = {np.linalg.norm(J_eff - A):.3e}")

        # ==================== 4. 特征值分解 ==================== #
        if verbose:
            print(f"\n特征值分析...")

        eigvals, eigvecs = np.linalg.eig(J_eff)

        # 排序（按实部从小到大）
        idx_sorted = np.argsort(np.real(eigvals))
        eigvals_sorted = eigvals[idx_sorted]
        eigvecs_sorted = eigvecs[:, idx_sorted]

        lambda_max = eigvals_sorted[-1]  # 最大（最右）
        lambda_min = eigvals_sorted[0]  # 最小（最左）

        # ==================== 5. 稳定性判断 ==================== #
        n_stable = np.sum(np.real(eigvals) < 0)
        n_unstable = np.sum(np.real(eigvals) >= 0)

        if np.real(lambda_max) < -0.01:
            stability_status = 'Stable'
        elif np.real(lambda_max) < 0.01:
            stability_status = 'Critical'
        else:
            stability_status = 'Unstable'

        if verbose:
            print(f"  λ_max: {np.real(lambda_max):.6f} + {np.imag(lambda_max):.6f}i")
            print(f"  λ_min: {np.real(lambda_min):.6f} + {np.imag(lambda_min):.6f}i")
            print(f"  稳定模态: {n_stable}")
            print(f"  不稳定模态: {n_unstable}")
            print(f"  状态: {stability_status}")

        # ==================== 6. 特征向量贡献 ==================== #
        contrib_max = np.abs(eigvecs_sorted[:, -1]) ** 2
        contrib_max /= contrib_max.sum()

        contrib_min = np.abs(eigvecs_sorted[:, 0]) ** 2
        contrib_min /= contrib_min.sum()

        # 分模态贡献
        var_contrib_stable = np.zeros(n_vars)
        var_contrib_unstable = np.zeros(n_vars)

        for i in range(n_vars):
            eigvec = eigvecs_sorted[:, i]
            contrib = np.abs(eigvec) ** 2
            contrib /= contrib.sum()

            if np.real(eigvals_sorted[i]) < 0:
                var_contrib_stable += contrib
            else:
                var_contrib_unstable += contrib

        if n_stable > 0:
            var_contrib_stable /= n_stable
        if n_unstable > 0:
            var_contrib_unstable /= n_unstable

        if verbose:
            print(f"\n最不稳定模态 (λ={np.real(lambda_max):.4f}) 主导变量:")
            idx_sorted_contrib = np.argsort(contrib_max)[::-1]
            for i in range(min(3, n_vars)):
                idx = idx_sorted_contrib[i]
                print(f"  {i + 1}. {available_vars[idx]:5s}: {contrib_max[idx] * 100:5.2f}%")

        # ==================== 7. 整理结果 ==================== #
        results = {
            # 基本信息
            'file_path': file_path,
            'file_name': os.path.basename(file_path).replace('.csv', ''),
            'n_samples': n_samples,
            'n_vars': n_vars,
            'var_names': available_vars,
            'order': self.order,
            'tau': self.tau,

            # # 原始数据统计
            # 'data_mean': data_mean_raw,
            # 'data_std': data_std_raw,

            # 张量
            'alpha': alpha,
            'A': A,
            'C': C,
            'E': E,

            # 有效Jacobian
            'J_eff': J_eff,
            'x_working_point': x_working_point,

            # 特征值
            'eigenvalues': eigvals,
            'eigenvectors': eigvecs,
            'eigenvalues_sorted': eigvals_sorted,
            'eigenvectors_sorted': eigvecs_sorted,
            'lambda_max': lambda_max,
            'lambda_min': lambda_min,

            # 稳定性
            'n_stable': n_stable,
            'n_unstable': n_unstable,
            'stability_status': stability_status,

            # 贡献度
            'contrib_max': contrib_max,
            'contrib_min': contrib_min,
            'var_contrib_stable': var_contrib_stable,
            'var_contrib_unstable': var_contrib_unstable,

            # 统计
            'A_mean': A.mean(),
            'A_std': A.std(),
            'A_min': A.min(),
            'A_max': A.max(),
            'C_norm': np.linalg.norm(C),
            'E_norm': np.linalg.norm(E),

            # 质量指标
            'sample_param_ratio': n_samples / self._count_params(n_vars)
        }

        # ==================== 8. 可视化 ==================== #
        if save_figures:
            if verbose:
                print(f"\n生成图表...")
            self._plot_summary(results, output_dir)
            self._plot_eigenvector_analysis(results, output_dir)

            if self.order >= 2:
                self._plot_nonlinear_effects(results, output_dir)

        if verbose:
            print(f"\n分析完成！结果保存在: {output_dir}")

        return results

    def _count_params(self, n_vars):
        """计算参数总数"""
        n = 1 + n_vars  # α + A
        if self.order >= 2:
            n += n_vars ** 3  # C
        if self.order >= 3:
            n += n_vars ** 4  # E
        return n

    def _plot_summary(self, results, output_dir):
        """绘制6子图汇总（美化版）"""

        # 设置现代化样式
        plt.style.use('seaborn-v0_8-darkgrid')  # 使用现代主题
        plt.rc('font', family='Arial', size=11)
        plt.rc('axes', linewidth=1.5)

        fig = plt.figure(figsize=(20, 11))  # 稍微大一点
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.4,
                               left=0.06, right=0.97, top=0.94, bottom=0.06)

        J = results['J_eff']
        n_vars = results['n_vars']
        var_names = results['var_names']
        eigvals = results['eigenvalues']
        lambda_max = results['lambda_max']
        lambda_min = results['lambda_min']
        contrib_max = results['contrib_max']
        contrib_min = results['contrib_min']

        # ==================== (a) Jacobian热图 - 改进版 ==================== #
        ax1 = fig.add_subplot(gs[0, 0])

        # 🔥 关键修复：调整颜色范围，确保对比度
        vmax = max(abs(J.min()), abs(J.max()))
        vmin = -vmax

        im = ax1.imshow(J, cmap='RdBu_r', aspect='auto',
                        norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
                        interpolation='nearest')

        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('$J_{eff}$', rotation=0, labelpad=25, fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        ax1.set_xticks(range(n_vars))
        ax1.set_yticks(range(n_vars))
        ax1.set_xticklabels(var_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax1.set_yticklabels(var_names, fontsize=11, fontweight='bold')
        ax1.set_title('(a) Effective Jacobian Matrix',
                      fontsize=15, pad=15, fontweight='bold', color='#2C3E50')

        # 网格线
        for i in range(n_vars + 1):
            ax1.axhline(i - 0.5, color='white', linewidth=1.2)
            ax1.axvline(i - 0.5, color='white', linewidth=1.2)

        # ==================== (b) 分布图 - 改进版 ==================== #
        ax2 = fig.add_subplot(gs[0, 1])

        J_diag = np.diag(J)
        J_off = J[~np.eye(n_vars, dtype=bool)]

        # 🔥 修复：合理的bins范围
        bins = np.linspace(min(J.min(), J_off.min()),
                           max(J.max(), J_off.max()), 25)

        ax2.hist(J_diag, bins=bins, alpha=0.8, label='Diagonal $J_{ii}$',
                 density=True, color='#3498DB', edgecolor='#2C3E50', linewidth=1.5)
        ax2.hist(J_off, bins=bins, alpha=0.7, label='Off-diagonal $J_{ij}$',
                 density=True, color='#E74C3C', edgecolor='#C0392B', linewidth=1.5)

        ax2.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Zero')
        ax2.set_xlabel('$J_{eff}$ Value', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True,
                   loc='best', framealpha=0.9)
        ax2.set_title('(b) Jacobian Distribution',
                      fontsize=15, pad=15, fontweight='bold', color='#2C3E50')
        ax2.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax2.tick_params(labelsize=10)

        # ==================== (c) 特征值谱 - 改进版 ==================== #
        ax3 = fig.add_subplot(gs[0, 2])

        # 主散点
        ax3.scatter(np.real(eigvals), np.imag(eigvals),
                    s=120, alpha=0.8, c='#3498DB', edgecolors='#2C3E50',
                    linewidths=2, zorder=3)

        # 最大特征值（红色五角星）
        ax3.scatter([np.real(lambda_max)], [np.imag(lambda_max)],
                    s=300, c='#E74C3C', marker='*', edgecolors='#C0392B',
                    linewidths=3, label=f'$\lambda_{{max}}$={np.real(lambda_max):.3f}',
                    zorder=5)

        # 最小特征值（绿色方块）
        ax3.scatter([np.real(lambda_min)], [np.imag(lambda_min)],
                    s=250, c='#2ECC71', marker='s', edgecolors='#27AE60',
                    linewidths=3, label=f'$\lambda_{{min}}$={np.real(lambda_min):.3f}',
                    zorder=5)

        # 参考线
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
        ax3.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2,
                    label='Stability boundary')

        ax3.set_xlabel('Re($\lambda$)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Im($\lambda$)', fontsize=13, fontweight='bold')
        ax3.set_title('(c) Eigenvalue Spectrum',
                      fontsize=15, pad=15, fontweight='bold', color='#2C3E50')
        ax3.legend(fontsize=10, frameon=True, fancybox=True, shadow=True,
                   loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax3.tick_params(labelsize=10)

        # 稳定区域标注
        xlim = ax3.get_xlim()
        ylim = ax3.get_ylim()
        if xlim[0] < 0 < xlim[1]:
            ax3.fill_betweenx(ylim, xlim[0], 0, alpha=0.15, color='green',
                              label='Stable region')
            ax3.fill_betweenx(ylim, 0, xlim[1], alpha=0.15, color='red')

        # ==================== (d) 最不稳定模态 - 改进版 ==================== #
        ax4 = fig.add_subplot(gs[1, 0])

        idx_sorted = np.argsort(contrib_max)

        # 渐变色
        colors = plt.cm.Reds(contrib_max[idx_sorted] / contrib_max.max())

        bars = ax4.barh(range(n_vars), contrib_max[idx_sorted] * 100,
                        color=colors, alpha=0.85, edgecolor='#2C3E50', linewidth=1.5)

        # 高亮主导变量
        for i, (idx, val) in enumerate(zip(idx_sorted, contrib_max[idx_sorted])):
            if val > 0.15:  # >15%
                bars[i].set_edgecolor('#C0392B')
                bars[i].set_linewidth(3)

        ax4.set_yticks(range(n_vars))
        ax4.set_yticklabels([var_names[i] for i in idx_sorted],
                            fontsize=11, fontweight='bold')
        ax4.set_xlabel('Contribution (%)', fontsize=13, fontweight='bold')
        ax4.set_title(f'(d) Most Unstable Mode\n($\lambda$={np.real(lambda_max):.4f})',
                      fontsize=15, pad=15, fontweight='bold', color='#E74C3C')
        ax4.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=0.8)
        ax4.tick_params(labelsize=10)

        # ==================== (e) 最稳定模态 - 改进版 ==================== #
        ax5 = fig.add_subplot(gs[1, 1])

        idx_sorted = np.argsort(contrib_min)

        # 渐变色
        colors = plt.cm.Greens(contrib_min[idx_sorted] / contrib_min.max())

        bars = ax5.barh(range(n_vars), contrib_min[idx_sorted] * 100,
                        color=colors, alpha=0.85, edgecolor='#2C3E50', linewidth=1.5)

        # 高亮主导变量
        for i, (idx, val) in enumerate(zip(idx_sorted, contrib_min[idx_sorted])):
            if val > 0.15:
                bars[i].set_edgecolor('#27AE60')
                bars[i].set_linewidth(3)

        ax5.set_yticks(range(n_vars))
        ax5.set_yticklabels([var_names[i] for i in idx_sorted],
                            fontsize=11, fontweight='bold')
        ax5.set_xlabel('Contribution (%)', fontsize=13, fontweight='bold')
        ax5.set_title(f'(e) Most Stable Mode\n($\lambda$={np.real(lambda_min):.4f})',
                      fontsize=15, pad=15, fontweight='bold', color='#2ECC71')
        ax5.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=0.8)
        ax5.tick_params(labelsize=10)

        # ==================== (f) 信息表格 - 改进版 ==================== #
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        # 状态标识
        status_emoji = {
            'Stable': '✓',
            'Critical': '⚠',
            'Unstable': '✗'
        }

        info_data = [
            ['Metric', 'Value'],
            ['━' * 18, '━' * 18],
            ['📊 Samples', f"{results['n_samples']:,}"],
            ['📈 Variables', f"{n_vars}"],
            ['🔢 Order', f"{results['order']}"],
            ['', ''],
            ['λ_max', f"{np.real(lambda_max):.6f}"],
            ['λ_min', f"{np.real(lambda_min):.6f}"],
            ['', ''],
            ['✓ Stable modes', f"{results['n_stable']}"],
            ['✗ Unstable modes', f"{results['n_unstable']}"],
            ['', ''],
            ['Status', f"{status_emoji.get(results['stability_status'], '')} {results['stability_status']}"],
        ]

        table = ax6.table(cellText=info_data, cellLoc='left', loc='center',
                          colWidths=[0.55, 0.45])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.2)

        # 表头样式
        for i in range(2):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white', size=12)
            table[(0, i)].set_height(0.08)

        # 分隔线
        table[(1, 0)].set_facecolor('#ECF0F1')
        table[(1, 1)].set_facecolor('#ECF0F1')

        # 状态行着色
        status_row = 12
        status_colors = {
            'Stable': '#D5F4E6',
            'Critical': '#FCF3CF',
            'Unstable': '#F5B7B1'
        }
        table[(status_row, 1)].set_facecolor(
            status_colors.get(results['stability_status'], '#FFFFFF'))
        table[(status_row, 1)].set_text_props(weight='bold', size=12)

        # 边框
        for key, cell in table.get_celld().items():
            cell.set_linewidth(1.5)
            cell.set_edgecolor('#95A5A6')

        # 总标题
        status_color = {
            'Stable': '#27AE60',
            'Critical': '#F39C12',
            'Unstable': '#E74C3C'
        }.get(results['stability_status'], '#2C3E50')

        fig.suptitle(f'Stability Analysis: {results["file_name"]} (Order={results["order"]})',
                     fontsize=18, fontweight='bold', color=status_color, y=0.98)

        # 保存
        plt.savefig(f'{output_dir}/figures/analysis_summary.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{output_dir}/figures/analysis_summary.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close()



    def _plot_eigenvector_analysis(self, results, output_dir):
        """特征向量详细分析（5子图）"""
        # ... (与您之前的代码基本相同)
        pass

    def _plot_nonlinear_effects(self, results, output_dir):
        """非线性效应分析（如果Order>=2）"""
        # 对比 A vs J_eff 的差异
        # 显示二阶/三阶项的贡献
        pass


# ===== 便捷函数 ===== #

def analyze_single_file(file_path, order=1, tau=1.0, output_dir=None, verbose=True):
    """
    快速分析单个文件

    Example:
    --------
    >>> result = analyze_single_file('DATA/04008_20230930.csv')
    >>> print(f"稳定性: {result['stability_status']}")
    >>> print(f"λ_max = {np.real(result['lambda_max']):.6f}")
    """
    analyzer = EMTransmitterStabilityAnalyzer(order=order, tau=tau)
    return analyzer.analyze_file(file_path, output_dir=output_dir, verbose=verbose)


# ===== 测试 ===== #
if __name__ == '__main__':
    import time

    test_file = r'D:\Graduate\Performance Evaluation\Code\Stability\StabilityAnalysis_latest\DATA\04008_20230930.csv'

    print("测试Order=1...")
    start = time.time()
    result1 = analyze_single_file(test_file, order=1)
    print(f"用时: {time.time() - start:.2f}秒\n")

    print("测试Order=2...")
    start = time.time()
    result2 = analyze_single_file(test_file, order=2)
    print(f"用时: {time.time() - start:.2f}秒")