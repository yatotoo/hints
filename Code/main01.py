import numpy as np
import pandas as pd

file_path = r'D:\Graduate\Performance Evaluation\Code\Stability\StabilityAnalysis_latest\DATA\04008_20230530.csv'
df = pd.read_csv(file_path)

print("="*60)
print("数据基本信息")
print("="*60)

print(f"\n数据形状: {df.shape}")
print(f"  时间点数: {df.shape[0]}")
print(f"  变量数: {df.shape[1]}")

print("\n列名:")
print(df.columns.tolist())

print("\n前10行:")
print(df.head(10))

print("\n数据类型:")
print(df.dtypes)

print("\n缺失值统计:")
print(df.isnull().sum())

print("\n数据统计:")
print(df.describe())

print("\n" + "="*60)

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# 选择用于分析的物理变量
analysis_vars = ['Toff', 'U', 'I', 'Freq', 'T1', 'T2', 'T3', 'AC', 'R', 'L']

# 提取数据
data = df[analysis_vars].values
print(f"\n原始数据形状: {data.shape}")

# 可视化检查数据
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, var in enumerate(analysis_vars):
    axes[i].plot(df[var], linewidth=0.5)
    # axes[i].plot(df[var][:1000], linewidth=0.5)
    axes[i].set_title(var)
    axes[i].set_xlabel('Time')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_preview.png', dpi=150)
plt.show()

print("\n数据预览图已保存")
print("="*60)
print("="*60)

# 计算时间步长
time_values = df['Time'].values
dt_samples = np.diff(time_values[:1000])  # 取前1000个样本
dt = np.median(dt_samples[dt_samples > 0])  # 排除0
print(f"时间步长 dt: {dt} (采样单位)")

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

print(f"\n归一化后:")
print(f"  均值: {data_normalized.mean(axis=0)}")  # 应该接近0
print(f"  标准差: {data_normalized.std(axis=0)}")  # 应该接近1

# 保存预处理后的数据
np.save('outputs/04009/power_electronics_data.npy', data_normalized)
np.save('outputs/04009/variable_names.npy', np.array(analysis_vars))

print(f"\n数据已保存到 ./outputs/power_electronics_data.npy")
print(f"变量名已保存到 ./outputs/variable_names.npy")

from sliding_window_analysis import extract_tensors_over_time, analyze_stability_over_time

# 加载数据
trajectory = np.load('outputs/04009/power_electronics_data.npy')
var_names = np.load('outputs/04009/variable_names.npy', allow_pickle=True)

print(f"\n完整数据: {trajectory.shape}")

# 先用部分数据测试（避免计算太久）
# 取前50000个点
# trajectory_subset = trajectory[:50000]
# 使用全部数据
trajectory_subset = trajectory
print(f"测试数据: {trajectory_subset.shape}")

# 设置参数
dt = 1.0  # 归一化的时间步长
window_size = 300  # 窗口大小
overlap_ratio = 0.95  # 95%重叠

print(f"\n滑动窗口参数:")
print(f"  窗口大小: {window_size}")
print(f"  重叠率: {overlap_ratio}")
print(f"  预计窗口数: {(len(trajectory_subset) - window_size) // int(window_size * (1 - overlap_ratio)) + 1}")

# 提取张量序列
print("\n开始提取张量...")
tensor_seq = extract_tensors_over_time(
    trajectory_subset,
    window_size=window_size,
    overlap_ratio=overlap_ratio,
    order=3,
    dt=dt,
    normalize=True
)

print(f"\n提取了 {len(tensor_seq['time'])} 个窗口")
print(f"A 矩阵序列形状: {tensor_seq['A'].shape}")

# 分析稳定性
print("\n分析稳定性...")
stability_results = analyze_stability_over_time(
    tensor_seq,
    trajectory_subset,
    window_size=window_size,
    overlap_ratio=overlap_ratio,
    dt=dt
)

# 保存结果
np.save('outputs/04009/power_electronics_tensors.npy', tensor_seq)
np.save('outputs/04009/power_electronics_stability.npy', stability_results)

print(f"\n结果已保存！")
print("="*60)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

# # 尝试启用LaTeX
try:
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    print("LaTeX渲染已启用")
except:
    print("LaTeX不可用，使用普通文本")
    plt.rc('text', usetex=False)
# 加载结果
tensor_seq = np.load('outputs/04009/power_electronics_tensors.npy', allow_pickle=True).item()
stability_results = np.load('outputs/04009/power_electronics_stability.npy', allow_pickle=True).item()
var_names = np.load('outputs/04009/variable_names.npy', allow_pickle=True)

print(f"变量名: {var_names}")

# 创建图
fig = plt.figure(figsize=(24, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                      left=0.08, right=0.95, top=0.95, bottom=0.08)

# (a) A矩阵热图
ax1 = fig.add_subplot(gs[0, 0])
A_mean = np.mean(tensor_seq['A'], axis=0)
im = ax1.imshow(A_mean, cmap='RdBu_r', aspect='auto',
               norm=TwoSlopeNorm(vmin=A_mean.min(), vcenter=0, vmax=A_mean.max()))
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('$A_{ij}^{eff} (mean)', rotation=0, labelpad=25, fontsize=16)
ax1.set_xticks(range(len(var_names)))
ax1.set_yticks(range(len(var_names)))
ax1.set_xticklabels(var_names, rotation=45, ha='right', fontsize=12)
ax1.set_yticklabels(var_names, fontsize=12)
ax1.set_title('(a) Time-averaged interactions', fontsize=18, pad=10)

# (b) 分布
ax2 = fig.add_subplot(gs[0, 1])
A_diag = np.concatenate([np.diag(A) for A in tensor_seq['A']])
A_off = np.concatenate([A[~np.eye(A.shape[0], dtype=bool)] for A in tensor_seq['A']])
ax2.hist(A_diag, bins=60, alpha=0.7, label='A_ii',
        density=True, color='lightblue', edgecolor='blue')
ax2.hist(A_off, bins=60, alpha=0.7, label='A_ij (i!=j)',
        density=True, color='coral', edgecolor='red')
ax2.set_xlabel('$A_{ij}^{eff}$', fontsize=16)
ax2.set_ylabel('PDF', fontsize=16)
ax2.legend()
ax2.set_title('(b) Distribution', fontsize=18, pad=10)
ax2.grid(True, alpha=0.3)

# (c) 张量分布
ax3_container = fig.add_subplot(gs[0, 2])
ax3_container.axis('off')
gs3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.3)

ax3_top = fig.add_subplot(gs3[0])
ax3_top.hist(tensor_seq['C'].flatten(), bins=80, alpha=0.8, density=True, color='steelblue')
ax3_top.set_ylabel('PDF', fontsize=14)
ax3_top.set_yscale('log')
ax3_top.set_title('(c) Tensor distributions', fontsize=18, pad=10)
ax3_top.tick_params(labelbottom=False)
ax3_top.text(0.95, 0.95, '$C_{ijk}$', transform=ax3_top.transAxes,
            fontsize=14, va='top', ha='right')

ax3_bottom = fig.add_subplot(gs3[1])
ax3_bottom.hist(tensor_seq['E'].flatten(), bins=80, alpha=0.8, density=True, color='royalblue')
ax3_bottom.set_xlabel('Value', fontsize=16)
ax3_bottom.set_ylabel('PDF', fontsize=14)
ax3_bottom.set_yscale('log')
ax3_bottom.text(0.95, 0.95, 'E_{ijkl}', transform=ax3_bottom.transAxes,
               fontsize=14, va='top', ha='right')

# (d) 特征值演化
ax4_container = fig.add_subplot(gs[1, 0])
ax4_container.axis('off')
gs4 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.1)

time_points = tensor_seq['time']
lambda_max_real = np.real(stability_results['lambda_max'])
lambda_min_real = np.real(stability_results['lambda_min'])

ax4_top = fig.add_subplot(gs4[0])
ax4_top.plot(time_points, lambda_max_real, 'k-', linewidth=2)
ax4_top.set_ylabel('Re($\lambda_{max}$)', fontsize=14)
ax4_top.set_title('(d) Eigenvalue evolution', fontsize=18, pad=10)
ax4_top.tick_params(labelbottom=False)
ax4_top.grid(True, alpha=0.3)

ax4_bottom = fig.add_subplot(gs4[1])
ax4_bottom.plot(time_points, lambda_min_real, 'r-', linewidth=2)
ax4_bottom.set_xlabel('Time (samples)', fontsize=16)
ax4_bottom.set_ylabel('Re($\lambda_{min}$)', fontsize=14)
ax4_bottom.grid(True, alpha=0.3)

# (e) 不动点数量
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(time_points, stability_results['n_unstable'],
           c='red', s=30, alpha=0.6, label='Unstable', edgecolors='darkred')
ax5.scatter(time_points, stability_results['n_stable'],
           c='blue', s=30, alpha=0.6, label='Stable', edgecolors='darkblue')
ax5.set_xlabel('Time (samples)', fontsize=16)
ax5.set_ylabel('Number of fixed points', fontsize=16)
ax5.set_title('(e) Fixed points evolution', fontsize=18, pad=10)
ax5.legend()
ax5.grid(True, alpha=0.3)

# (f) 特征值谱
ax6 = fig.add_subplot(gs[1, 2])
all_eig = np.concatenate(stability_results['all_eigenvalues'])
ax6.scatter(np.real(all_eig), np.imag(all_eig),
           alpha=0.4, s=15, c='steelblue')
ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax6.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax6.set_xlabel('Re(lambda)', fontsize=16)
ax6.set_ylabel('Im(lambda)', fontsize=16)
ax6.set_title('(f) Eigenvalue spectrum', fontsize=18, pad=10)
ax6.grid(True, alpha=0.3)

plt.savefig('./outputs/power_electronics_fig3.png', dpi=300, bbox_inches='tight')
plt.savefig('./outputs/power_electronics_fig3.pdf', bbox_inches='tight')
plt.show()

# 不动点未分离
print("\n图已生成！")
print("PNG: ./outputs/power_electronics_fig3.png")
print("PDF: ./outputs/power_electronics_fig3.pdf")
print("="*60)

import os

# 确保目录存在
os.makedirs('outputs/figures', exist_ok=True)

# 关闭LaTeX
plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')

# 加载数据
tensor_seq = np.load('outputs/tensors/tensor_seq_filtered.npy', allow_pickle=True).item()
stability_results = np.load('outputs/stability/stability_results.npy', allow_pickle=True).item()
var_names = np.load('outputs/data/variable_names.npy', allow_pickle=True)

print(f"数据加载完成")
print(f"  窗口数: {len(tensor_seq['time'])}")
print(f"  变量数: {len(var_names)}")

# ==================== 创建图 ==================== #
fig = plt.figure(figsize=(24, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                      left=0.08, right=0.96, top=0.95, bottom=0.08)

# 论文配色方案
color_diag = '#5DADE2'      # 淡蓝色（对角线）
color_offdiag = '#EC7063'   # 珊瑚红（非对角线）
color_stable = '#3498DB'    # 蓝色（稳定）
color_unstable = '#E74C3C'  # 红色（不稳定）

# ==================== (a) A矩阵热图 ==================== #
ax1 = fig.add_subplot(gs[0, 0])
A_mean = np.mean(tensor_seq['A'], axis=0)

im = ax1.imshow(A_mean, cmap='RdBu_r', aspect='auto',
               norm=TwoSlopeNorm(vmin=A_mean.min(), vcenter=0, vmax=A_mean.max()))

cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('$A_{ij}$ (mean)', rotation=0, labelpad=30, fontsize=15)
cbar.ax.tick_params(labelsize=12)

ax1.set_xticks(range(len(var_names)))
ax1.set_yticks(range(len(var_names)))
ax1.set_xticklabels(var_names, rotation=45, ha='right', fontsize=12)
ax1.set_yticklabels(var_names, fontsize=12)
ax1.set_title('(a) Time-averaged interactions', fontsize=17, pad=12, fontweight='bold')

# 添加网格线
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)

# ==================== (b) 分布 ==================== #
ax2 = fig.add_subplot(gs[0, 1])

A_diag = np.concatenate([np.diag(A) for A in tensor_seq['A']])
A_off = np.concatenate([A[~np.eye(A.shape[0], dtype=bool)] for A in tensor_seq['A']])

ax2.hist(A_diag, bins=50, alpha=0.75, label='Diagonal $A_{ii}$',
        density=True, color=color_diag, edgecolor='white', linewidth=0.5)
ax2.hist(A_off, bins=50, alpha=0.75, label='Off-diagonal $A_{ij}$ ($i≠j$)',
        density=True, color=color_offdiag, edgecolor='white', linewidth=0.5)

ax2.set_xlabel('$A_{ij}$ value', fontsize=15)
ax2.set_ylabel('Probability density', fontsize=15)
ax2.legend(fontsize=13, frameon=True, fancybox=True, shadow=True)
ax2.set_title('(b) Distribution of interaction strengths', fontsize=17, pad=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=12)

# ==================== (c) C张量分布（仅二阶） ==================== #
ax3 = fig.add_subplot(gs[0, 2])

C_flat = tensor_seq['C'].flatten()
ax3.hist(C_flat, bins=80, alpha=0.85, density=True,
        color='#3498DB', edgecolor='white', linewidth=0.3)

ax3.set_xlabel('$C_{ijk}$ value', fontsize=15)
ax3.set_ylabel('Probability density', fontsize=15)
ax3.set_yscale('log')
ax3.set_title('(c) Second-order tensor distribution', fontsize=17, pad=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--', which='both')
ax3.tick_params(labelsize=12)

# 添加统计信息
C_mean = np.mean(C_flat)
C_std = np.std(C_flat)
ax3.text(0.98, 0.97, f'$\mu$ = {C_mean:.2e}\n$\sigma$ = {C_std:.2e}',
        transform=ax3.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ==================== (d) 特征值演化 ==================== #
ax4_container = fig.add_subplot(gs[1, 0])
ax4_container.axis('off')
gs4 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.12)

time_points = tensor_seq['time']
lambda_max_real = np.real(stability_results['lambda_max'])
lambda_min_real = np.real(stability_results['lambda_min'])

# lambda_max
ax4_top = fig.add_subplot(gs4[0])
ax4_top.plot(time_points, lambda_max_real, 'k-', linewidth=1.8, alpha=0.8)
ax4_top.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)

# 标注关键事件
event_time = 1.1e6
ax4_top.axvline(event_time, color='#E67E22', linestyle='--', alpha=0.8, linewidth=2)
ax4_top.text(event_time, ax4_top.get_ylim()[1]*0.85, 'Critical event',
            fontsize=11, color='#E67E22', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', alpha=0.8))

ax4_top.set_ylabel('Re($\lambda_{max}$)', fontsize=14)
ax4_top.set_title('(d) Eigenvalue evolution', fontsize=17, pad=12, fontweight='bold')
ax4_top.tick_params(labelbottom=False, labelsize=12)
ax4_top.grid(True, alpha=0.3, linestyle='--')

# lambda_min
ax4_bottom = fig.add_subplot(gs4[1])
ax4_bottom.plot(time_points, lambda_min_real, color='#C0392B', linewidth=1.8, alpha=0.8)
ax4_bottom.axvline(event_time, color='#E67E22', linestyle='--', alpha=0.8, linewidth=2)

ax4_bottom.set_xlabel('Time (samples)', fontsize=15)
ax4_bottom.set_ylabel('Re($\lambda_{min}$)', fontsize=14)
ax4_bottom.grid(True, alpha=0.3, linestyle='--')
ax4_bottom.tick_params(labelsize=12)

# ==================== (e) 稳定性模态 ==================== #
ax5 = fig.add_subplot(gs[1, 1])

ax5.scatter(time_points, stability_results['n_unstable'],
           c=color_unstable, s=25, alpha=0.6, label='Unstable modes',
           edgecolors='darkred', linewidths=0.5)
ax5.scatter(time_points, stability_results['n_stable'],
           c=color_stable, s=25, alpha=0.6, label='Stable modes',
           edgecolors='darkblue', linewidths=0.5)

# 标注关键事件
ax5.axvline(event_time, color='#E67E22', linestyle='--', alpha=0.8, linewidth=2)

ax5.set_xlabel('Time (samples)', fontsize=15)
ax5.set_ylabel('Number of modes', fontsize=15)
ax5.set_title('(e) Stability mode evolution', fontsize=17, pad=12, fontweight='bold')
ax5.legend(fontsize=13, loc='upper right', frameon=True, fancybox=True, shadow=True)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.tick_params(labelsize=12)

# ==================== (f) 特征值谱 ==================== #
ax6 = fig.add_subplot(gs[1, 2])

all_eig = np.concatenate(stability_results['all_eigenvalues'])
ax6.scatter(np.real(all_eig), np.imag(all_eig),
           alpha=0.4, s=12, c='#3498DB', edgecolors='none')

# 添加参考线
ax6.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax6.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

ax6.set_xlabel('Re($\lambda$)', fontsize=15)
ax6.set_ylabel('Im($\lambda$)', fontsize=15)
ax6.set_title('(f) Eigenvalue spectrum', fontsize=17, pad=12, fontweight='bold')
ax6.grid(True, alpha=0.3, linestyle='--')
ax6.tick_params(labelsize=12)

# 添加稳定性边界标注
ax6.text(0.02, 0.98, 'Stable region\n(Re($\lambda$) < 0)',
        transform=ax6.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ==================== 保存 ==================== #
plt.savefig('outputs/figures/fig3_final.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figures/fig3_final.pdf', bbox_inches='tight')
plt.show()

print("\n最终版 Fig.3 已生成！")
print("  PNG: outputs/figures/fig3_final.png")
print("  PDF: outputs/figures/fig3_final.pdf")
print("="*60)





