import numpy as np
from tqdm import tqdm
from interaction_extraction import InteractionExtractor


def extract_tensors_over_time(trajectory, window_size, overlap_ratio,
                              order=3, dt=0.01, normalize=True):
    """
    滑动窗口提取张量序列

    Parameters:
    -----------
    trajectory : np.ndarray, shape (T, N)
        完整时间序列
    window_size : int
        窗口大小（时间步数）
    overlap_ratio : float
        重叠率（0-1），例如 0.95 表示 95% 重叠
    order : int
        张量阶数
    dt : float
        时间步长
    normalize : bool
        是否归一化

    Returns:
    --------
    dict : 包含时间序列的张量
        {
            'time': 时间点列表,
            'alpha': alpha 序列,
            'A': A 矩阵序列,
            'C': C 张量序列,
            'E': E 张量序列
        }
    """
    T, N = trajectory.shape
    stride = int(window_size * (1 - overlap_ratio))

    # 计算窗口数量
    n_windows = (T - window_size) // stride + 1

    print(f"滑动窗口分析:")
    print(f"  总长度: {T}")
    print(f"  窗口大小: {window_size}")
    print(f"  步长: {stride}")
    print(f"  窗口数量: {n_windows}")

    # 初始化存储
    alpha_list = []
    A_list = []
    C_list = []
    E_list = []
    time_list = []

    # 创建提取器
    extractor = InteractionExtractor(order=order, tau=dt)

    # 滑动窗口
    for i in tqdm(range(n_windows), desc="提取张量"):
        start = i * stride
        end = start + window_size

        # 提取窗口数据
        window_data = trajectory[start:end]

        # 提取张量
        tensors = extractor.extract(window_data, normalize=normalize)

        # 保存
        alpha_list.append(tensors['alpha'])
        A_list.append(tensors['A'])
        C_list.append(tensors['C'])
        E_list.append(tensors['E'])
        time_list.append((start + end) / 2 * dt)  # 窗口中心时间

    return {
        'time': np.array(time_list),
        'alpha': np.array(alpha_list),
        'A': np.array(A_list),
        'C': np.array(C_list),
        'E': np.array(E_list)
    }


def analyze_stability_over_time(tensor_sequence, trajectory, window_size,
                                overlap_ratio, dt=0.01):
    """分析稳定性随时间的演化"""
    from stability_analysis import StabilityAnalyzer

    n_windows = len(tensor_sequence['time'])
    T, N = trajectory.shape
    stride = int(window_size * (1 - overlap_ratio))

    lambda_max_list = []
    lambda_min_list = []
    n_stable_list = []
    n_unstable_list = []
    all_eigenvalues = []
    all_eigenvectors = []

    print("\n分析稳定性...")

    for i in tqdm(range(n_windows), desc="稳定性分析"):
        start = i * stride
        end = start + window_size
        window_data = trajectory[start:end]

        analyzer = StabilityAnalyzer(
            tensor_sequence['alpha'][i],
            tensor_sequence['A'][i],
            tensor_sequence['C'][i],
            tensor_sequence['E'][i]
        )

        fixed_point = np.mean(window_data, axis=0)
        J = analyzer.compute_jacobian(fixed_point)

        # 计算特征值和特征向量
        eigvals, eigvecs = np.linalg.eig(J)

        # 排序
        idx = np.argsort(np.real(eigvals))
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        lambda_max_list.append(eigvals[-1])
        lambda_min_list.append(eigvals[0])
        all_eigenvalues.append(eigvals)
        all_eigenvectors.append(eigvecs)

        n_stable = np.sum(np.real(eigvals) < 0)
        n_unstable = np.sum(np.real(eigvals) >= 0)
        n_stable_list.append(n_stable)
        n_unstable_list.append(n_unstable)

    return {
        'lambda_max': np.array(lambda_max_list),
        'lambda_min': np.array(lambda_min_list),
        'n_stable': np.array(n_stable_list),
        'n_unstable': np.array(n_unstable_list),
        'all_eigenvalues': all_eigenvalues,
        'all_eigenvectors': all_eigenvectors
    }


# 在分析稳定性时，过滤掉病态的Jacobian
def analyze_stability_over_time_robust(tensor_sequence, trajectory, window_size,
                                       overlap_ratio, dt=0.01):
    """改进版：过滤异常值"""
    from stability_analysis import StabilityAnalyzer

    n_windows = len(tensor_sequence['time'])
    T, N = trajectory.shape
    stride = int(window_size * (1 - overlap_ratio))

    lambda_max_list = []
    lambda_min_list = []
    n_stable_list = []
    n_unstable_list = []
    all_eigenvalues = []
    all_eigenvectors = []
    valid_indices = []  # 记录有效的窗口

    print("\n分析稳定性（带异常值过滤）...")

    for i in tqdm(range(n_windows), desc="稳定性分析"):
        start = i * stride
        end = start + window_size
        window_data = trajectory[start:end]

        try:
            analyzer = StabilityAnalyzer(
                tensor_sequence['alpha'][i],
                tensor_sequence['A'][i],
                tensor_sequence['C'][i],
                tensor_sequence['E'][i]
            )

            fixed_point = np.mean(window_data, axis=0)
            J = analyzer.compute_jacobian(fixed_point)

            # 检查Jacobian是否正常
            if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                continue

            eigvals, eigvecs = np.linalg.eig(J)

            # 过滤异常特征值
            if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
                continue

            # 过滤过大的特征值（阈值可调）
            if np.max(np.abs(np.real(eigvals))) > 1e6:
                continue

            # 排序
            idx = np.argsort(np.real(eigvals))
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            lambda_max_list.append(eigvals[-1])
            lambda_min_list.append(eigvals[0])
            all_eigenvalues.append(eigvals)
            all_eigenvectors.append(eigvecs)
            valid_indices.append(i)

            n_stable = np.sum(np.real(eigvals) < 0)
            n_unstable = np.sum(np.real(eigvals) >= 0)
            n_stable_list.append(n_stable)
            n_unstable_list.append(n_unstable)

        except:
            continue

    print(f"\n有效窗口: {len(valid_indices)} / {n_windows}")

    return {
        'lambda_max': np.array(lambda_max_list),
        'lambda_min': np.array(lambda_min_list),
        'n_stable': np.array(n_stable_list),
        'n_unstable': np.array(n_unstable_list),
        'all_eigenvalues': all_eigenvalues,
        'all_eigenvectors': all_eigenvectors,
        'valid_indices': valid_indices,
        'time': tensor_sequence['time'][valid_indices]
    }