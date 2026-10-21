"""生成一个微缩版 CD2025 结构的合成数据集，用于在真实数据下载完成前
验证 HermNet 训练/测试全流程能跑通（冒烟测试）。
结构: dataset/CD2025/L1.h5 ... L4.h5, 键 X:[N,2,1024], Y:[N]
合成 4 类"调制"（相位/频率/幅度模式不同），类数用 4 加快收敛。
"""
import h5py
import numpy as np
import os

np.random.seed(0)
os.makedirs('./dataset/CD2025', exist_ok=True)

N_PER_CLS = 50          # 每类样本数（真实数据是500，这里缩小10倍）
NUM_CLS = 4             # 冒烟测试用4类
L = 1024
t = np.arange(L)

def make_sample(cls, phase_shift=0.0, freq_scale=1.0):
    f = (5 + 3 * cls) * freq_scale          # 每类不同频率
    amp = 1.0 + 0.3 * cls                    # 每类不同幅度
    sig = amp * np.sin(2 * np.pi * f * t / L + phase_shift)
    noise = 0.1 * np.random.randn(L)
    i = sig + noise
    q = amp * np.cos(2 * np.pi * f * t / L + phase_shift) + 0.1 * np.random.randn(L)
    return np.stack([i, q]).astype(np.float32)

for domain_idx, name in enumerate(['L1', 'L2', 'L3', 'L4']):
    X, Y = [], []
    # 每个"场景"加不同的公共相位偏移和频率偏移，模拟域偏移
    phase_shift = domain_idx * 0.7
    freq_scale = 1.0 + 0.05 * domain_idx
    for cls in range(NUM_CLS):
        for _ in range(N_PER_CLS):
            X.append(make_sample(cls, phase_shift + 0.1*np.random.randn(),
                                 freq_scale * (1 + 0.02*np.random.randn())))
            Y.append(cls)
    X = np.stack(X); Y = np.array(Y, dtype=np.int64)
    with h5py.File(f'./dataset/CD2025/{name}.h5', 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)
    print(f'{name}.h5  X{X.shape}  Y{Y.shape}')
print('合成数据生成完毕')
