import os
import numpy as np
import pandas as pd
import scipy.signal
from scipy.optimize import nnls
from sklearn.linear_model import Lasso

def simpleoutput(range):
    print('this is a simple python output')
    print('random data is: ')
    np.random.seed(0)
    res= np.random.rand(3,range) # 3 rows, range columns
    print('output matrix is: ')
    print(res)
    print('done')
    return res

def mix(a,b):
    r1 = a + b
    r2 = a - b
    return (r1, r2)

# 以上皆为测试
##################################################################
# 以下为解混相关代码

# 创建光谱库
def create_library(directory):
    library = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            mine_name = filename.split(".")[0]
            with open(os.path.join(directory, filename), 'r') as f:
                df = pd.read_csv(f)
                library[mine_name] = df
    return library

# 对数变换
def log_transform(spectrum, attr='reflectance'):
    y = spectrum[attr]
    y_new = np.log10(y)
    spectrum['log'] = y_new
    return y_new

# 基线校正
def baseline_correction(spectrum, attr='log', lam=1e5, p=0.01, niter=10):
    y = spectrum[attr]
    y_base = scipy.signal.savgol_filter(y, 21, 3)
    y_new = spectrum['baseline'] =y-y_base
    return y_new

# 归一化
def normalize(spectrum, attr='log'):
    y = spectrum[attr]
    y_new = (y - y.min()) / (y.max() - y.min())
    spectrum['normalized'] = y_new
    return y_new

# 噪声去除
def denoise(spectrum, attr='log'):
    y = spectrum[attr]
    y_new = scipy.signal.wiener(y, mysize=5)
    spectrum['denoised'] = y_new
    return y_new

# 预处理
def preprocess(spectrum):
    log_transform(spectrum)
    baseline_correction(spectrum)
    normalize(spectrum, 'baseline')
    # denoise(spectrum)
    return spectrum

# 矩阵化
def get_matrix(spectrum_lib, test_spectrum, attr='normalized'):
    # 从字典中取出所有的光谱数据
    normalized_arrays = [df[attr].values for df in spectrum_lib.values()]
    # 将所有的光谱数据合并成一个矩阵，每一行代表一个光谱数据
    merged_array = np.vstack(normalized_arrays)
    # 将矩阵转置
    transposed_array = merged_array.T
    # 取出测试光谱数据
    test_array = test_spectrum[attr].values
    return transposed_array, test_array

def demix(mixing_spectrum: pd.DataFrame):
    mixing_spectrum = pd.DataFrame(mixing_spectrum, columns=['wavelength', 'reflectance'])
    # 定义数据目录
    directories = [
        "./data/library/txt/",
        "./data/library/csv/",
        "./data/test/single/txt/",
        "./data/test/single/csv/",
        "./data/test/multi/txt/",
        "./data/test/multi/csv/",
        ]
    spectrum_lib = create_library(directories[1])
    if len(spectrum_lib) == 0:
        print("No spectrum library found.")
        return [-1]
    else:
        print(f"Found {len(spectrum_lib)} spectra in the library.")
        # return [len(spectrum_lib)]
    for mine_name, df in spectrum_lib.items():
        preprocess(df)
    preprocess(mixing_spectrum)
    A, y = get_matrix(spectrum_lib, mixing_spectrum, 'log')
    # 使用 scipy.optimize.nnls 进行优化
    x, rnorm = nnls(A, y)
    # 记录丰度大于阈值的成分
    components = []
    for i, mine_name in enumerate(spectrum_lib.keys()):
        if x[i] > 0.0:
            components.append((mine_name, x[i]))
    # 从大到小排序
    components.sort(key=lambda x: x[1], reverse=True)
    # 归一化
    total = sum([abundance for mine_name, abundance in components])
    components = [(mine_name, abundance/total) for mine_name, abundance in components]
    # 输出结果
    for mine_name, abundance in components:
        print(f'{mine_name}: {abundance*100:.2f}%')
    return components

def demix_lasso(mixing_spectrum: pd.DataFrame):
    mixing_spectrum = pd.DataFrame(mixing_spectrum, columns=['wavelength', 'reflectance'])
    # 定义数据目录
    directories = [
        "./data/library/txt/",
        "./data/library/csv/",
        "./data/test/single/txt/",
        "./data/test/single/csv/",
        "./data/test/multi/txt/",
        "./data/test/multi/csv/",
        "D:/codesaves/python/MineralSpectralAnalysis/src/data/library/"
        ]
    spectrum_lib = create_library(directories[1])
    if len(spectrum_lib) == 0:
        print("No spectrum library found.")
        return [-1]
    else:
        print(f"Found {len(spectrum_lib)} spectra in the library.")
        # return [len(spectrum_lib)]
    
    for mine_name, df in spectrum_lib.items():
        preprocess(df)
    preprocess(mixing_spectrum)
    
    X_known, y_unknown = get_matrix(spectrum_lib, mixing_spectrum, 'log')
    X_known = np.array(X_known).T
    
    # 创建LASSO回归模型，并拟合已知光谱端元数据
    lasso = Lasso(alpha=0.0005, positive=True)
    lasso.fit(X_known.T, y_unknown)
    
    # 获取端元系数（即端元的权重）
    endmember_coefficients = lasso.coef_
    
    # 获得拟合光谱
    fitted_mixture_spectrum_log_org = np.dot(X_known.T, endmember_coefficients)
    fitted_mixture_spectrum_org_org=fitted_mixture_spectrum_log_org
    
    # 处理端元系数
    mines=spectrum_lib.keys()
    
    # 提取权重大于0的端元
    endmember_coefficients_less = [(mine_name, coef) for mine_name, coef in zip(mines, endmember_coefficients) if coef > 0]
    
    # 对权重进行排序
    endmember_coefficients_less.sort(key=lambda x: x[1], reverse=True)
    
    # 只保留前max_mines个端元
    endmember_coefficients_less = endmember_coefficients_less[:2]
    
    # 归一化权重，使其成为比例 for reporting
    total = sum([coef for mine_name, coef in endmember_coefficients_less])
    endmember_coefficients_less = [(mine_name, coef/total) for mine_name, coef in endmember_coefficients_less]
    
    # 剔除权重小于min_account的端元
    endmember_coefficients_less = [(mine_name, coef) for mine_name, coef in endmember_coefficients_less if coef > 0.1]
    
    # 归一化权重，使其成为比例 for reporting
    total = sum([coef for mine_name, coef in endmember_coefficients_less])
    endmember_coefficients_less = [(mine_name, coef/total) for mine_name, coef in endmember_coefficients_less]

    return endmember_coefficients_less

# 函数测试
# un=pd.read_csv('./data/test/multi/csv/al20ka80.csv')
un=pd.read_csv('D:/codesaves/python/MineralSpectralAnalysis/src/data/unknown/al20ka80.csv')
print(demix_lasso(un))