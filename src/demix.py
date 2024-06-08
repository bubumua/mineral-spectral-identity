import os
import pandas as pd
import numpy as np
import scipy.signal
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from scipy.optimize import nnls
from sklearn.model_selection import cross_val_score

def create_library(directory:str)->list:
    """给定一个目录，返回一个包含所有光谱数据的字典

    Args:
        directory (string): string of path of the directory

    Returns:
        dict: a dictionary with mine names as keys and pandas dataframes as values
    """
    library = {}
    for filename in os.listdir(directory):
        # 筛选csv文件
        if filename.endswith('.csv'):
            # 去除后缀，只保留矿物名称
            mine_name = filename.split(".")[0]
            # 读取为一个pandas dataframe
            with open(os.path.join(directory, filename), 'r') as f:
                df = pd.read_csv(f)
                # 以矿物名为键，dataframe为值
                library[mine_name] = df
    return library

def log_transform(spectrum, attr='reflectance'):
    """
    Applies a logarithmic transformation to the specified attribute of the spectrum DataFrame.
    
    Parameters:
        spectrum (pd.DataFrame): The input DataFrame containing the spectrum data.
        attr (str, optional): The attribute column to be transformed. Defaults to 'reflectance'.
    
    Returns:
        list: The transformed attribute values as a list.
    """
    y = spectrum[attr]
    y_new = np.log10(y)
    spectrum['log'] = y_new
    return y_new

def baseline_correction(spectrum:pd.DataFrame, attr='log', win_length=51, p=3)->list:
    """
    Perform baseline correction on the specified attribute of the spectrum DataFrame.

    Parameters
    ----------
    spectrum : pd.DataFrame
        Input DataFrame containing the spectrum data.
    attr : str, optional
        The attribute column to perform baseline correction on. Defaults to 'log'.
    win_length : int, optional
        The window length of the Savitzky-Golay filter. Defaults to 21.
    p : int, optional
        The polynomial order of the Savitzky-Golay filter. Defaults to 3.

    Returns
    -------
    list
        The corrected attribute values as a list.
    """
    y = spectrum[attr]
    y_base = scipy.signal.savgol_filter(y, win_length, polyorder=p)
    y_new = spectrum['baseline'] =y-y_base
    return y_new

def normalize(spectrum:pd.DataFrame, attr='log')->list:
    """
    Normalize the specified attribute of the spectrum DataFrame.

    Parameters
    ----------
    spectrum : pd.DataFrame
        Input DataFrame containing the spectrum data.
    attr : str, optional
        The attribute column to be normalized. Defaults to 'log'.

    Returns
    -------
    list
        The normalized attribute values as a list.
    """
    y = spectrum[attr]
    y_new = (y - y.min()) / (y.max() - y.min())
    spectrum['normalized'] = y_new
    return y_new

def denoise(spectrum:pd.DataFrame, attr='log',size=5)->list:
    """
    Apply Wiener filter for denoising to the specified attribute of the spectrum DataFrame.

    Parameters
    ----------
    spectrum : pd.DataFrame
        Input DataFrame containing the spectrum data.
    attr : str, optional
        The attribute column to be denoised. Defaults to 'log'.
    size : int, optional
        The size of the Wiener filter window. Defaults to 5.

    Returns
    -------
    list
        The denoised attribute values as a list.
    """
    y = spectrum[attr]
    y_new = scipy.signal.wiener(y, mysize=size)
    spectrum['denoised'] = y_new
    return y_new

def preprocess(spectrum:pd.DataFrame)->pd.DataFrame:
    """
    Preprocess the spectrum DataFrame by applying a series of operations including log transformation,
    baseline correction, normalization, and denoising.

    Parameters
    ----------
    spectrum : pd.DataFrame
        Input DataFrame containing the spectrum data.

    Returns
    -------
    pd.DataFrame
        The preprocessed spectrum DataFrame.
    """
    log_transform(spectrum)
    baseline_correction(spectrum)
    normalize(spectrum, 'baseline')
    return spectrum

def preprocess_library(spectrum_lib:dict)->dict:
    """
    Preprocess a spectrum library by applying the 'preprocess' function to each DataFrame in the library.

    Parameters
    ----------
    spectrum_lib : dict
        Dictionary where keys represent mine names and values are DataFrames containing the spectrum data.

    Returns
    -------
    dict
        The preprocessed spectrum library.
    """
    for mine_name, df in spectrum_lib.items():
        preprocess(df)
    return spectrum_lib

def get_matrix(spectrum_lib:dict, test_spectrum:pd.DataFrame, attr='normalized')->tuple:
    """将给定的光谱库和未知光谱转为矩阵输出

    Args:
        spectrum_lib (dict): list of pandas dataframes
        test_spectrum (pd.DataFrame): dataframe of the unknown spectrum
        attr (str, optional): 取出的属性名. Defaults to 'log'.

    Returns:
        2d matrix: 返回两个值，第一个是端元矩阵，第二个是测试光谱数据
    """
    
    # 从字典中取出所有的光谱数据
    normalized_arrays = [df[attr].values for df in spectrum_lib.values()]
    
    # 将所有的光谱数据合并成一个矩阵，每一行代表一个光谱数据
    merged_array = np.vstack(normalized_arrays)
    
    # 将矩阵转置
    transposed_array = merged_array.T
    
    # 取出测试光谱数据
    test_array = test_spectrum[attr].values
    
    return transposed_array, test_array

def unmix_lasso(spectrum_lib:dict,unknown_spectrum:pd.DataFrame,attr='normalized',alpha=0.001,max_mines=2,min_account=0.1)->tuple:
    """
    Perform unmixing using LASSO regression on a spectrum library and an unknown spectrum.

    Parameters:
        spectrum_lib (dict): List of pandas DataFrames representing the spectrum library.
        unknown_spectrum (pd.DataFrame): DataFrame representing the unknown spectrum.
        attr (str, optional): The attribute name to extract. Defaults to 'log'.
        alpha (float, optional): The regularization strength of the LASSO regression. Defaults to 0.0005.
        max_mines (int, optional): The maximum number of mines to consider. Defaults to 2.
        min_account (float, optional): The minimum weight for a mine to be considered. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the endmember coefficients, explained variance, residual sum of squares, and the
        fitted mixture spectrum.

    """
    
    # 获取端元数据和未知光谱数据
    X_known,y_unknown=get_matrix(spectrum_lib, unknown_spectrum,attr)
    X_known = np.array(X_known).T
    
    # 创建LASSO回归模型，并拟合已知光谱端元数据
    lasso = Lasso(alpha=alpha, positive=True)
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
    endmember_coefficients_less = endmember_coefficients_less[:max_mines]
    
    # 归一化权重，使其成为比例 for reporting
    total = sum([coef for mine_name, coef in endmember_coefficients_less])
    endmember_coefficients_less = [(mine_name, coef/total) for mine_name, coef in endmember_coefficients_less]
    
    # 剔除权重小于min_account的端元
    endmember_coefficients_less = [(mine_name, coef) for mine_name, coef in endmember_coefficients_less if coef > min_account]
    
    # 归一化权重，使其成为比例 for reporting
    total = sum([coef for mine_name, coef in endmember_coefficients_less])
    endmember_coefficients_less = [(mine_name, coef/total) for mine_name, coef in endmember_coefficients_less]
    
    # 创建一个只包含已选择端元的矩阵
    selected_endmember_matrix = np.column_stack([spectrum_lib[mine_name][attr].values for mine_name, coef in endmember_coefficients_less])
    selected_endmember_matrix_org = np.column_stack([spectrum_lib[mine_name]['reflectance'].values for mine_name, coef in endmember_coefficients_less])
    
    # 使用这个矩阵和端元系数进行点积运算，得到拟合的混合光谱
    fitted_mixture_spectrum_log = np.dot(selected_endmember_matrix, [coef for mine_name, coef in endmember_coefficients_less])
    fitted_mixture_spectrum_linear = np.dot(selected_endmember_matrix_org, [coef for mine_name, coef in endmember_coefficients_less])
    
    # 计算各种评价指标
    explained_var = explained_variance_score(y_unknown, fitted_mixture_spectrum_log)
    rss = np.sum((y_unknown - fitted_mixture_spectrum_log) ** 2)
    
    # 还原混合光谱
    fitted_mixture_spectrum_log = 10 ** fitted_mixture_spectrum_log
    
    return endmember_coefficients_less, explained_var, rss, fitted_mixture_spectrum_log

def unmix_ridge(spectrum_lib:dict, unknown_spectrum:pd.DataFrame, attr='normalized', alpha=0.2, max_mines=2,min_account=0.1)->tuple:
    """
    Perform unmixing using Ridge regression on a spectrum library and an unknown spectrum.

    Parameters:
        spectrum_lib (dict): Dictionary representing the spectrum library with mine names as keys and pandas DataFrames
                             as values.
        unknown_spectrum (pd.DataFrame): DataFrame representing the unknown spectrum.
        attr (str, optional): The attribute name to extract. Defaults to 'log'.
        alpha (float, optional): The regularization strength of the Ridge regression. Defaults to 0.2.
        max_mines (int, optional): The maximum number of mines to consider. Defaults to 2.
        min_account (float, optional): The minimum weight for a mine to be considered. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the endmember coefficients, explained variance, residual sum of squares, and the
        fitted mixture spectrum.

    """
    
    # 获取端元数据和未知光谱数据
    X_known, y_unknown = get_matrix(spectrum_lib, unknown_spectrum, attr)
    X_known = np.array(X_known).T
    
    # 创建Ridge回归模型，并拟合已知光谱端元数据
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_known.T, y_unknown)
    
    # 获取端元系数（即端元的权重）
    endmember_coefficients = ridge.coef_
    
    # 处理端元系数
    mines = spectrum_lib.keys()
    
    # 提取权重大于0的端元
    endmember_coefficients_less = [(mine_name, coef) for mine_name, coef in zip(mines, endmember_coefficients) if coef > 0]
    
    # 对权重进行排序
    endmember_coefficients_less.sort(key=lambda x: x[1], reverse=True)
    
    # 只保留前max_mines个端元
    endmember_coefficients_less = endmember_coefficients_less[:max_mines]
    
    # 归一化权重，使其成为比例 for reporting
    total = sum([coef for mine_name, coef in endmember_coefficients_less])
    endmember_coefficients_less = [(mine_name, coef/total) for mine_name, coef in endmember_coefficients_less]
    
    # 剔除权重小于min_account的端元
    endmember_coefficients_less = [(mine_name, coef) for mine_name, coef in endmember_coefficients_less if coef > min_account]
    
    # 归一化权重，使其成为比例 for reporting
    total = sum([coef for mine_name, coef in endmember_coefficients_less])
    endmember_coefficients_less = [(mine_name, coef/total) for mine_name, coef in endmember_coefficients_less]
    
    # 创建一个只包含已选择端元的矩阵
    selected_endmember_matrix = np.column_stack([spectrum_lib[mine_name][attr].values for mine_name, coef in endmember_coefficients_less])
    
    # 使用这个矩阵和端元系数进行点积运算，得到拟合的混合光谱
    fitted_mixture_spectrum_log = np.dot(selected_endmember_matrix, [coef for mine_name, coef in endmember_coefficients_less])
    
    # 计算各种评价指标
    explained_var = explained_variance_score(y_unknown, fitted_mixture_spectrum_log)
    rss = np.sum((y_unknown - fitted_mixture_spectrum_log) ** 2)
    
    # 还原混合光谱
    fitted_mixture_spectrum = 10 ** fitted_mixture_spectrum_log
    
    return endmember_coefficients_less, explained_var, rss, fitted_mixture_spectrum

def unmix_linear(spectrum_lib:dict, unknown_spectrum:pd.DataFrame, attr='normalized', max_mines=2, min_account=0.05)->tuple:
    
    """
    使用线性回归对光谱库和未知光谱进行解混。

    参数：
    - spectrum_lib (dict)：表示光谱库的字典，其中矿物名称为键，对应的 pandas DataFrame 为值。
    - unknown_spectrum (pd.DataFrame)：表示未知光谱的 DataFrame。
    - attr (str, optional)：要提取的属性名称，默认为 'log'。
    - max_mines (int, optional)：要考虑的最大矿物数，默认为2。
    - min_account (float, optional)：被考虑的矿物的最小权重，默认为0.05。

    返回值：
    - tuple：包含端元系数（endmember coefficients）、解释方差（explained variance）、残差平方和（residual sum平方）和拟合的混合光谱（fitted mixture spectrum）的元组。
    """
    
    # 获取端元数据和未知光谱数据
    X_known, y_unknown = get_matrix(spectrum_lib, unknown_spectrum, attr)
    X_known = np.array(X_known).T

    # 初始化已选择的端元列表和已选择的端元系数
    selected_endmembers = []
    selected_coefficients = []

    # 基于性能指标选择最佳端元
    while True:
        best_endmember = None
        best_mse = float('inf')
        # 对于每个端元，计算添加该端元后的性能指标
        for endmember_name, endmember_data in spectrum_lib.items():
            if endmember_name not in selected_endmembers:
                # 添加一个端元并拟合线性回归模型
                current_endmembers = selected_endmembers + [endmember_name]
                X_current = np.column_stack([endmember_data[attr].values for endmember_name, endmember_data in spectrum_lib.items() if endmember_name in current_endmembers])
                # 线性拟合
                model = LinearRegression(positive=True).fit(X_current, y_unknown)
                y_pred = model.predict(X_current)
                # 计算均方误差
                mse = mean_squared_error(y_unknown, y_pred)
                # 如果性能更好，则更新最佳端元
                if mse < best_mse:
                    best_mse = mse
                    best_endmember = endmember_name

        # 将最佳端元添加到已选择的列表中
        selected_endmembers.append(best_endmember)

        # 重新拟合线性回归模型并获取系数
        X_selected = np.column_stack([endmember_data[attr].values for endmember_name, endmember_data in spectrum_lib.items() if endmember_name in selected_endmembers])
        model = LinearRegression(positive=True).fit(X_selected, y_unknown)
        selected_coefficients = model.coef_
        
        # 如果某些停止准则满足，则退出循环
        if len(selected_endmembers) > max_mines-1:
            break
    
    # 归一化系数
    total = sum(selected_coefficients)
    selected_coefficients = selected_coefficients/total
    endmember_coefficients = list(zip(selected_endmembers, selected_coefficients))
    endmember_coefficients_less=[(mine_name, coef) for mine_name, coef in endmember_coefficients if coef > min_account]
    
    # 创建一个只包含已选择端元的矩阵
    selected_endmember_matrix = np.column_stack([spectrum_lib[mine_name][attr].values for mine_name, coef in endmember_coefficients_less])
    
    # 使用这个矩阵和端元系数进行点积运算，得到拟合的混合光谱
    fitted_mixture_spectrum_log = np.dot(selected_endmember_matrix, [coef for mine_name, coef in endmember_coefficients_less])
    
    # 计算各种评价指标
    explained_var = explained_variance_score(y_unknown, fitted_mixture_spectrum_log)
    rss=np.sum((y_unknown-fitted_mixture_spectrum_log)**2)
    
    # 还原混合光谱
    fitted_mixture_spectrum = 10 ** fitted_mixture_spectrum_log
    
    return endmember_coefficients_less, explained_var, rss, fitted_mixture_spectrum
