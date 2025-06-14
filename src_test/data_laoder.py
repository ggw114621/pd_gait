import numpy as np

# 加载数据
X_data = np.load("Xdata.npy")  # 所有患者的数据段，形状为 (n_samples, n_features, n_timesteps)
# Xdata.npy  ----> (65915, 100, 18)
y_data = np.load("ydata.npy")  # 所有患者的标签，形状为 (n_samples,)    -----> (65915, 1)
data_person = np.load("data_person.npy")  # 每个患者的数据段数量，形状为 (n_patients + 1,)    ---> (307, )
 

# 打印部分数据内容
print("X_data的前5个数据段:")
print(X_data[:5])  # 打印X_data的前5个数据段
print("y_data的前5个标签:")
print(y_data[65910:])  # 打印y_data的前5个标签
print("data_person:")
print(data_person)  # 打印每个患者的数据段数量
def separate_fold(X_data, y_data, data_person, fold_number, total_fold=5):
    """
    按“患者”级别划分训练集和验证集
    :param X_data: 所有患者的数据段，形状为 (n_samples, n_features, n_timesteps)
    :param y_data: 所有患者的标签，形状为 (n_samples,)
    :param data_person: 每个患者的数据段数量，形状为 (n_patients + 1,)
    :param fold_number: 当前折数（从0开始）
    :param total_fold: 总折数（默认10折）
    :return: X_train, y_train, X_val, y_val
    """
    n_patients = len(data_person) - 1  # 患者数量
    proportion = 1 / total_fold  # 每折的患者数量
    n_patients_per_fold = int(n_patients * proportion)  # 每折的患者数量
    start_patient = fold_number * n_patients_per_fold  # 当前折的起始患者索引
    end_patient = (fold_number + 1) * n_patients_per_fold  # 当前折的结束患者索引

    # 根据患者索引，提取当前折的验证集
    id_start = data_person[start_patient]  # 当前折的起始数据段索引
    id_end = data_person[end_patient]  # 当前折的结束数据段索引
    X_val = X_data[id_start:id_end]  # 当前折的验证集
    y_val = y_data[id_start:id_end]  # 当前折的验证集标签

    # 提取训练集（即剩余数据）
    X_train = np.delete(X_data, np.arange(id_start, id_end), axis=0)  # 删除当前折的数据段
    y_train = np.delete(y_data, np.arange(id_start, id_end), axis=0)  # 删除当前折的标签

    return X_train, y_train, X_val, y_val


