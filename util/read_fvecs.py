#读取处理fvecs文件，将其变成向量

import numpy as np
def read_Fvecs(file_path):
    """
        读取 fvecs 文件中的向量。
        :param file_path: fvecs 文件路径
        :return: numpy 数组，每一行是一个向量
        """
    with open(file_path, 'rb') as f:
        data = f.read()

    vectors = []
    offset = 4
    dims=960# 读取向量的维度
    while offset < len(data):
        # 读取向量数据
        vector = np.frombuffer(data, dtype=np.float32, count=dims, offset=offset)
        vectors.append(vector)
        offset += dims * 4+4# 跳过当前向量数据

    return np.vstack(vectors)