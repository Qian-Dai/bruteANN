import tarfile
import os

# 设置.tar.gz文件的路径和解压后的目录路径
tar_gz_path = '../gist.tar.gz'
extract_dir = ''

# 使用tarfile模块打开.tar.gz文件
with tarfile.open(tar_gz_path, "r:gz") as tar:
    # 解压所有文件到指定目录
    tar.extractall(path=extract_dir)

print(f"Files extracted to {extract_dir}")
