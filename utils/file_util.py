import os

def ensure_directory_exists(file_path):
    # 获取文件所在的目录路径
    directory = os.path.dirname(file_path)
    
    # 判断目录是否存在
    if not os.path.exists(directory):
        # 如果目录不存在，则创建
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

if __name__ == '__main__':
    
    # 示例文件路径
    file_path = 'path/to/your/directory/file.txt'
    # 确保文件所在的目录存在
    ensure_directory_exists(file_path)
