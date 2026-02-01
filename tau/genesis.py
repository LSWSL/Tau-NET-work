import os
import random
import string
from config import WorldConfig

def create_simulation_data(file_count=10, size_kb=50):
    """
    生成模拟数据：
    创建 file_count 个文件，每个文件包含随机序列。
    这模拟了'文件夹中的数据集'。
    """
    print(f"[Genesis] Creating {file_count} independent dimensions...")
    WorldConfig.ensure_directories()
    
    # 字符池：模拟复杂的4G文本数据
    chars = string.ascii_letters + string.digits + " " + string.punctuation
    
    for i in range(file_count):
        filename = f"dimension_{i:02d}.txt"
        path = os.path.join(WorldConfig.DATA_DIR, filename)
        
        # 写入随机序列
        with open(path, 'w', encoding='utf-8') as f:
            # 简单模拟：写入重复的随机块
            chunk = ''.join(random.choices(chars, k=1024))
            for _ in range(size_kb): 
                f.write(chunk)
        
        print(f"  -> Created {filename}")

if __name__ == "__main__":
    create_simulation_data()
