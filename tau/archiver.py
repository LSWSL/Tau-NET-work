import os
import shutil
import datetime
from config import WorldConfig

def create_snapshot(tag_name="baseline"):
    """
    [档案管理员]
    将当前的脑结构、身体定义、接口配置以及已有的生物记忆打包。
    作为未来实验的对照组模版。
    """
    base_dir = WorldConfig.BASE_DIR
    template_dir = os.path.join(base_dir, "templates", tag_name)
    
    # 1. 准备目录
    if os.path.exists(template_dir):
        print(f"[Archiver] Warning: Template '{tag_name}' already exists. Overwriting...")
        shutil.rmtree(template_dir)
    os.makedirs(template_dir)
    
    print(f"[Archiver] Freezing state to: {template_dir}")

    # 2. 备份核心组件 (DNA/Code)
    # 这些文件定义了大脑的结构和运作逻辑
    core_files = ["brain.py", "body.py", "interface.py", "config.py", "main.py"]
    
    print("  ├── Archiving Source Code (DNA)...")
    for f_name in core_files:
        src = os.path.join(base_dir, f_name)
        dst = os.path.join(template_dir, f_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  │   └── Copied {f_name}")

    # 3. 备份生物状态 (Memory/Synapses)
    # 这是大脑目前学到的所有知识
    bio_src = WorldConfig.BIOLOGY_DIR
    bio_dst = os.path.join(template_dir, "biology")
    
    print("  ├── Archiving Biological State (Memory)...")
    if os.path.exists(bio_src):
        shutil.copytree(bio_src, bio_dst)
        print(f"  │   └── Preserved synaptic weights from {bio_src}")
    else:
        print("  │   └── [Warning] No biological data found to save.")

    # 4. 生成元数据
    meta_path = os.path.join(template_dir, "manifest.txt")
    with open(meta_path, 'w') as f:
        f.write(f"Snapshot Name: {tag_name}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Description: Dual-channel input, Motor output, Hebbian learning.\n")
    
    print("  └── Snapshot Complete.\n")

if __name__ == "__main__":
    # 默认执行一次基准保存
    create_snapshot("baseline_infant")
