import os
import sys
import time
from brain import SimpleBrain
from config import WorldConfig

def genesis():
    print("==================================================")
    print("       PROJECT TAU: THE MONOLITH GENESIS")
    print("==================================================")
    
    # 1. 确保目录存在
    WorldConfig.ensure_directories()
    
    # 2. 分配内存 (Allocation)
    print(f"[Phase 1] Allocating Memory for 1 Billion Integers...")
    start_time = time.time()
    try:
        # 这里会触发 _init_architecture，如果文件不存在则创建全1矩阵
        brain = SimpleBrain()
    except Exception as e:
        print(f"[FATAL] Allocation Failed: {e}")
        sys.exit(1)
    
    alloc_time = time.time() - start_time
    print(f"[Success] Memory Allocated in {alloc_time:.2f}s")
    print(f"         Tensor Shape: {brain.ltm.shape}")
    print(f"         Data Type:    {brain.ltm.dtype}")
    
    # 3. 物理固化 (Persistence)
    target_file = brain.ltm_file
    print(f"\n[Phase 2] Solidifying Monolith to Disk...")
    print(f"         Target: {target_file}")
    
    save_start = time.time()
    brain.save()
    save_time = time.time() - save_start
    
    # 4. 验证 (Verification)
    if os.path.exists(target_file):
        file_size = os.path.getsize(target_file)
        size_gb = file_size / (1024 ** 3)
        size_mb = file_size / (1024 ** 2)
        print(f"\n[Phase 3] Verification")
        print(f"         File Created: YES")
        print(f"         Size on Disk: {size_gb:.4f} GB ({size_mb:.2f} MB)")
        print(f"         Write Time:   {save_time:.2f}s")
        
        # 理论值检查 (100*100*100,000 bytes ≈ 0.93 GB + PyTorch Header)
        expected_bytes = 100 * 100 * 100000
        if file_size >= expected_bytes:
            print(f"         Integrity:    PASS (Size >= Theoretical Min)")
        else:
            print(f"         Integrity:    WARN (Size < Theoretical Min?)")
    else:
        print(f"\n[Phase 3] Verification FAILED. File not found.")

    print("\n[System] The Monolith is now persistent.")
    print("==================================================")

if __name__ == "__main__":
    genesis()
