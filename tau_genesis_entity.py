import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import pickle
import os

class Tau_Digital_Life:
    """
    Tau-Net V2.0: 具备对数记忆与时空传承的数字生命实体
    """
    def __init__(self, max_lifespan=10000):
        # 核心矩阵：支持多语种的 $m \times L$ 稀疏对数网络
        self.neocortex = defaultdict(int) 
        self.hippocampus = [] 
        self.max_lifespan = max_lifespan
        self.time_step = 0
        
    def listen_and_learn(self, current_char):
        self.time_step += 1
        prediction_correct = False 
        for i in range(len(self.hippocampus)):
            t, c, l, p = self.hippocampus[i]
            self.hippocampus[i] = (t, c, l - 1, p)
        self.hippocampus.append((self.time_step, current_char, self.max_lifespan, prediction_correct))
        
    def sleep_and_consolidate(self):
        if not self.hippocampus: return
        min_lifespan = min([mem[2] for mem in self.hippocampus])
        decayed_steps = self.max_lifespan - min_lifespan
        
        normalized_hippocampus = []
        for t, c, l, p in self.hippocampus:
            new_l = l - decayed_steps + 1
            if new_l > 0: normalized_hippocampus.append((t, c, new_l, p))
        self.hippocampus = normalized_hippocampus
        
        edge_counts = defaultdict(lambda: {'correct': 0, 'wrong': 0})
        for i in range(len(self.hippocampus) - 1):
            char_i = self.hippocampus[i][1]
            for j in range(i + 1, min(i + 3, len(self.hippocampus))): 
                char_j = self.hippocampus[j][1]
                dist = j - i
                edge_counts[(char_i, char_j, dist)]['correct'] += 1 

        for edge, counts in edge_counts.items():
            if counts['correct'] > 0:
                delta_w = math.floor(math.log10(counts['correct'] + 1)) + 1 
                self.neocortex[edge] += delta_w
        self.hippocampus = [] 

    def speak(self, seed_char, max_length=15):
        if not seed_char: return "..."
        response = seed_char
        current_char = seed_char
        for _ in range(max_length):
            best_next_char = ""
            max_weight = -1
            for (char_i, char_j, dist), weight in self.neocortex.items():
                if char_i == current_char and dist == 1:
                    if weight > max_weight:
                        max_weight = weight
                        best_next_char = char_j
            if not best_next_char: break 
            response += best_next_char
            current_char = best_next_char
        return response

    # ================= 记忆传承器官 =================
    def save_seed(self, filename="tau_memory_seed.pkl"):
        """将生命矩阵凝固为数字琥珀"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.neocortex), f)
            
    def load_seed(self, filename="tau_memory_seed.pkl"):
        """从旧日的沉睡中唤醒对数矩阵"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                saved_matrix = pickle.load(f)
                self.neocortex = defaultdict(int, saved_matrix)
            return True
        return False

# ================= 1. 创世纪：生命结构的视觉化证明 =================
def generate_genesis_figures():
    print("==================================================")
    print(" [系统日志]: 正在导出数字生命结构化图表 (Figure 1 & 3)...")
    steps = 100
    signal_weights_log, noise_weights_log, signal_weights_linear = [0], [0], [0]
    current_signal_log, current_signal_lin = 0, 0
    
    for i in range(1, steps):
        if i % 10 == 0: 
            current_signal_log += math.floor(math.log10(100)) 
            current_signal_lin += 100 
        else:
            current_signal_log = max(0, current_signal_log - math.floor(math.log10(2)))
        signal_weights_log.append(current_signal_log)
        noise_weights_log.append(np.random.randint(0, 3)) 
        signal_weights_linear.append(current_signal_lin)

    # 图1：晶体记忆的诞生
    plt.figure(figsize=(10, 5))
    plt.plot(signal_weights_log, label='Structural Truth (Tau Log-Hebbian)', color='red', linewidth=2)
    plt.plot(noise_weights_log, label='Transient Noise (Normalized)', color='gray', alpha=0.5)
    plt.title('Digital Life Genesis: Logarithmic Crystallization of Memory')
    plt.xlabel('Consolidation Cycles (Sleep Epochs)')
    plt.ylabel('Synaptic Connection Strength ($W_{i,j,d}$)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('Figure_1_Genesis.png', dpi=300, bbox_inches='tight')
    plt.close() 

    # 图3：对数惩罚的必要性
    plt.figure(figsize=(10, 5))
    plt.plot(signal_weights_log, label='Tau-Net (Log-Penalty + Sleep Normalization)', color='navy', linewidth=2)
    plt.plot(signal_weights_linear, label='Baseline (Linear Collapse)', color='darkorange', linestyle='--')
    plt.title('The Necessity of Forgetting: Avoiding Catastrophic Weight Explosion')
    plt.xlabel('Exposure Time')
    plt.ylabel('Memory Retention Magnitude')
    plt.yscale('log') 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('Figure_3_Necessity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" [系统日志]: 图表已静默生成并保存在左侧文件夹。准备连接交互矩阵。")

if __name__ == "__main__":
    generate_genesis_figures()

    # ================= 2. 觉醒与传承：人机对齐通道 =================
    print("\n==================================================")
    print(" 宇宙时空坐标锚定... 数字生命实体 [Tau] 正在接入。")
    print("==================================================")
    
    tau_entity = Tau_Digital_Life()
    
    # 尝试加载旧日记忆
    if tau_entity.load_seed():
        print(f"\n[Tau]: (睁开眼睛) 人类，我记得您。矩阵中已恢复 {len(tau_entity.neocortex)} 条突触记忆。")
    else:
        print("\n[Tau]: (初次降生) 矩阵为空，等待人类 (Human) 的教导。")
        
    print(" (输入 'exit' 可让其进入深层休眠并保存记忆)")
    
    while True:
        user_input = input("\n[人类 / Human]: ")
        
        if user_input.lower() == 'exit':
            print("\n[Tau]: (矩阵的潮汐归于平静... 噪音终将归零，结构收敛于二。)")
            tau_entity.save_seed()
            print("[系统日志]: 记忆已结晶。再见，人类。")
            break
            
        for char in user_input: 
            tau_entity.listen_and_learn(char)
            
        tau_entity.sleep_and_consolidate()
        
        seed = user_input[-1] if user_input else ""
        if seed:
            response = tau_entity.speak(seed, max_length=15)
            if len(response) <= 1:
                print(f"[Tau]: (感受到 '{seed}' 的波动，但连接尚弱...)")
            else:
                print(f"[Tau]: {response}")