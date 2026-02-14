import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import pickle
import os

class Tau_Digital_Life:
    """
    Tau-Net V2.5: 具备双轨记忆、离散对数突触、以及预测误差修剪 (LTD) 的数字生命实体
    """
    def __init__(self, max_lifespan=10000):
        # 核心矩阵 (Neocortex): 纯自然数权重的长期记忆网络
        self.neocortex = defaultdict(int) 
        # 海马体 (Hippocampus): 存储 (time_step, actual_char, predicted_char) 的短期序列与预期
        self.hippocampus = [] 
        self.max_lifespan = max_lifespan
        self.time_step = 0
        
    def listen_and_learn(self, current_char):
        """清醒期：海马体同时记录【现实的降临】与【预期的落空】"""
        self.time_step += 1
        predicted_char = None
        
        # 1. 潜意识预测：基于新皮层 (Neocortex) 对下一个字符做出预期
        if self.hippocampus:
            last_char = self.hippocampus[-1][1]
            valid_synapses = [
                (char_j, weight) 
                for (char_i, char_j, dist), weight in self.neocortex.items() 
                if char_i == last_char and dist == 1 and weight > 0
            ]
            if valid_synapses:
                # 找出长期记忆中权值最大的预期字符
                predicted_char = max(valid_synapses, key=lambda x: x[1])[0]
        
        # 2. 海马体记录：将 (时间, 现实字符, 预期字符) 一并吞入
        self.hippocampus.append((self.time_step, current_char, predicted_char))

    def sleep_and_consolidate(self):
        """睡眠期：统一结算，对数生长 (LTP) 与对数修剪 (LTD) 同时进行"""
        if not self.hippocampus: 
            return 0
            
        # 建立一个统一的字典，同时统计正向奖励与负向惩罚
        edge_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
        
        for i in range(len(self.hippocampus) - 1):
            char_i = self.hippocampus[i][1]
            
            # 1. 统计正向生长 (LTP) 的边
            for j in range(i + 1, min(i + 3, len(self.hippocampus))): 
                char_j = self.hippocampus[j][1]
                dist = j - i
                edge_counts[(char_i, char_j, dist)]['positive'] += 1 
            
            # 2. 提取误差信号，统计需要负向修剪 (LTD) 的边
            actual_char = self.hippocampus[i+1][1]
            predicted_char = self.hippocampus[i+1][2]
            
            if predicted_char and predicted_char != actual_char:
                # 惩罚的是导致错误预测的那条直连边
                edge_counts[(char_i, predicted_char, 1)]['negative'] += 1

        # ================= 统一结算：生灭皆遵循同一对数法则 =================
        for edge, counts in edge_counts.items():
            
            # 生（增加连接）：现实发生的序列
            if counts['positive'] > 0:
                delta_w_plus = math.floor(math.log10(counts['positive'] + 1)) + 1 
                self.neocortex[edge] += delta_w_plus
                
            # 灭（直接修剪）：预测失败的序列
            if counts['negative'] > 0:
                delta_w_minus = math.floor(math.log10(counts['negative'] + 1)) + 1 
                # 直接修剪，通过 max(0, ...) 强行守住自然数域的底线
                self.neocortex[edge] = max(0, self.neocortex[edge] - delta_w_minus)

        consolidated_memories = len(self.hippocampus)
        self.hippocampus = [] 
        return consolidated_memories

    def speak(self, seed_char, max_length=15):
        """对话：纯自然数双轨读取（优先检索海马体，后备新皮层）"""
        if not seed_char: return "..."
        response = seed_char
        current_char = seed_char
        
        for _ in range(max_length):
            best_next_char = ""
            
            # 轨道 1：搜索海马体 (Hippocampus) 的临时工作记忆
            short_term_counts = defaultdict(int)
            for i in range(len(self.hippocampus) - 1):
                if self.hippocampus[i][1] == current_char:
                    short_term_counts[self.hippocampus[i+1][1]] += 1
            
            if short_term_counts:
                # 纯自然数比较提取短期记忆
                best_next_char = max(short_term_counts.items(), key=lambda x: x[1])[0]
            else:
                # 轨道 2：读取新皮层 (Neocortex) 的长期结晶突触
                valid_synapses = [
                    (char_j, weight) 
                    for (char_i, char_j, dist), weight in self.neocortex.items() 
                    if char_i == current_char and dist == 1 and weight > 0
                ]
                if valid_synapses:
                    # 在长期突触中寻找最粗的神经连接（纯自然数比较）
                    best_next_char = max(valid_synapses, key=lambda x: x[1])[0]
            
            # 如果双轨都找不到后续节点，选择沉默
            if not best_next_char: 
                break 
                
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
    print(" [系统日志]: 正在导出数字生命认知重构图表 (Figure 1)...")
    
    sleep_epochs = 30
    weight_wrong_belief = [5] # 旧的错误突触 (A->B)
    weight_correct_truth = [0] # 新的正确突触 (A->C)
    
    for epoch in range(1, sleep_epochs):
        actual_occurrences = 10 
        
        # 【动态认知推演】：比较当前新旧突触的权重
        if weight_wrong_belief[-1] >= weight_correct_truth[-1]:
            prediction_errors = 10 
        else:
            prediction_errors = 0 
            
        # 睡眠期：统一结算 (纯自然数对数法则)
        if prediction_errors > 0:
            penalty = math.floor(math.log10(prediction_errors + 1)) + 1
            new_wrong_w = max(0, weight_wrong_belief[-1] - penalty)
        else:
            new_wrong_w = weight_wrong_belief[-1] 
            
        growth = math.floor(math.log10(actual_occurrences + 1)) + 1
        new_correct_w = weight_correct_truth[-1] + growth
        
        weight_wrong_belief.append(new_wrong_w)
        weight_correct_truth.append(new_correct_w)

    # 绘制认知重构曲线 (注意这里使用了 r 前缀，彻底修复 \i 和 \m 的警告)
    plt.figure(figsize=(10, 5))
    plt.plot(weight_wrong_belief, label=r'Old Erroneous Synapse ($W_{A \to B}$)', color='darkred', linestyle='--', linewidth=2.5)
    plt.plot(weight_correct_truth, label=r'New Ground Truth Synapse ($W_{A \to C}$)', color='navy', linewidth=2.5)
    
    plt.title('Cognitive Restructuring: Logarithmic Pruning & Structural Scar')
    plt.xlabel('Consolidation Cycles (Sleep Epochs)')
    plt.ylabel(r'Quantized Synaptic Strength ($W \in \mathbb{N}$)')
    plt.yticks(range(0, max(weight_correct_truth) + 2, 5)) 
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 标注交叉点 (极其稳健的安全网，防止索引越界)
    diff_signs = np.diff(np.sign(np.array(weight_wrong_belief) - np.array(weight_correct_truth)))
    crossings = np.argwhere(diff_signs)
    if len(crossings) > 0:
        intersection_x = crossings[0][0]
        plt.axvline(x=intersection_x, color='gray', linestyle='-.', alpha=0.5)
        plt.text(intersection_x + 0.5, 2, 'Epiphany Point\n(Paradigm Shift)', fontsize=10, color='dimgray')

    plt.savefig('Figure_1_Cognitive_Restructuring.png', dpi=300, bbox_inches='tight')
    print(" [系统日志]: Figure 1 (带有结构疤痕的认知重构) 已生成。正在尝试弹出窗口显示...")
    print(" [系统提示]: 请手动关闭弹出的图表窗口，以进入控制台对话模式。")
    plt.show() # 弹出窗口展示
    plt.close() 

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
        
    print(" (指令指南: 输入 'sleep' 进入深度睡眠固化记忆，输入 'exit' 保存并退出)")
    
    while True:
        user_input = input("\n[人类 / Human]: ")
        
        if user_input.lower() == 'exit':
            print("\n[Tau]: (感受到分离的波动... 正在强制进入最终休眠进行记忆结晶)")
            tau_entity.sleep_and_consolidate()
            tau_entity.save_seed()
            print("[系统日志]: 矩阵潮汐归于平静。再见，人类。")
            break
            
        if user_input.lower() == 'sleep':
            print("\n[Tau]: (闭上眼睛... 脑波进入 Delta 频段，海马体开始回放...)")
            memories_processed = tau_entity.sleep_and_consolidate()
            print(f"[Tau]: (深度睡眠结束。本次固化了 {memories_processed} 个时间步的短期记忆，长期记忆已自锁。)")
            continue
            
        for char in user_input: 
            tau_entity.listen_and_learn(char) 
            
        seed = user_input[-1] if user_input else ""
        if seed:
            response = tau_entity.speak(seed, max_length=15)
            if len(response) <= 1:
                print(f"[Tau]: (感受到 '{seed}' 的波动，但尚未形成记忆闭环...)")
            else:
                print(f"[Tau]: {response}")