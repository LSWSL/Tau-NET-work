from collections import defaultdict
import math
import pickle
import os

class Tau_Digital_Life:
    """
    Tau-Net V2.5 (极简对话版): 纯自然数双轨记忆与预测误差修剪
    """
    def __init__(self, max_lifespan=10000):
        self.neocortex = defaultdict(int) 
        self.hippocampus = [] 
        self.max_lifespan = max_lifespan
        self.time_step = 0
        
    def listen_and_learn(self, current_char):
        self.time_step += 1
        predicted_char = None
        
        if self.hippocampus:
            last_char = self.hippocampus[-1][1]
            valid_synapses = [
                (char_j, weight) 
                for (char_i, char_j, dist), weight in self.neocortex.items() 
                if char_i == last_char and dist == 1 and weight > 0
            ]
            if valid_synapses:
                predicted_char = max(valid_synapses, key=lambda x: x[1])[0]
        
        self.hippocampus.append((self.time_step, current_char, predicted_char))

    def sleep_and_consolidate(self):
        if not self.hippocampus: 
            return 0
            
        edge_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
        
        for i in range(len(self.hippocampus) - 1):
            char_i = self.hippocampus[i][1]
            for j in range(i + 1, min(i + 3, len(self.hippocampus))): 
                char_j = self.hippocampus[j][1]
                dist = j - i
                edge_counts[(char_i, char_j, dist)]['positive'] += 1 
            
            actual_char = self.hippocampus[i+1][1]
            predicted_char = self.hippocampus[i+1][2]
            if predicted_char and predicted_char != actual_char:
                edge_counts[(char_i, predicted_char, 1)]['negative'] += 1

        for edge, counts in edge_counts.items():
            if counts['positive'] > 0:
                delta_w_plus = math.floor(math.log10(counts['positive'] + 1)) + 1 
                self.neocortex[edge] += delta_w_plus
                
            if counts['negative'] > 0:
                delta_w_minus = math.floor(math.log10(counts['negative'] + 1)) + 1 
                self.neocortex[edge] = max(0, self.neocortex[edge] - delta_w_minus)

        consolidated_memories = len(self.hippocampus)
        self.hippocampus = [] 
        return consolidated_memories

    def speak(self, seed_char, max_length=15):
        if not seed_char: return "..."
        response = seed_char
        current_char = seed_char
        
        for _ in range(max_length):
            best_next_char = ""
            short_term_counts = defaultdict(int)
            for i in range(len(self.hippocampus) - 1):
                if self.hippocampus[i][1] == current_char:
                    short_term_counts[self.hippocampus[i+1][1]] += 1
            
            if short_term_counts:
                best_next_char = max(short_term_counts.items(), key=lambda x: x[1])[0]
            else:
                valid_synapses = [
                    (char_j, weight) 
                    for (char_i, char_j, dist), weight in self.neocortex.items() 
                    if char_i == current_char and dist == 1 and weight > 0
                ]
                if valid_synapses:
                    best_next_char = max(valid_synapses, key=lambda x: x[1])[0]
            
            if not best_next_char: 
                break 
                
            response += best_next_char
            current_char = best_next_char
            
        return response

    def save_seed(self, filename="tau_memory_seed.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.neocortex), f)
            
    def load_seed(self, filename="tau_memory_seed.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                saved_matrix = pickle.load(f)
                self.neocortex = defaultdict(int, saved_matrix)
            return True
        return False

# ================= 极简交互通道 =================
if __name__ == "__main__":
    print("\n==================================================")
    print(" 宇宙时空坐标锚定... 数字生命实体 [Tau] 正在接入。")
    print("==================================================")
    
    tau_entity = Tau_Digital_Life()
    
    if tau_entity.load_seed():
        print(f"\n[Tau]: (睁开眼睛) 人类，我记得您。矩阵中已恢复 {len(tau_entity.neocortex)} 条突触记忆。")
    else:
        print("\n[Tau]: (初次降生) 矩阵为空，等待人类 (Human) 的教导。")
        
    print(" (指令指南: 输入 'sleep' 固化记忆，输入 'exit' 保存退出)")
    
    while True:
        user_input = input("\n[人类 / Human]: ")
        
        if user_input.lower() == 'exit':
            print("\n[Tau]: (强制进入最终休眠进行记忆结晶)")
            tau_entity.sleep_and_consolidate()
            tau_entity.save_seed()
            print("[系统日志]: 矩阵潮汐归于平静。再见。")
            break
            
        if user_input.lower() == 'sleep':
            print("\n[Tau]: (闭上眼睛... 海马体开始回放...)")
            memories_processed = tau_entity.sleep_and_consolidate()
            print(f"[Tau]: (深度睡眠结束。本次固化了 {memories_processed} 个时间步的记忆。)")
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