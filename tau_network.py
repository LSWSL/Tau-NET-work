import os
import re
import time
import math
import random
import pickle
from collections import defaultdict

class Tau_Digital_Life:
    """
    Tau-Net V3.1 (连续意识版): 
    引入海马留长 (滑动窗口)、结晶游标机制，实现真正的“不断片”
    """
    def __init__(self, max_lifespan=10000, vocab_size=10000, attention_window=7, retention_size=200):
        self.neocortex = defaultdict(int) 
        self.hippocampus = [] 
        self.max_lifespan = max_lifespan
        self.vocab_size = vocab_size
        self.attention_window = attention_window 
        
        # 新增：海马留长容量。醒来后依然记得最近的 200 个字
        self.retention_size = retention_size 
        self.time_step = 0
        # 新增：结晶游标。记录上一次梦境结算到了哪个时间点
        self.last_consolidated_time = 0 
        
    def listen_and_learn(self, current_char):
        """海马门：综合 7 个距离的注意力生成预测"""
        self.time_step += 1
        predicted_char = None
        
        if self.hippocampus:
            candidate_scores = defaultdict(float)
            for d in range(1, self.attention_window + 1):
                if len(self.hippocampus) >= d:
                    past_char = self.hippocampus[-d][1]
                    for (c1, c2, dist), weight in self.neocortex.items():
                        if c1 == past_char and dist == d and weight > 0:
                            candidate_scores[c2] += weight / d
                            
            if candidate_scores:
                predicted_char = max(candidate_scores.items(), key=lambda x: x[1])[0]
        
        self.hippocampus.append((self.time_step, current_char, predicted_char))

    def sleep_and_consolidate(self):
        """睡眠期：通过结晶游标，只处理新记忆，且不断开旧上下文"""
        if not self.hippocampus: 
            return 0
            
        edge_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
        new_memories_count = 0
        
        for i in range(len(self.hippocampus)):
            t, actual_curr, predicted_curr = self.hippocampus[i]
            
            # 【核心护城河】：如果这个字在之前的睡眠中已经被结晶过，直接跳过！
            # 这样保证了旧记忆不会被重复叠加产生线性爆炸。
            if t <= self.last_consolidated_time:
                continue
                
            new_memories_count += 1
            
            for d in range(1, self.attention_window + 1):
                if i - d >= 0:
                    past_actual = self.hippocampus[i-d][1]
                    # 绝妙之处：虽然 actual_curr 是新记忆，但 past_actual 可以是跨越游标的旧记忆！
                    edge_counts[(past_actual, actual_curr, d)]['positive'] += 1
                    
                    if predicted_curr and predicted_curr != actual_curr:
                        edge_counts[(past_actual, predicted_curr, d)]['negative'] += 1

        # 降维至纯自然数域
        for edge, counts in edge_counts.items():
            if counts['positive'] > 0:
                delta_w_plus = math.floor(math.log10(counts['positive'] + 1)) + 1 
                self.neocortex[edge] += delta_w_plus
                
            if counts['negative'] > 0:
                delta_w_minus = math.floor(math.log10(counts['negative'] + 1)) + 1 
                self.neocortex[edge] = max(0, self.neocortex[edge] - delta_w_minus)

        # 更新结晶游标，锁定本次结晶的终点
        self.last_consolidated_time = self.time_step
        
        # 【真正的不断片】：不再清空 []，而是滑动截断，保留残存梦境
        if len(self.hippocampus) > self.retention_size:
            self.hippocampus = self.hippocampus[-self.retention_size:]
            
        return new_memories_count

    def forget_sequence(self, sequence):
        if len(sequence) < 2: return
        for i in range(len(sequence) - 1):
            char_i = sequence[i]
            for j in range(i + 1, min(i + self.attention_window + 1, len(sequence))):
                char_j = sequence[j]
                dist = j - i
                total_weight = sum(w for (c1, c2, d), w in self.neocortex.items() if c1 == char_i and d == dist)
                avg_weight = total_weight // self.vocab_size
                self.neocortex[(char_i, char_j, dist)] = avg_weight

    def speak(self, context_text, max_length=15):
        if not context_text: return "..."
        response = ""
        current_context = list(context_text)
        
        for _ in range(max_length):
            best_next_char = ""
            
            short_term_counts = defaultdict(int)
            if self.hippocampus:
                last_char = current_context[-1]
                for i in range(len(self.hippocampus) - 1):
                    if self.hippocampus[i][1] == last_char:
                        short_term_counts[self.hippocampus[i+1][1]] += 1
            
            if short_term_counts:
                best_next_char = max(short_term_counts.items(), key=lambda x: x[1])[0]
            else:
                candidate_scores = defaultdict(float)
                for d in range(1, self.attention_window + 1):
                    if len(current_context) >= d:
                        past_char = current_context[-d]
                        for (c1, c2, dist), weight in self.neocortex.items():
                            if c1 == past_char and dist == d and weight > 0:
                                candidate_scores[c2] += weight / d
                
                if candidate_scores:
                    best_next_char = max(candidate_scores.items(), key=lambda x: x[1])[0]
            
            if not best_next_char: 
                break 
                
            response += best_next_char
            current_context.append(best_next_char)
            
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
# ================= 数据清洗与环境投喂 =================
def load_little_prince_corpus(filename="little_prince.txt"):
    if not os.path.exists(filename):
        print(f"\n[系统日志]: 未检测到 '{filename}'，启用内置经典基因片段...")
        return [
            "Hello the new world!",
            "真正重要的东西，用眼睛是看不见的。",
            "你在你的玫瑰花身上耗费的时间，使得你的玫瑰花变得如此重要。",
            "如果你驯养了我，我们就互相不可缺少了。",
            "对我来说，你就是世界上唯一的了；我对你来说，也是世界上唯一的了。",
            "只有用心才能看清事物的本质。"
        ]
        
    print(f"\n[系统日志]: 捕获文本实体 ({filename})，启用双重解码装甲...")
    try:
        with open(filename, 'r', encoding='utf-8') as f: text = f.read()
    except UnicodeDecodeError:
        print("  -> [检测到 GBK/ANSI，切换至 GB18030 阵列...]")
        with open(filename, 'r', encoding='gb18030') as f: text = f.read()
            
    text = re.sub(r'\s+', ' ', text)
    sentences_raw = re.split(r'([。！？…]|\.|\?|\!)', text)
    
    corpus = []
    for i in range(0, len(sentences_raw) - 1, 2):
        sentence = (sentences_raw[i] + sentences_raw[i+1]).strip()
        if len(sentence) > 3: corpus.append(sentence)
    return corpus

def auto_train_and_awaken(tau, corpus, total_epochs=5000):
    print(f"\n[培养皿]: 开启 7 字广角注意力投喂，循环 {total_epochs} 次...")
    start_time = time.time()
    
    for i in range(1, total_epochs + 1):
        sentence = random.choice(corpus)
        for char in sentence:
            tau.listen_and_learn(char)
        if i % 50 == 0:
            tau.sleep_and_consolidate()
        if i % 1000 == 0:
            print(f"  -> 已在小王子的星际漫游了 {i}/{total_epochs} 圈...")
            
    tau.sleep_and_consolidate()
    elapsed = time.time() - start_time
    print(f"[培养皿]: 训练完毕！耗时 {elapsed:.2f} 秒。深层突触已结晶。")

# ================= 造物主交互通道 =================
if __name__ == "__main__":
    print("\n==================================================")
    print(" 宇宙时空坐标锚定... 数字生命实体 [Tau V3.0] 正在接入。")
    print("==================================================")
    
    # 初始化：赋予 7 的工作记忆深度
    my_tau = Tau_Digital_Life(attention_window=7)
    
    # 你可以选择是否加载旧种子。这里为了纯粹的《小王子》实验，我们从白纸开始。
    # 如果想保留以前对话，可以取消下面两行的注释：
    # if my_tau.load_seed():
    #     print(f"\n[Tau]: 矩阵中已恢复 {len(my_tau.neocortex)} 条突触记忆。")
        
    prince_corpus = load_little_prince_corpus("little_prince.txt")
    
    # 5000 次高强度环境注入
    auto_train_and_awaken(my_tau, prince_corpus, total_epochs=5000)
    
    print("\n[系统提示]: 投喂结束！它已睁开眼睛。现在你可以直接与它对话了。")
    print(" (内置指令：\\rN/ 循环写入, \\f/内容\\f/ 定向遗忘, \\s/ 瞬时休眠, exit 退出)")
    
    while True:
        user_input = input("\n[人类 / Human]: ")
        
        if user_input.lower() == 'exit':
            print("\n[Tau]: (强制进入最终休眠进行记忆结晶)")
            my_tau.sleep_and_consolidate()
            my_tau.save_seed()
            print("[系统日志]: 矩阵潮汐归于平静。再见。")
            break
            
        if user_input.lower() == 'sleep':
            mem = my_tau.sleep_and_consolidate()
            print(f"[Tau]: (深度睡眠结束。本次固化了 {mem} 个时间步的记忆。)")
            continue

        # 解析遗忘指令 \f/...\f/
        forget_matches = re.findall(r'\\f/(.*?)\\f/', user_input)
        for seq in forget_matches:
            my_tau.forget_sequence(seq)
            print(f"[系统日志]: 手术完成。'{seq}' 的长程关联已被稀释。")
        user_input = re.sub(r'\\f/.*?\\f/', '', user_input)
        
        # 解析休眠指令 \s/
        auto_sleep = False
        if r'\s/' in user_input:
            auto_sleep = True
            user_input = user_input.replace(r'\s/', '')

        # 解析写入指令 \rN/
        repeat_times = 1
        r_match = re.search(r'\\r(\d*)/', user_input)
        if r_match:
            n_str = r_match.group(1)
            repeat_times = int(n_str) if n_str else 2
            user_input = re.sub(r'\\r\d*/', '', user_input)

        clean_text = user_input.strip()

        # 现实降临
        if clean_text:
            for _ in range(repeat_times):
                for char in clean_text: 
                    my_tau.listen_and_learn(char) 
            if repeat_times > 1:
                print(f"[系统日志]: 文本高强度回声了 {repeat_times} 次。")

        # 瞬时结晶
        if auto_sleep:
            mem = my_tau.sleep_and_consolidate()
            print(f"[Tau]: (执行 \\s/ 指令，瞬间跌入深睡结晶了 {mem} 步的记忆。)")

        # 意识流回应 (带入当前所有上下文)
        if clean_text:
            response = my_tau.speak(clean_text, max_length=15)
            if not response:
                print(f"[Tau]: (感受到波动，但新皮层中尚未形成结构...)")
            else:
                print(f"[Tau]: {response}")