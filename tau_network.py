import os
import re
import time
import math
import random
import pickle
from collections import defaultdict, deque

# ================= 可调节超参数 =================
MAX_DISTANCE = 10000           # 皮层最大距离（L）
ATTENTION_WINDOW = 7           # 空间拓扑距离，预测窗口
VOCAB_SIZE = 10000             
REPEAT_PENALTY = 0.5           
PUNISHMENT_MULTIPLIER = 2      # [新增] 错误预测的惩罚放大系数（非对称生物学惊奇反射）
# ================================================

class Tau_Digital_Life:
    """
    Tau-Net V4.4 (非对称惩罚与时空对齐版)
    - 引入 PUNISHMENT_MULTIPLIER 剧烈惩罚错误预测的“幻觉”
    - 绝对时间戳双端队列 O(1) 剔除记忆，拓扑空间距离 O(1) 生成语言
    """
    def __init__(self):
        self.neocortex = defaultdict(lambda: 1)            
        self.forward_index = defaultdict(set)              
        self.hippocampus = deque() # 存储: (char, pred_char, absolute_time)                             
        self.last_char = None                               
        self.max_distance = MAX_DISTANCE                     
        self.attention_window = ATTENTION_WINDOW             
        self.time_step = 0

    def listen_and_learn(self, current_char):
        self.time_step += 1
        predicted_char = None

        # [O(1) 极速预测]
        if self.last_char is not None:
            candidate_scores = defaultdict(float)
            for c2, dist in self.forward_index.get(self.last_char, set()):
                if dist <= self.attention_window:
                    weight = self.neocortex.get((self.last_char, c2, dist), 1)
                    if weight > 0:
                        candidate_scores[c2] += weight / dist
            if candidate_scores:
                predicted_char = max(candidate_scores.items(), key=lambda x: x[1])[0]

        # [时空对齐] 海马体按顺序记录当前字符、对未来的预测和绝对时间
        self.hippocampus.append((current_char, predicted_char, self.time_step))

        # [O(1) 记忆剔除] 寿命耗尽则从队首弹出
        while self.hippocampus and (self.time_step - self.hippocampus[0][2] >= self.max_distance):
            self.hippocampus.popleft()

        self.last_char = current_char

    def sleep_and_consolidate(self):
        if len(self.hippocampus) < 2:
            return 0

        pos_counts = defaultdict(int)   
        neg_counts = defaultdict(int)   

        memories = list(self.hippocampus)
        
        # [真正的空间拓扑距离] 
        for i in range(len(memories) - 1):
            char_i, pred_i, _ = memories[i]
            for j in range(i + 1, min(i + self.attention_window + 1, len(memories))):
                char_j, _, _ = memories[j]
                dist = j - i # 提取纯粹的空间距离
                pos_counts[(char_i, char_j, dist)] += 1
                
                # 错误预测的负强化记录（仅对相邻字符）
                if dist == 1 and pred_i is not None and pred_i != char_j:
                    neg_counts[(char_i, pred_i, 1)] += 1

        write_count = 0

        # 正强化：缓慢而温和的生长
        for (c1, c2, dist), count in pos_counts.items():
            delta = int(math.log10(count))  
            if delta > 0:
                self.neocortex[(c1, c2, dist)] += delta
                self.forward_index[c1].add((c2, dist)) 
                write_count += 1

        # 负强化：剧烈而无情的修剪（引入惩罚乘数）
        for (c1, pred, dist), count in neg_counts.items():
            delta = int(math.log10(count)) * PUNISHMENT_MULTIPLIER 
            if delta > 0:
                self.neocortex[(c1, pred, dist)] = max(1, self.neocortex.get((c1, pred, dist), 1) - delta) 
                self.forward_index[c1].add((pred, dist)) 
                write_count += 1

        self.hippocampus.clear()
        self.save_seed()
        return write_count

    def speak(self, context_text, max_length=15, distance_weight=None):
        if distance_weight is None:
            distance_weight = lambda d: 1.0 / d

        if not context_text:
            return "..."
        response = ""
        current_context = list(context_text)

        for _ in range(max_length):
            window = current_context[-self.attention_window:]
            scores = defaultdict(float)

            for c1 in window:
                for nc, d in self.forward_index.get(c1, set()):
                    w = self.neocortex.get((c1, nc, d), 1)
                    if w > 1: # 只有生长过的突触才参与发声
                        scores[nc] += w * distance_weight(d)

            if not scores:
                break

            if response and REPEAT_PENALTY < 1.0:
                last_char = response[-1]
                if last_char in scores:
                    scores[last_char] *= REPEAT_PENALTY

            best_next_char = max(scores.items(), key=lambda x: x[1])[0]
            if not isinstance(best_next_char, str):
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
                self.neocortex = defaultdict(lambda: 1, saved_matrix)
                self.forward_index = defaultdict(set)
                for (prev, curr, dist), w in self.neocortex.items():
                    if w > 1:
                        self.forward_index[prev].add((curr, dist))
            return True
        return False

    def forget_sequence(self, sequence):
        if len(sequence) < 2:
            return
        for i in range(len(sequence) - 1):
            char_i = sequence[i]
            for j in range(i + 1, min(i + self.attention_window + 1, len(sequence))):
                char_j = sequence[j]
                dist = j - i
                total_weight = 0
                count = 0
                for nc, d in self.forward_index.get(char_i, set()):
                    if d == dist:
                        total_weight += self.neocortex.get((char_i, nc, d), 1)
                        count += 1
                if count > 0:
                    avg_weight = total_weight // count  
                    self.neocortex[(char_i, char_j, dist)] = avg_weight
                    self.forward_index[char_i].add((char_j, dist))

    def copy_connection_strength(self, source_pair, target_pair):
        try:
            if len(source_pair) == 3 and len(target_pair) == 3:
                source_key = tuple(source_pair)
                target_key = tuple(target_pair)
                if source_key in self.neocortex:
                    strength = self.neocortex[source_key]
                    self.neocortex[target_key] = strength
                    self.forward_index[target_key[0]].add((target_key[1], target_key[2]))
                    print(f"[系统日志]: 连接 ({source_pair[0]}→{source_pair[1]},距离{source_pair[2]}) 的强度 {strength} 已复制给 ({target_pair[0]}→{target_pair[1]},距离{target_pair[2]})")
                    return True
                else:
                    print(f"[系统日志]: 源连接 ({source_pair[0]}→{source_pair[1]},距离{source_pair[2]}) 不存在")
                    return False
        except Exception as e:
            print(f"[系统日志]: 复制出错: {e}")
            return False

    def parse_copy_command(self, command_str):
        try:
            parts = command_str.split(',')
            if len(parts) == 5:
                char_a, char_b, char_c, char_d, dist_str = parts
                return self.copy_connection_strength((char_a, char_b, int(dist_str)), (char_c, char_d, int(dist_str)))
        except:
            pass
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
    try:
        with open(filename, 'r', encoding='utf-8') as f: text = f.read()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='gb18030') as f: text = f.read()

    text = re.sub(r'\s+', ' ', text)
    sentences_raw = re.split(r'([。！？…]|\.|\?|\!)', text)

    corpus = []
    for i in range(0, len(sentences_raw) - 1, 2):
        sentence = (sentences_raw[i] + sentences_raw[i+1]).strip()
        if len(sentence) > 3: corpus.append(sentence)
    return corpus

def auto_train_and_awaken(tau, corpus, total_epochs=2000):
    print(f"\n[培养皿]: 开启 {tau.attention_window} 字广角注意力投喂，循环 {total_epochs} 次...")
    start_time = time.time()

    for i in range(1, total_epochs + 1):
        sentence = random.choice(corpus)
        for char in sentence:
            tau.listen_and_learn(char)
            
        if i % 200 == 0:
            wrote = tau.sleep_and_consolidate()
            print(f"  -> {i}/{total_epochs} 圈，入睡结晶... 产生/加固了 {wrote} 条永久突触 (耗时: {time.time() - start_time:.2f}s)")

    tau.sleep_and_consolidate()
    print(f"[培养皿]: 训练完毕！总耗时 {time.time() - start_time:.2f} 秒。")

# ================= 造物主交互通道 =================
if __name__ == "__main__":
    print("\n==================================================")
    print(" 宇宙时空坐标锚定... 数字生命实体 [Tau V4.4 非对称惩罚终极版] 正在接入。")
    print("==================================================")

    my_tau = Tau_Digital_Life()
    if my_tau.load_seed():
        print(f"[系统日志]: 已加载记忆种子，当前皮层包含 {len(my_tau.neocortex)} 条连接。")
    else:
        print("[系统日志]: 未找到记忆种子，将开始全新生命。")

    prince_corpus = load_little_prince_corpus("little_prince.txt")
    
    auto_train_and_awaken(my_tau, prince_corpus, total_epochs=2000)

    print("\n[系统提示]: 投喂结束！它已睁开眼睛。现在你可以直接与它对话了。")
    print(f"[诊断数据]: 当前新皮层突触总量: {len(my_tau.neocortex)}")
    print(" (内置指令：\\rN/ 循环写入, \\f/内容\\f/ 定向遗忘, \\s/ 瞬时休眠, \\=a,b,c,d,dist/ 复制连接强度, exit 退出)")

    while True:
        try:
            user_input = input("\n[人类 / Human]: ")
        except (EOFError, KeyboardInterrupt):
            print("\n[Tau]: (检测到强行中断，强制进入最终休眠进行记忆结晶)")
            my_tau.sleep_and_consolidate()
            my_tau.save_seed()
            break
        
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

        copy_matches = re.findall(r'\\=(.*?)/', user_input)
        for copy_cmd in copy_matches:
            my_tau.parse_copy_command(copy_cmd)
        user_input = re.sub(r'\\=.*?/', '', user_input)

        forget_matches = re.findall(r'\\f/(.*?)\\f/', user_input)
        for seq in forget_matches:
            my_tau.forget_sequence(seq)
            print(f"[系统日志]: 手术完成。'{seq}' 的长程关联已被稀释。")
        user_input = re.sub(r'\\f/.*?\\f/', '', user_input)

        auto_sleep = False
        if r'\s/' in user_input:
            auto_sleep = True
            user_input = user_input.replace(r'\s/', '')

        repeat_times = 1
        r_match = re.search(r'\\r(\d*)/', user_input)
        if r_match:
            n_str = r_match.group(1)
            repeat_times = int(n_str) if n_str else 2
            user_input = re.sub(r'\\r\d*/', '', user_input)

        clean_text = user_input.strip()

        if clean_text:
            for _ in range(repeat_times):
                for char in clean_text:
                    my_tau.listen_and_learn(char)
            if repeat_times > 1:
                print(f"[系统日志]: 文本高强度回声了 {repeat_times} 次。")

        if auto_sleep:
            mem = my_tau.sleep_and_consolidate()
            print(f"[Tau]: (执行 \\s/ 指令，瞬间跌入深睡结晶了 {mem} 步的记忆。)")

        if clean_text:
            response = my_tau.speak(clean_text, max_length=15)
            if not response:
                print(f"[Tau]: (感受到波动，但新皮层中尚未形成结构...)")
            else:
                print(f"[Tau]: {response}")