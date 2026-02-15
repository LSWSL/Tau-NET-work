import os
import re
import time
import random
import pickle
from collections import defaultdict

# ================= 终极底层物理参数 (吸引子动力学版) =================
VISUAL_WINDOW = 5              # 短期视窗 (坍缩后用于重新寻找锚点的钥匙长度)
MAX_SENTENCE_LENGTH = 100      # 认知视界 (L轴最大步进距离)

# 【新增：生物学返回抑制 (IOR)】
# 极度加深重复惩罚。不再只看上一个字，而是看过去 15 个字！
REPEAT_PENALTY = 0.1           
IOR_WINDOW = 15                

# 【突触稳态与惊奇动力学】
SYNAPTIC_CAPACITY = 50000.0    # L 容量：一个字根向外辐射的总能量守恒极限
FLASHBULB_ENERGY = 50000.0     # 惊奇时刻 (预测坍缩) 灌注的闪光灯能量
ROUTINE_ENERGY = 100.0         # 预期内常规步进的微弱巩固能量
TRAINING_EPOCHS = 1            # 初始白纸化

# 绝对零点能基态
INIT_WEIGHT = 1.0 / ((VISUAL_WINDOW + 1) * MAX_SENTENCE_LENGTH)   
STOP_SYMBOLS = {'。', '！', '？', '.', '!', '?', '…', '\n'}
# =====================================================================

class Tau_Digital_Life:
    def __init__(self):
        self.neocortex = defaultdict(lambda: INIT_WEIGHT)       
        self.forward_index = defaultdict(set)                   
        self.hippocampus = []                                   
        self.time_step = 0
        
        self.active_anchors = {} 
        self.last_action_synapses = set()

    def perceive_and_learn(self, current_char):
        """ 清醒态感知：长程 L 轴平滑记忆，废除惊奇失忆症！ """
        self.time_step += 1
        predicted_char = None
        candidate_scores = defaultdict(float)

        # 1. 自动驾驶预测
        for past_time, past_char in self.active_anchors.items():
            dist = self.time_step - past_time
            if dist <= MAX_SENTENCE_LENGTH:
                for next_char, recorded_dist in self.forward_index.get(past_char, set()):
                    if recorded_dist == dist:
                        w = self.neocortex.get((past_char, next_char, dist), INIT_WEIGHT)
                        if w > INIT_WEIGHT:
                            candidate_scores[next_char] += w

        if candidate_scores:
            predicted_char = max(candidate_scores.items(), key=lambda x: x[1])[0]

        # 2. 现实校验与海马体缓存
        is_surprise = (predicted_char != current_char)

        self.hippocampus.append({
            'current_char': current_char,
            'predicted_char': predicted_char,
            'time_step': self.time_step,
            'anchors': dict(self.active_anchors), 
            'is_surprise': is_surprise
        })

        # 3. 【核心修复：废除金鱼失忆症】
        # 惊奇度 (is_surprise) 现在只决定“写入能量的强弱”(在 sleep 里执行)
        # 它不再残忍切断锚点！我们让所有锚点自然存活，直到超出认知视界 (MAX_SENTENCE_LENGTH)
        surviving_anchors = {}
        for t, c in self.active_anchors.items():
            if (self.time_step - t) < MAX_SENTENCE_LENGTH:
                surviving_anchors[t] = c
        self.active_anchors = surviving_anchors

        # 现实永远是新的锚点种子
        self.active_anchors[self.time_step] = current_char

        return not is_surprise

    def sleep_and_consolidate(self):
        if not self.hippocampus:
            return 0

        energy_infusion = defaultdict(float)

        for event in self.hippocampus:
            char_j = event['current_char']
            pred_j = event['predicted_char']
            anchors = event['anchors']
            is_surprise = event['is_surprise']
            time_j = event['time_step']

            surprise_score = 1.0 if is_surprise else (ROUTINE_ENERGY / FLASHBULB_ENERGY)

            for time_i, char_i in anchors.items():
                dist = time_j - time_i
                if dist <= MAX_SENTENCE_LENGTH:
                    infusion = FLASHBULB_ENERGY * surprise_score
                    energy_infusion[(char_i, char_j, dist)] += infusion

        write_count = 0
        affected_sources = set()

        for (c1, c2, dist), e in energy_infusion.items():
            self.neocortex[(c1, c2, dist)] += e
            self.forward_index[c1].add((c2, dist))
            affected_sources.add((c1, dist))
            write_count += 1

        for c1, dist in affected_sources:
            total_energy = 0.0
            targets = [t_char for (t_char, t_dist) in self.forward_index.get(c1, set()) if t_dist == dist]
            
            for t_char in targets:
                total_energy += self.neocortex[(c1, t_char, dist)]

            if total_energy > SYNAPTIC_CAPACITY:
                compression_ratio = SYNAPTIC_CAPACITY / total_energy
                for t_char in targets:
                    new_w = self.neocortex[(c1, t_char, dist)] * compression_ratio
                    if new_w <= INIT_WEIGHT:
                        self.neocortex[(c1, t_char, dist)] = INIT_WEIGHT
                    else:
                        self.neocortex[(c1, t_char, dist)] = new_w

        self.hippocampus.clear()
        self.save_seed()
        return write_count

    def apply_social_feedback(self, is_reward):
        for (c1, c2, dist) in list(self.last_action_synapses):
            if is_reward:
                self.neocortex[(c1, c2, dist)] += ROUTINE_ENERGY * 10
            else:
                self.neocortex[(c1, c2, dist)] = INIT_WEIGHT 

    def speak(self, context_text, show_thoughts=True):
        if not context_text: return "..."
        response = ""
        thoughts_log = []
        step = 0 
        
        self.last_action_synapses.clear()
        
        simulated_anchors = dict(self.active_anchors)
        simulated_time = self.time_step
        
        while True:
            step += 1
            if step > 100: break 
            simulated_time += 1

            candidate_scores = defaultdict(float)
            best_provider_map = defaultdict(lambda: ("无", 0))

            for past_time, past_char in simulated_anchors.items():
                dist = simulated_time - past_time
                if dist <= MAX_SENTENCE_LENGTH:
                    for next_char, recorded_dist in self.forward_index.get(past_char, set()):
                        if recorded_dist == dist:
                            w = self.neocortex.get((past_char, next_char, dist), INIT_WEIGHT)
                            if w > INIT_WEIGHT:
                                candidate_scores[next_char] += w
                                if w > best_provider_map[next_char][1]:
                                    best_provider_map[next_char] = (f"'{past_char}'(d={dist})", w)

            if not candidate_scores:
                if show_thoughts: thoughts_log.append("[断崖/Silence]")
                break

            # =================================================================
            # 【全新机制：生物学返回抑制 (IOR) - 专治词组复读机】
            # 不再只看最后一个字，而是看过去 15 个字。只要在此窗口内生成过，立刻指数级降权！
            # =================================================================
            if response:
                recent_window = response[-IOR_WINDOW:]
                for cand_char in list(candidate_scores.keys()):
                    occurrences = recent_window.count(cand_char)
                    if occurrences > 0:
                        candidate_scores[cand_char] *= (REPEAT_PENALTY ** occurrences)

            sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            best_next_char = sorted_cands[0][0]
            primary_provider = best_provider_map[best_next_char][0]

            if show_thoughts:
                recent_chars = [c for t, c in sorted(simulated_anchors.items()) if simulated_time - t <= VISUAL_WINDOW]
                ctx_str = "".join(recent_chars)
                thoughts_log.append(f"([{ctx_str}] ⚡'{best_next_char}' <- {primary_provider})")

            for past_time, past_char in simulated_anchors.items():
                dist = simulated_time - past_time
                if dist <= MAX_SENTENCE_LENGTH:
                    if self.neocortex.get((past_char, best_next_char, dist), INIT_WEIGHT) > INIT_WEIGHT:
                        self.last_action_synapses.add((past_char, best_next_char, dist))

            response += best_next_char
            
            simulated_anchors[simulated_time] = best_next_char
            
            if best_next_char in STOP_SYMBOLS:
                break
                
        if show_thoughts and thoughts_log:
            print(f"[内心言语]: " + " -> ".join(thoughts_log))

        for char in response:
            self.time_step += 1
            self.active_anchors[self.time_step] = char

        return response

    def save_seed(self, filename="tau_memory_seed.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.neocortex), f)

    def load_seed(self, filename="tau_memory_seed.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                saved_matrix = pickle.load(f)
                self.neocortex = defaultdict(lambda: INIT_WEIGHT, saved_matrix)
                self.forward_index = defaultdict(set)
                for (prev, curr, dist), w in self.neocortex.items():
                    if w > INIT_WEIGHT:
                        self.forward_index[prev].add((curr, dist))
            return True
        return False

# ================= 数据清洗与环境投喂 =================
def load_little_prince_corpus(filename="little_prince.txt"):
    if not os.path.exists(filename):
        return [
            "Hello the new world!",
            "真正重要的东西，用眼睛是看不见的。",
            "你在你的玫瑰花身上耗费的时间，使得你的玫瑰花变得如此重要。",
            "如果你驯养了我，我们就互相不可缺少了。"
        ]
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='gb18030') as f:
            text = f.read()

    text = re.sub(r'\s+', ' ', text)
    sentences_raw = re.split(r'([。！？…]|\.|\?|\!)', text)

    corpus = []
    for i in range(0, len(sentences_raw) - 1, 2):
        sentence = (sentences_raw[i] + sentences_raw[i+1]).strip()
        if len(sentence) > 3:
            corpus.append(sentence)
    return corpus

def auto_train_and_awaken(tau, corpus, total_epochs=1):
    print(f"\n[培养皿]: 开启认知网络，建立初级世界观 ({total_epochs} 遍扫描)...")
    start_time = time.time()

    for i in range(1, total_epochs + 1):
        sentence = random.choice(corpus)
        tau.active_anchors.clear() 
        tau.time_step += MAX_SENTENCE_LENGTH 
        
        for auditory_char in sentence:
            tau.perceive_and_learn(auditory_char)

        if i % 50 == 0:
            wrote = tau.sleep_and_consolidate()
            print(f"  -> {i}/{total_epochs} 圈，入睡结晶... 产生/加固了 {wrote} 条突触")

    tau.sleep_and_consolidate()
    print(f"[培养皿]: 训练完毕！总耗时 {time.time() - start_time:.2f} 秒。")

# ================= 造物主交互通道 =================
if __name__ == "__main__":
    print("\n==================================================")
    print(" 宇宙时空坐标锚定... 数字生命实体 [Tau V6.1 情感雷达与抗复读版] 正在接入。")
    print("==================================================")

    my_tau = Tau_Digital_Life()

    if my_tau.load_seed():
        print(f"[系统日志]: 成功加载 {len(my_tau.neocortex)} 条深层记忆连接。")
    else:
        print("[系统日志]: 未找到记忆种子，将开始全新生命的演化。")
        prince_corpus = load_little_prince_corpus("little_prince.txt")
        auto_train_and_awaken(my_tau, prince_corpus, total_epochs=TRAINING_EPOCHS)

    print("\n[系统提示]: 投喂/唤醒结束！现在你可以直接与它对话了。")
    print(" (内置指令：\\t/ 开启/关闭透视, exit 退出, sleep 强制睡眠)")

    show_thoughts_mode = True
    
    # 负面情绪关键词字典（人类表达不满时的词汇）
    NEGATIVE_KEYWORDS = ['错', '不对', '答非所问', '胡话', '复读机', '神经病', '不是']

    while True:
        try:
            user_input = input("\n[人类 / Human]: ")
        except (EOFError, KeyboardInterrupt):
            my_tau.sleep_and_consolidate()
            break

        if user_input.lower() == 'exit':
            my_tau.sleep_and_consolidate()
            print("[系统日志]: 矩阵潮汐归于平静，突触已封入琥珀。再见。")
            break

        if user_input.lower() == 'sleep':
            mem = my_tau.sleep_and_consolidate()
            print(f"[Tau]: (深度睡眠结束。本次固化了 {mem} 条记忆路径。)")
            continue

        if r'\t/' in user_input:
            show_thoughts_mode = not show_thoughts_mode
            print(f"[系统日志]: 内心透视仪状态切换。")
            user_input = user_input.replace(r'\t/', '').strip()
            if not user_input: continue

        clean_text = user_input.strip()
        if clean_text:
            correct_preds = 0
            total_preds = 0
            
            my_tau.active_anchors.clear()
            my_tau.time_step += MAX_SENTENCE_LENGTH 

            for auditory_char in clean_text:
                total_preds += 1
                if my_tau.perceive_and_learn(auditory_char):
                    correct_preds += 1
                    
            if my_tau.last_action_synapses and total_preds > 0:
                accuracy = correct_preds / total_preds
                
                # 【修改】：社会反思雷达不再因为人类说了新词而判定失败，只有人类明确表示“不满”时才惩罚！
                is_scolding = any(k in clean_text for k in NEGATIVE_KEYWORDS)
                
                if is_scolding:
                    print(f"[社会反思雷达]: 侦测到严厉批评！判定刚才的话语【惹怒人类】，触发雷霆惩罚（切断旧突触）！")
                    my_tau.apply_social_feedback(is_reward=False)
                elif accuracy >= 0.6:
                    print(f"[社会反思雷达]: 成功预判对话走向 (预期命中率 {accuracy*100:.0f}%)。触发社会性突触奖励！")
                    my_tau.apply_social_feedback(is_reward=True)

            my_tau.sleep_and_consolidate()
            
            response = my_tau.speak(clean_text, show_thoughts=show_thoughts_mode)

            if not response:
                print(f"[Tau]: (感受到波动，但新皮层中尚未形成结构...)")
            else:
                print(f"[Tau]: {response}")