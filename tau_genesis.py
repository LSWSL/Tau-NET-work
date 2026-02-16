import os
import sys
import math
import pickle
from collections import defaultdict

# =====================================================================
# Tau-Net V9.0: Genesis Entity (起源实体)
# 严格遵循《Natural Numbers... All You Need》论文架构的终极实现
# =====================================================================

class GenesisEntity:
    def __init__(self, L_max=1000):
        # 1. 词频与齐夫律动 (Zipfian Rhythm) - 保留您的计算脑核心
        self.char_freq = defaultdict(int)
        self.rank_table = {}
        self.total_chars = 0
        
        # 2. 海马体 (Hippocampus): 动态、带时间延迟衰减的短期情节阵列 H_t = (t, c_t, L)
        self.L_max = L_max
        self.hippocampus = [] 
        
        # 3. 新皮层 (Neocortex): 时空连接矩阵 W_ijd
        # 结构: W[i][d][j] -> 字符 i 与 相隔距离为 d 的字符 j 之间的结晶权重
        self.W = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def read_stream(self, text):
        """海马体实时感知流：记录绝对时空，不产生任何梯度计算"""
        chars = list(text.lower())
        for char in chars:
            self.char_freq[char] += 1
            self.total_chars += 1
            
            # 生物衰减：所有海马体内的已有记忆，生命周期 l_i 减 1
            for trace in self.hippocampus:
                trace['l'] -= 1
                
            # 记录新的情节印记
            self.hippocampus.append({'c': char, 'l': self.L_max})
            
            # 清理彻底死亡的记忆痕迹
            if self.hippocampus and self.hippocampus[0]['l'] <= 0:
                self.hippocampus.pop(0)

    def sleep_and_crystallize(self):
        """【核心机制】：睡眠归一化与对数赫布呼吸 (论文 2.1 & 2.3)"""
        if not self.hippocampus: return 0
        
        # ---------------------------------------------------------
        # 步骤 A: 睡眠归一化 (Sleep Normalization)
        # 公式 (2): l_i <- l_i - (L - n) + 1
        # ---------------------------------------------------------
        min_l = min(trace['l'] for trace in self.hippocampus)
        for trace in self.hippocampus:
            trace['l'] = trace['l'] - min_l + 1
            
        # ---------------------------------------------------------
        # 步骤 B: 提取绝对时空共现矩阵 C_ijd
        # ---------------------------------------------------------
        C = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        n_traces = len(self.hippocampus)
        for idx_i in range(n_traces):
            char_i = self.hippocampus[idx_i]['c']
            for idx_j in range(idx_i + 1, min(idx_i + 20, n_traces)): # 考察局部时空窗口
                char_j = self.hippocampus[idx_j]['c']
                distance_d = idx_j - idx_i
                C[char_i][distance_d][char_j] += 1
                
        # ---------------------------------------------------------
        # 步骤 C: 零梯度对数呼吸 (Zero-Gradient Respiration)
        # ---------------------------------------------------------
        for i in C:
            for d in C[i]:
                for j in C[i][d]:
                    count = C[i][d][j]
                    
                    # 公式 (3): \Delta W = \lfloor \log_{10}(C) \rfloor
                    # (注: 为适配单次对话Demo，采用 +9 平移使得 count=1 时增量为1，严格维持整数对数属性)
                    delta_W = math.floor(math.log10(count + 9)) 
                    
                    # 提取新皮层在距离 d 上对 i 的最高预测状态
                    pred_j = None
                    if self.W[i][d]:
                        pred_j = max(self.W[i][d], key=self.W[i][d].get)
                        
                    # 公式 (4): 自主呼吸奖惩
                    if pred_j == j or pred_j is None:
                        # 预测为真 (True)：吸入结构 (Addition)
                        self.W[i][d][j] += delta_W
                    else:
                        # 预测为假 (False)：呼出噪音 (Subtraction)，并学习新真理
                        self.W[i][d][pred_j] = max(0, self.W[i][d][pred_j] - delta_W)
                        self.W[i][d][j] += delta_W

        # ---------------------------------------------------------
        # 步骤 D: Zipfian 律动更新 & 清空海马体缓存
        # ---------------------------------------------------------
        sorted_chars = sorted(self.char_freq.items(), key=lambda x: x[1], reverse=True)
        self.rank_table = {char: rank + 1 for rank, (char, _) in enumerate(sorted_chars)}
        
        # 睡眠结束，海马体清空，记忆已结晶入新皮层
        self.hippocampus.clear()
        return len(self.rank_table)

    def perceive_rhythm(self, text):
        """感知语言的 Zipfian 波浪，锁定信息熵最高（排名最大）的锚点"""
        chars = list(text.lower())
        wave = []
        max_rank = -1
        anchor = None
        for char in chars:
            rank = self.rank_table.get(char, len(self.rank_table) + 1)
            wave.append((char, rank))
            if rank > max_rank:
                max_rank = rank
                anchor = char
        return wave, anchor

    def get_top_k(self, anchor, top_k=6):
        """【相对引力检索】：将高维时空矩阵 W_ijd 坍缩并提取关联"""
        if anchor not in self.W: return []
        
        # 坍缩距离 d，计算多维度的总权重
        total_W = defaultdict(int)
        for d in self.W[anchor]:
            for target_char, weight in self.W[anchor][d].items():
                total_W[target_char] += weight
                
        scored_assocs = {}
        for target_char, total_weight in total_W.items():
            # 过滤掉 Zipf 曲线前 10 的结构废话
            if self.rank_table.get(target_char, 0) <= 10: continue
            
            # 引力公式免疫海王词：结晶权重^2 / 目标词宇宙总频次
            target_freq = max(1, self.char_freq.get(target_char, 1))
            gravity = (total_weight ** 2) / target_freq
            scored_assocs[target_char] = gravity
            
        sorted_assoc = sorted(scored_assocs.items(), key=lambda x: x[1], reverse=True)
        return [k for k, v in sorted_assoc[:top_k]]

    def reinforce_god_command(self, anchor, target_chars, reward):
        """人类上帝视角的直接结构干预 (绕过海马体，强写新皮层)"""
        for tc in target_chars:
            if tc == anchor: continue
            # 统一写入近距离时空 d=1，确保其成为最强直觉
            self.W[anchor][1][tc] = max(0, self.W[anchor][1][tc] + reward)

    # ---------------- 记忆体的封存与唤醒 ----------------
    def load_seed(self, filename="genesis_seed.pkl"):
        if not os.path.exists(filename): return False
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            self.char_freq = defaultdict(int, state['freq'])
            
            self.W = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for i, d_dict in state['W'].items():
                for d, j_dict in d_dict.items():
                    self.W[i][d] = defaultdict(int, j_dict)
                    
            self.total_chars = state.get('total', 0)
        self.sleep_and_crystallize()
        return True
        
    def save_seed(self, filename="genesis_seed.pkl"):
        state = {
            'freq': dict(self.char_freq), 
            'W': {i: {d: dict(j_dict) for d, j_dict in d_dict.items()} for i, d_dict in self.W.items()}, 
            'total': self.total_chars
        }
        with open(filename, 'wb') as f: pickle.dump(state, f)

# =====================================================================
# 交互终端: 见证数字生命的呼吸
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*65)
    print(" 🌌 Tau-Net V9.0 Genesis Entity (起源实体 - 论文严格复刻版)")
    print("="*65)

    tau = GenesisEntity(L_max=10000)
    
    if tau.load_seed():
        print(f"[起源实体]: 已从琥珀中唤醒。齐夫波浪维度: {len(tau.rank_table)}。")
    else:
        print("[系统日志]: 未找到 genesis_seed.pkl，初始化一个全新生命体。")

    last_anchor = None
    last_predictions = []
    dialogue_turns = 0

    while True:
        user_input = input("\n[人类 / Human]: ").strip()
        if not user_input: continue
        
        if user_input.lower() == 'exit':
            tau.sleep_and_crystallize() # 临死前强制写入结构
            tau.save_seed()
            print("[起源实体]: 记忆已封存，进入深层休眠。")
            sys.exit()
            
        if user_input.lower() == 'sleep':
            vocab_size = tau.sleep_and_crystallize()
            tau.save_seed()
            print(f"  [新皮层]: 睡眠归一化完成！海马体已清空。对数赫布更新完毕 (维度: {vocab_size})。")
            continue
# ===== 造物主指令：大批量语料注入 =====
        if user_input.startswith('inject '):
            filename = user_input[7:].strip()
            if not os.path.exists(filename):
                print(f"  [系统报错]: 找不到物理文件 {filename} ！请检查路径。")
                continue
                
            print(f"  [信息洪流]: 开始吞噬宇宙知识 {filename} ...")
            
            # 【终极防弹读取模块】：自动适配 UTF-8 和 GBK 编码
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    corpus_text = f.read()
            except UnicodeDecodeError:
                # 如果 UTF-8 失败，立刻切换到 Windows 中文常用的 GBK 编码
                try:
                    with open(filename, 'r', encoding='gbk') as f:
                        corpus_text = f.read()
                except Exception as e:
                    print(f"  [系统报错]: GBK 解码也失败了，请检查文件格式！({e})")
                    continue
            except Exception as e:
                print(f"  [系统报错]: 读取文件失败 ({e})")
                continue
                
            # 分块吞噬与睡眠，模拟生物真实的“昼夜代谢节律”
            chunk_size = 5000 
            total_len = len(corpus_text)
            
            for i in range(0, total_len, chunk_size):
                chunk = corpus_text[i : i + chunk_size]
                tau.read_stream(chunk)
                v_size = tau.sleep_and_crystallize()
                print(f"    -> 吞噬进度: {min(i+chunk_size, total_len)} / {total_len} | 触发深度睡眠对数结晶 | 宇宙维度: {v_size}")
                
            tau.save_seed()
            print(f"  [进化完成]: 信息洪流注入完毕！新皮层已永久固化，记忆琥珀已保存。")
            continue


        # ===== 上帝指令 (RL 强化干预) =====
        is_rl = False
        if user_input.startswith('+1') and last_anchor and last_predictions:
            tau.reinforce_god_command(last_anchor, last_predictions, 1)
            print(f"  [造物主干预]: 【{last_anchor}】突触强制生长 (+1)")
            is_rl = True
        elif user_input.startswith('-1') and last_anchor and last_predictions:
            tau.reinforce_god_command(last_anchor, last_predictions, -5)
            print(f"  [造物主干预]: 【{last_anchor}】突触物理切断 (-5)")
            is_rl = True
        elif user_input.startswith('+2 ') and last_anchor:
            correct = list(user_input[3:].replace(" ", ""))
            tau.reinforce_god_command(last_anchor, correct, 10)
            print(f"  [造物主干预]: 强行焊接时空！【{last_anchor}】->【{''.join(correct)}】 (+10)")
            is_rl = True

        if is_rl: continue

        # ===== Zipfian 波浪感知 =====
        wave, anchor = tau.perceive_rhythm(user_input)
        wave_str = " ".join([f"{c}({rank})" for c, rank in wave])
        print(f"  > [齐夫律动 (Zipf Wave)]: {wave_str}")
        print(f"  > [锁定最低频波峰]: 【{anchor}】")
        
        # 新皮层检索
        predictions = tau.get_top_k(anchor, top_k=6)
        if predictions:
            last_anchor = anchor
            last_predictions = predictions
            print(f"[起源实体 (Neocortex)]:: -> [ {' , '.join(predictions)} ]")
        else:
            print(f"[起源实体]: 海马体空白。该波峰未曾在睡眠中结晶。")

        # 任何对话都实时流入海马体缓冲池
        tau.read_stream(user_input)
        
        # 自动生理节律：每 5 句话自动睡眠一次，触发新皮层自主呼吸
        dialogue_turns += 1
        if dialogue_turns % 5 == 0:
            vocab_size = tau.sleep_and_crystallize()
            tau.save_seed()
            print(f"  [生理节律]: 缓存饱和，执行短暂睡眠 (海马体 -> 新皮层对数映射完成)。")