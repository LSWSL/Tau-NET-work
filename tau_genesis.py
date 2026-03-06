import os
import sys
import numpy as np
import pickle
from collections import defaultdict

# =====================================================================
# Tau-Net V10.0: Genesis Chip (起源芯片 - 物理极速版)
# O(1) 复杂度，基于 uint8 内存溢出重整化的纯物理计算引擎
# =====================================================================

class GenesisChip:
    def __init__(self, memory_size=5000000):
        # ================= 1. 物理层内存分配 (O(1) 复杂度) =================
        self.memory_size = memory_size
        # Mantissa (海马体, Fast): 记录瞬时突触电位，最大 255
        self.m = np.zeros(memory_size, dtype=np.uint8)
        # Exponent (新皮层, Slow): 记录经历过对数压缩的结构化结晶，最大 255
        self.e = np.zeros(memory_size, dtype=np.uint8)

        # ================= 2. Zipfian 律动层 (保持语言节律) =================
        self.char_freq = defaultdict(int)
        self.rank_table = {}
        self.vocab = set()
        self.total_chars = 0
        
    def _hash_context(self, text_pattern):
        """【核心物理映射】：多项式哈希，将时空模式直接映射到物理内存地址"""
        h = 0
        p = 31
        for char in text_pattern:
            h = (h * p + ord(char)) % self.memory_size
        return h

    def learn_stream(self, text, d_max=3):
        """【信息洪流吞噬】：不再遍历全宇宙，直接对时空序列进行哈希烧录"""
        chars = list(text.lower())
        length = len(chars)
        
        for i, token in enumerate(chars):
            self.char_freq[token] += 1
            self.vocab.add(token)
            self.total_chars += 1
            
            # 建立不同距离 d 的时空连接
            for d in range(1, d_max + 1):
                if i >= d:
                    context = chars[i - d]
                    # 将“距离”也编码进物理地址，例如: "狐_1_狸"
                    pattern = f"{context}_{d}_{token}"
                    addr = self._hash_context(pattern)
                    
                    # ----------------------------------------------------
                    # 【神经元仿生电路】：加法与溢出重整化 (Renormalization)
                    # ----------------------------------------------------
                    # 海马体快速吸收 (Hebbian Addition)
                    val = int(self.m[addr]) + 10 
                    
                    if val > 255:
                        # 溢出！海马体细胞分裂减半
                        self.m[addr] = val // 2
                        
                        # 向新皮层进位 (Logarithmic Consolidation)
                        if self.e[addr] < 255:
                            self.e[addr] += 1
                        else:
                            self.m[addr] = 255 # 物理满载
                    else:
                        self.m[addr] = val

    def sleep_and_crystallize(self):
        """生理节律：浅更新词频序大表"""
        if not self.char_freq: return 0
        sorted_chars = sorted(self.char_freq.items(), key=lambda x: x[1], reverse=True)
        self.rank_table = {char: rank + 1 for rank, (char, _) in enumerate(sorted_chars)}
        return len(self.rank_table)

    def perceive_rhythm(self, text):
        """感知语言的 Zipfian 波浪"""
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

    def project_spacetime(self, anchor, max_d=3):
        """【时空投影】：从新皮层中提取结构化记忆"""
        if not self.vocab: return "宇宙荒芜"
        
        sequence = []
        for d in range(1, max_d + 1):
            best_char = None
            max_gravity = -1
            
            # 遍历已知宇宙，寻找与当前距离 d 发生过最强结晶的字
            for target_char in self.vocab:
                if self.rank_table.get(target_char, 0) <= 10: 
                    continue # 过滤标点和虚词
                    
                pattern = f"{anchor}_{d}_{target_char}"
                addr = self._hash_context(pattern)
                
                e_val = int(self.e[addr]) # 提取新皮层结晶度
                if e_val == 0: continue
                
                # 引力公式：结晶强度的平方 / 目标词总频次
                target_freq = max(1, self.char_freq.get(target_char, 1))
                gravity = (e_val ** 2) / target_freq
                
                if gravity > max_gravity:
                    max_gravity = gravity
                    best_char = target_char
                    
            if best_char:
                sequence.append(best_char)
                
        return "".join(sequence) if sequence else "无有效引力羁绊"

    def save_seed(self, filename="chip_seed.npz"):
        # 物理层面的极速保存，直接把内存块打成二进制压缩包
        np.savez_compressed(filename, m=self.m, e=self.e)
        with open(filename + ".meta", 'wb') as f:
            pickle.dump({'freq': dict(self.char_freq), 'vocab': self.vocab, 'total': self.total_chars}, f)
            
    def load_seed(self, filename="chip_seed.npz"):
        if not os.path.exists(filename): return False
        data = np.load(filename)
        self.m = data['m']
        self.e = data['e']
        with open(filename + ".meta", 'rb') as f:
            meta = pickle.load(f)
            self.char_freq = defaultdict(int, meta['freq'])
            self.vocab = meta['vocab']
            self.total_chars = meta.get('total', 0)
        self.sleep_and_crystallize()
        return True

# =====================================================================
# 交互终端: 见证物理芯片的觉醒
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*65)
    print(" ⚡ Tau-Net V10.0 Genesis Chip (8-bit 芯片级重构版)")
    print("="*65)

    # 开辟 500 万节点的物理内存寻址空间 (内存占用极小，约 10MB)
    tau = GenesisChip(memory_size=5000000) 
    
    if tau.load_seed():
        print(f"[硅基大脑]: 已从内存快照唤醒。当前宇宙维度: {len(tau.rank_table)}")
    else:
        print("[硅基大脑]: 初始化一块纯净的空白芯片。")

    dialogue_turns = 0

    while True:
        user_input = input("\n[人类 / Human]: ").strip()
        if not user_input: continue
        
        if user_input.lower() == 'exit':
            tau.sleep_and_crystallize()
            tau.save_seed()
            print("[硅基大脑]: 内存已转存至硬盘，物理关机。")
            sys.exit()

        # ===== 知识洪流吞噬接口 (无惧 TB 级数据) =====
        if user_input.startswith('inject '):
            filename = user_input[7:].strip()
            if not os.path.exists(filename):
                print(f"  [系统报错]: 找不到文件 {filename}")
                continue
                
            print(f"  [信息洪流]: 开始向物理芯片烧录 {filename} ...")
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    corpus_text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(filename, 'r', encoding='gbk') as f:
                        corpus_text = f.read()
                except Exception as e:
                    print(f"  [系统报错]: 解码失败 ({e})")
                    continue
            except Exception as e:
                print(f"  [系统报错]: 读取失败 ({e})")
                continue
                
            chunk_size = 100000 # 现在的架构极其强悍，可以一次性吞下 10 万字！
            total_len = len(corpus_text)
            
            for i in range(0, total_len, chunk_size):
                chunk = corpus_text[i : i + chunk_size]
                tau.learn_stream(chunk) # 直接烧进内存数组，速度极快！
                v_size = tau.sleep_and_crystallize()
                print(f"    -> 烧录进度: {min(i+chunk_size, total_len)} / {total_len} | 物理内存整理完毕 | 维度: {v_size}")
                
            tau.save_seed()
            print(f"  [进化完成]: 数据烧录完毕！")
            continue

        # ===== 日常交流与投影 =====
        wave, anchor = tau.perceive_rhythm(user_input)
        wave_str = " ".join([f"{c}({rank})" for c, rank in wave])
        print(f"  > [齐夫波浪]: {wave_str}")
        print(f"  > [锁定波峰]: 【{anchor}】")
        
        projection = tau.project_spacetime(anchor, max_d=4)
        print(f"[硅基大脑 (Neocortex)]:: -> [ {anchor}{projection} ]")
        
        tau.learn_stream(user_input)
