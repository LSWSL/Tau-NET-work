import os
import numpy as np
import random
import math
from collections import deque
from config import WorldConfig
from body import VocalTract

class TauCore:
    """
    [Tau-Net Core Engine]
    对应论文中的核心数学模型：
    - Memory: m * 2^e
    - Learning: Integer Addition + Bitwise Shift
    - Forgetting: Stochastic Decay based on Exponent
    """
    def __init__(self, memory_size=500000):
        # 对应论文 Section 2.2: Dual-Channel Logarithmic Memory
        # m = Mantissa (Hippocampus/Fast), e = Exponent (Neocortex/Slow)
        self.memory_size = memory_size
        self.m = np.zeros(memory_size, dtype=np.uint8)
        self.e = np.zeros(memory_size, dtype=np.uint8)
        
        # 哈希参数 (Section 2.1)
        self.prime_p = 31
        
        # 学习参数 (Section 2.3.1)
        self.delta = 10  # Synaptic injection constant

    def _hash_context(self, tokens):
        """Polynomial Rolling Hash"""
        h = 0
        for token in tokens:
            # 简单的字符转整数编码
            char_code = ord(token[0]) if token else 0
            h = (h * self.prime_p + char_code) % self.memory_size
        return h

    def learn(self, context_tokens, target_token):
        """
        [Section 2.3.1] Learning: Addition and Renormalization
        """
        if not target_token: return
        
        # 构造完整模式进行哈希
        full_pattern = list(context_tokens) + [target_token]
        addr = self._hash_context(full_pattern)
        
        # 1. Addition (Hebbian-like increment)
        # 使用 int 防止 uint8 溢出
        val = int(self.m[addr]) + self.delta
        
        # 2. Renormalization (Bitwise Shift)
        # "Computationally, this is executed as a bitwise right shift"
        if val > 255:
            self.m[addr] = val >> 1  # m = m / 2
            if self.e[addr] < 255:
                self.e[addr] += 1    # e = e + 1
        else:
            self.m[addr] = val

    def predict(self, context_tokens):
        """
        从记忆中检索可能的下一个词。
        由于哈希表不支持反向查询，这里我们采用简单的‘采样’或‘基于上下文的检索’。
        为了适配 VatBrain 的接口，我们返回一个基于当前上下文强度的伪概率分布。
        
        注意：真正的 O(1) 预测需要反向索引，但为了保持论文中 "Count-Min Sketch" 的
        简洁性，我们在实验中通常只追踪特定的感兴趣候选项，或者这里做一个简化：
        我们只检查最常见的字符集。
        """
        candidates = WorldConfig.VALID_SYMBOLS
        dist = {}
        total_strength = 0.0
        
        for char in candidates:
            # 试探性哈希：如果下一个是 char，强度是多少？
            full_pattern = list(context_tokens) + [char]
            addr = self._hash_context(full_pattern)
            
            m_val = self.m[addr]
            e_val = self.e[addr]
            
            if m_val > 0:
                # V = m * 2^e
                strength = float(m_val) * (2.0 ** float(e_val))
                dist[char] = strength
                total_strength += strength
                
        if total_strength == 0:
            return {}
            
        # 归一化为概率
        return {k: v / total_strength for k, v in dist.items()}

    def forget_stochastic(self):
        """
        [Section 2.3.2] Forgetting: Stochastic Decay
        m_{t+1} = m_t - I(r < 2^-e)
        """
        # 生成随机因子 r ~ U[0, 1)
        # 为了高效，我们只对非零元素操作 (虽然 numpy 批量操作很快)
        # 这里模拟论文的逐个更新逻辑的向量化版本
        
        # 计算衰减阈值 threshold = 1.0 / (2^e)
        # 优化：位运算 2^e
        # 注意：float 转换
        decay_thresholds = 1.0 / np.power(2.0, self.e)
        
        random_factors = np.random.random(self.memory_size)
        
        # 只有当 随机数 < 阈值 时才衰减
        decay_mask = random_factors < decay_thresholds
        
        # 且 m 必须大于 0
        active_mask = self.m > 0
        
        # 执行衰减 (减 1)
        target_indices = np.where(decay_mask & active_mask)
        self.m[target_indices] -= 1
        
        # 如果 m 减到 0，是否保留 e？
        # 论文隐含：如果 m=0，该条目实际上失效。
        # 这里可以选择是否重置 e，通常保持 e 作为长期结构痕迹更好，
        # 或者当 m=0 时也把 e 清零以节省空间。
        # 此处保持论文的极简公式：只减 m。

    def save(self):
        np.save(os.path.join(WorldConfig.BIOLOGY_DIR, "tau_m.npy"), self.m)
        np.save(os.path.join(WorldConfig.BIOLOGY_DIR, "tau_e.npy"), self.e)

    def load(self):
        try:
            self.m = np.load(os.path.join(WorldConfig.BIOLOGY_DIR, "tau_m.npy"))
            self.e = np.load(os.path.join(WorldConfig.BIOLOGY_DIR, "tau_e.npy"))
        except:
            print("[Tau-Net] No existing brain dump found. Starting fresh.")

class WorkingMemory:
    """简单的工作记忆，用于保存最近的 N 个 Token"""
    def __init__(self, capacity=5):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, token):
        self.buffer.append(token)
        
    def get_context(self):
        return list(self.buffer)

class VatBrain:
    """
    [Agent Wrapper]
    将 TauCore 包装为仿真所需的 Agent。
    """
    def __init__(self):
        self.core = TauCore()  # <--- 核心替换为 Tau-Net
        self.core.load()
        self.wm = WorkingMemory(capacity=WorldConfig.CONTEXT_WINDOW)
        self.body = VocalTract()
        
        self.state = "AWAKE"
        self.sleep_timer = 0
        self.age = 0
        self.silence_streak = 0
        
        self.current_motor_output = (None, 0.0)
        self.current_attention_focus = (None, 0.0)
        
        print(f"[Subject] Tau-Net Architecture Activated.")

    def perceive(self, signal):
        if self.state == "ASLEEP":
            self.process_sleep()
            return None, 0.0

        if not signal: return None, 0.0
        self.age += 1

        # 1. 提取输入 (Focus on Fovea)
        fovea_char = signal.visual[WorldConfig.FOVEA_INDEX]
        
        # 2. 获取当前上下文
        context = self.wm.get_context()
        
        # 3. 预测 (Predict) -> 用于注意力/说话
        # 基于上下文预测下一个最可能的词
        dist = self.core.predict(context)
        
        # 采样逻辑 (Temperature Sampling)
        best_char, best_prob = self._sample(dist)
        self.current_attention_focus = (best_char, best_prob)
        
        # 4. 学习 (Learn) -> Online Learning
        # Tau-Net 支持在线学习，不需要等到睡眠
        # 但我们仍然保留睡眠作为深度去噪 (Stochastic Decay) 的时间
        if signal.structure not in ["NOISE_ARTIFACT", "BOUNDARY_START"]:
            self.core.learn(context, fovea_char)
        
        # 5. 更新工作记忆
        self.wm.push(fovea_char)
        
        # 6. 行动 (Act)
        motor_char = None
        motor_conf = 0.0
        
        if self.silence_streak < 7 and best_char:
            output_str = self.body.articulate(best_char, best_prob)
            if output_str:
                self.current_motor_output = (output_str, best_prob)
                self.silence_streak = 0
                motor_char = output_str
                motor_conf = best_prob
            else:
                self.current_motor_output = (None, 0.0)
                self.silence_streak += 1
        else:
            self.body.rest()
            self.silence_streak += 1
            
        return motor_char, motor_conf

    def _sample(self, dist):
        if not dist: return None, 0.0
        # 简单的贪婪或概率采样
        chars = list(dist.keys())
        probs = list(dist.values())
        if not chars: return None, 0.0
        # 简单的加权随机
        chosen = random.choices(chars, weights=probs, k=1)[0]
        return chosen, dist[chosen]

    def trigger_sleep(self):
        self.state = "ASLEEP"
        self.sleep_timer = WorldConfig.SLEEP_DURATION
        print("\n[Brain] Entering Hypnagogic State (Stochastic Pruning)...")

    def process_sleep(self):
        """
        在 Tau-Net 中，睡眠 = 密集执行随机衰减 (Forgetting)
        这模拟了去除海马体噪声、保留皮层结构的过程。
        """
        if self.sleep_timer > 0:
            # 每次睡眠 tick 执行一次全局衰减
            self.core.forget_stochastic()
            self.sleep_timer -= 1
        else:
            self.state = "AWAKE"
            self.core.save() # 醒来前保存
            self.silence_streak = 0

    def save(self):
        self.core.save()