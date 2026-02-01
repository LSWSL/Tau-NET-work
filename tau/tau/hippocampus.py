# ==============================================================================
# TAU HIPPOCAMPUS: SCIENTIFIC NOTATION DECAY
# ==============================================================================
# STRUCTURE: Dual-Channel Reservoir (Mantissa + Exponent)
# DYNAMICS:  Value -= 1 (with Borrowing)
# EFFECT:    Weak memories fade fast; Strong memories last forever.
# ==============================================================================

import torch
from config import WorldConfig

class Hippocampus:
    def __init__(self, brain_ref):
        self.brain = brain_ref
        
        # 1. 维度与通道
        self.dim = 10000 
        self.device = self.brain.device
        
        # 2. 双通道状态 (uint8)
        self.mantissa = torch.zeros(self.dim, dtype=torch.uint8, device=self.device)
        self.exponent = torch.zeros(self.dim, dtype=torch.uint8, device=self.device)
        
        # 3. 注入强度 (每次注入增加的量)
        # 设定为增加 Mantissa, 允许自动进位
        self.inject_amount = 50 

        # 缓冲区 (用于关联学习)
        from collections import deque
        self.window_size = WorldConfig.N_ORDER
        self.context_buffer = deque(maxlen=self.window_size)

    def encode_and_inject(self, atom_char):
        """
        [注入] M += 50. 如果溢出 -> M/=2, E+=1
        """
        symbol_idx = self.brain.encode_single(atom_char)
        
        # 哈希映射到 5 个位置
        seed = symbol_idx * 7919 
        indices = [(seed + i * 31337) % self.dim for i in range(5)]
        idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        
        # 1. 增加尾数
        # 注意：我们需要先转为 int16 防止直接溢出，处理完后再转回 uint8
        m_vals = self.mantissa[idx_tensor].long() + self.inject_amount
        e_vals = self.exponent[idx_tensor].long()
        
        # 2. 处理溢出 (Renormalize)
        # 如果 M > 255: M = M // 2, E = E + 1
        # 我们用循环处理多次溢出 (比如 +50 后可能很大，虽然这里一般一次就够)
        mask = m_vals > 255
        if mask.any():
            m_vals[mask] = m_vals[mask] // 2
            e_vals[mask] += 1
            # 封顶 E
            e_vals[e_vals > 255] = 255
            
        self.mantissa[idx_tensor] = m_vals.to(torch.uint8)
        self.exponent[idx_tensor] = e_vals.to(torch.uint8)

    def tick(self):
        """
        [科学衰减] Value -= 1
        逻辑：
        1. M -= 1
        2. IF M underflow (0->255) AND E > 0: E -= 1
        3. IF M underflow AND E == 0: Memory Dies (M=0, E=0)
        """
        # 1. 识别即将下溢的 (M=0)
        mask_m_zero = (self.mantissa == 0)
        
        # 2. 全局减 1 (uint8 0-1 会变成 255)
        self.mantissa -= 1
        
        # 3. 处理借位 (Borrow from Exponent)
        # 刚才 M=0 的位置，现在变成了 255，我们需要让 E-1
        # 前提是 E > 0
        mask_borrow = mask_m_zero & (self.exponent > 0)
        self.exponent[mask_borrow] -= 1
        
        # 4. 处理彻底死亡 (Death)
        # 刚才 M=0 且 E=0 的位置，现在 M变成了255 (错误)，E变成了255 (下溢，错误)
        # 我们需要把它们重置为 0
        # mask_death = mask_m_zero & (self.exponent == 0) 
        # 但由于上面我们只对 exponent>0 做了减法，所以 E=0 的位置 E 还是 0
        # 只需要把 M 修正回 0 即可
        mask_death = mask_m_zero & (self.exponent == 0)
        self.mantissa[mask_death] = 0

    def get_energy(self):
        """计算总能量: sum(M * 2^E)"""
        # 抽样计算，防止计算量太大
        m = self.mantissa[:1000].float()
        e = self.exponent[:1000].float()
        energy = (m * torch.pow(2.0, e)).sum().item()
        return energy
    
    def get_active_count(self):
        return torch.count_nonzero(self.mantissa).item()

    # --- 关联学习接口 ---
    def consolidate(self, current_atom, target_atom):
        self.context_buffer.append(current_atom)
        if len(self.context_buffer) < self.window_size: return
        
        # 注入海马体产生回声 (Short Term Memory)
        self.encode_and_inject(current_atom)
        
        # 刻录到长期记忆 (Long Term Memory)
        context_list = list(self.context_buffer)
        context_idx = self.brain.encode_context(context_list)
        target_idx = self.brain.encode_single(target_atom)
        self.brain.write_association(context_idx, target_idx)

    def reset(self):
        self.context_buffer.clear()
