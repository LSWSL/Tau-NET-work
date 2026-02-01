import torch
import os
import gc
from config import WorldConfig

class SimpleBrain:
    def __init__(self):
        # N-gram 稀疏哈希配置
        self.io_dim = WorldConfig.MEMORY_SIZE 
        self.time_dim = 2000 # 每个上下文槽位能记录多少种不同的后续 (Target)
        
        self.device = torch.device("cpu")
        self.VOID_VAL = 0 # 0 表示空
        
        self.ltm_file = os.path.join(WorldConfig.OUTPUT_DIR, f"ltm_scinote_{WorldConfig.N_ORDER}gram.pt")
        
        self.atoms = WorldConfig.VALID_SYMBOLS
        self.atom_to_idx = {c: i for i, c in enumerate(self.atoms)}
        self.idx_to_atom = {i: c for i, c in enumerate(self.atoms)}
        self.vocab_size = len(self.atoms)

        self._init_architecture()

    def _init_architecture(self):
        # 稀疏矩阵: [Context_Hash, Slot_Index]
        shape = (self.io_dim, self.time_dim)
        gc.collect()
        
        if os.path.exists(self.ltm_file):
            try:
                state = torch.load(self.ltm_file, map_location=self.device)
                print(f"[Brain] Loading Scientific Memory...")
                self.ltm_mantissa = state['mantissa']
                self.ltm_exponent = state['exponent']
                self.ltm_targets  = state['targets'] # 记录 target 是谁
                self.pointers     = state['pointers']
            except:
                self._create_genesis(shape)
        else:
            self._create_genesis(shape)

    def _create_genesis(self, shape):
        print(f"[Brain] Genesis: Allocating Dual-Channel Memory (Mantissa + Exponent)...")
        # 通道 1: 尾数 (精度)
        self.ltm_mantissa = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        # 通道 2: 指数 (量级)
        self.ltm_exponent = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        # 通道 3: 目标索引 (内容)
        self.ltm_targets  = torch.full(shape, 255, dtype=torch.uint8, device=self.device)
        # 指针
        self.pointers = torch.zeros(shape[0], dtype=torch.long, device=self.device)

    def save(self):
        state = {
            'mantissa': self.ltm_mantissa,
            'exponent': self.ltm_exponent,
            'targets':  self.ltm_targets,
            'pointers': self.pointers
        }
        torch.save(state, self.ltm_file)

    def encode_single(self, char):
        return self.atom_to_idx.get(char, 0)

    def encode_context(self, atom_list):
        # 多项式哈希
        h = 0
        p = 31
        m = self.io_dim
        for char in atom_list:
            idx = self.atom_to_idx.get(char, 0) + 1
            h = (h * p + idx) % m
        return h

    def decode(self, idx):
        if idx == 255: return "?"
        return self.idx_to_atom.get(idx, WorldConfig.SYMBOL_VOID)

    def write_association(self, context_idx, target_idx):
        """
        [科学计数法写入逻辑]
        """
        # 1. 检查该 Context 下，是否已经记录过这个 Target ?
        # 这是一个线性查找，但在稀疏矩阵中通常很快 (pointers[context_idx] 很小)
        row_len = self.pointers[context_idx].item()
        
        # 优化：只在有效范围内查找
        # 获取该行的 targets
        targets = self.ltm_targets[context_idx, :row_len]
        
        # 查找 target_idx 是否存在
        # (Tensor操作)
        matches = (targets == target_idx).nonzero(as_tuple=True)[0]
        
        if len(matches) > 0:
            # A. 已经存在 -> 增强记忆 (Increment)
            col_idx = matches[0].item()
            self._increment_cell(context_idx, col_idx)
        else:
            # B. 不存在 -> 新建记忆 (Append)
            if row_len < self.time_dim:
                self.ltm_targets[context_idx, row_len] = target_idx
                # 初始化为 M=1, E=0 (Value = 1)
                self.ltm_mantissa[context_idx, row_len] = 1
                self.ltm_exponent[context_idx, row_len] = 0
                self.pointers[context_idx] += 1

    def _increment_cell(self, r, c):
        """
        [核心算法] 科学计数加法
        Val = M * 2^E
        Goal: Val += 1
        """
        m = self.ltm_mantissa[r, c].item()
        e = self.ltm_exponent[r, c].item()
        
        # 策略：只有当增加量相对于当前量级有意义时，才更新
        # 简单实现：
        # 如果 M 未满 ( < 255 )，直接 M++
        # 如果 M 已满 ( = 255 )，进行重归一化 (Renormalize)
        
        if m < 255:
            self.ltm_mantissa[r, c] += 1
        else:
            # 溢出！量级提升！
            # 逻辑：255 * 2^E  ->  128 * 2^(E+1)
            # 也就是 M 除以 2，E 加 1
            new_m = m // 2
            new_e = e + 1
            if new_e < 255: # 防止指数溢出
                self.ltm_mantissa[r, c] = new_m
                self.ltm_exponent[r, c] = new_e
                # 此时我们已经腾出了空间，可以再 +1 (可选)
                # 但为了纯粹的量级跃迁，保持这样即可

    def query_distribution(self, context_idx):
        """
        [科学计数法读取]
        Score = Mantissa * 2^Exponent
        """
        row_len = self.pointers[context_idx].item()
        if row_len == 0:
            return None, 0.0
            
        # 提取有效数据
        valid_targets = self.ltm_targets[context_idx, :row_len].long()
        valid_m = self.ltm_mantissa[context_idx, :row_len].float()
        valid_e = self.ltm_exponent[context_idx, :row_len].float()
        
        # 1. 还原真实强度 (Reconstruct Magnitude)
        # Score = M * (2 ** E)
        scores = valid_m * torch.pow(2.0, valid_e)
        
        # 2. 归一化为概率
        total_score = scores.sum()
        if total_score == 0: return None, 0.0
        
        probs = scores / total_score
        
        # 3. 找最大
        max_idx = torch.argmax(probs).item()
        final_target = valid_targets[max_idx].item()
        final_prob = probs[max_idx].item()
        
        return final_target, final_prob
