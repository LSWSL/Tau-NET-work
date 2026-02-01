import json
import os
from config import WorldConfig

class VocalTract:
    """
    [发声器官 - 高灵敏度版]
    适配 Tau 约束下的微弱神经信号。
    """
    def __init__(self):
        self.config_path = os.path.join(WorldConfig.BIOLOGY_DIR, "phonetics.json")
        self.constraints = self._load_constraints()
        
        self.valid_symbols = set()
        self.vowels = set(self.constraints['categories']['VOWELS'])
        self.consonants = set(self.constraints['categories']['CONSONANTS'])
        
        for cat, symbols in self.constraints['categories'].items():
            for s in symbols:
                self.valid_symbols.add(s)
                
        # 机会水平 (Chance Level)
        # 约 1/30 ≈ 0.033
        self.chance_level = 1.0 / len(self.valid_symbols) if self.valid_symbols else 0.0
        
        self.is_active = False 
        self.last_type = None  
        self.monitor_buffer = []
        
        # [调整] 生物参数
        self.base_cost = 0.0
        # [关键] 大幅降低连击惩罚
        # 因为现在 P(e) 只有 0.12，如果惩罚还是 0.3，任何辅音都发不出来
        # 我们设为 0.02 (2%)，这意味着连续辅音需要比平时多 2% 的信心
        self.cluster_penalty = 0.02 

        print(f"[Body] High-Sensitivity Mode.")
        print(f"       Chance Threshold: {self.chance_level:.4f}")
        print(f"       Cluster Penalty:  {self.cluster_penalty:.4f}")

    def _load_constraints(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _get_type(self, char):
        char = char.lower()
        if char in self.vowels: return 'V'
        if char in self.consonants: return 'C'
        return 'O'

    def articulate(self, intended_char, probability):
        output_stream = ""
        
        current_type = self._get_type(intended_char)
        current_penalty = self.base_cost
        
        if self.last_type == 'C' and current_type == 'C':
            current_penalty += self.cluster_penalty
            
        # 计算有效驱动力
        effective_drive = probability - current_penalty

        # 能量门槛 (Tau Threshold)
        if effective_drive < self.chance_level:
            if self.is_active:
                output_stream += WorldConfig.SYMBOL_END
                self.is_active = False
                self.last_type = None
            self._log(output_stream)
            return output_stream

        if not self.is_active:
            output_stream += WorldConfig.SYMBOL_START
            self.is_active = True
            self.last_type = 'O'

        # 执行
        char_lower = intended_char.lower()
        if char_lower in self.valid_symbols or intended_char in self.valid_symbols:
            output_stream += intended_char
            self.last_type = current_type
        
        self._log(output_stream)
        return output_stream

    def _log(self, content):
        if content:
            self.monitor_buffer.append(content)

    def flush_monitor(self):
        res = "".join(self.monitor_buffer)
        self.monitor_buffer = []
        return res

    def rest(self):
        if self.is_active:
            self._log(WorldConfig.SYMBOL_END)
        self.is_active = False
