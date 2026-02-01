import random
from config import WorldConfig

class VocalTract:
    """
    [发声器官 - 全集符号原子版]
    约束：
    1. 每次只能输出一个原子符号 (len=1)。
    2. 无温度，无惩罚，仅执行。
    3. 默认游走：不再只是元音，而是包括辅音和IPA的基础发音。
    """
    def __init__(self):
        self.state = "REST"
        self.chance_level = 1.0 / max(len(WorldConfig.VALID_SYMBOLS), 1)
        self.silence_threshold = 0.6 
        
        # 基础游走链 (用于 Babbling)
        # 即使拥有全集符号，婴儿的本能依然倾向于易发音的符号
        self.babble_chain = ['b', 'a', 'm', 'a', 'd', 'a', 'ə', 'i', 'u', 'o']
        self.babble_idx = 0
        
        print(f"[Body] Atomic Articulator Ready. Vocabulary Size: {len(WorldConfig.VALID_SYMBOLS)}")

    def receive_feedback(self, reward_val):
        pass # Stoic Body (心如止水)

    def _get_next_babble(self):
        char = self.babble_chain[self.babble_idx]
        self.babble_idx = (self.babble_idx + 1) % len(self.babble_chain)
        return char

    def articulate(self, intended_char, probability):
        # 原子性检查：如果 intended_char 长度超过1，必须截断
        # 但在我们的系统中，Brain 应该保证只输出单字符
        if len(intended_char) > 1: 
            intended_char = intended_char[0]
            
        raw_output = "a"
        
        if intended_char in [" ", WorldConfig.SYMBOL_SILENCE, WorldConfig.SYMBOL_VOID, "_"]:
            if probability > self.silence_threshold:
                self.state = "INHIBITION"
                self.babble_idx = 0 
                raw_output = "." 
            else:
                self.state = "BABBLE"
                raw_output = self._get_next_babble()
        elif probability > self.chance_level:
            self.state = "ACTIVE"
            raw_output = intended_char
        else:
            self.state = "BABBLE"
            raw_output = self._get_next_babble()
            
        return raw_output

    def rest(self):
        self.state = "REST"
