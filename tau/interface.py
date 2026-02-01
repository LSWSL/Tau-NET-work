from typing import List, Dict
from collections import deque
import random
from config import WorldConfig

class SensoryInput:
    def __init__(self, visual_field: List[str], foveal_structure: str, auditory_input: str):
        self.visual = visual_field 
        self.structure = foveal_structure
        self.auditory = auditory_input

class VatInterface:
    """
    [接口层 - 父母混音版]
    Auditory Channel = Mixer(Internal_Voice, Parental_Voice)
    """
    def __init__(self):
        self.retina_buffer = deque(
            [WorldConfig.SYMBOL_VOID] * WorldConfig.RECEPTIVE_FIELD, 
            maxlen=WorldConfig.RECEPTIVE_FIELD
        )
        self.struct_map = {
            WorldConfig.SYMBOL_START: "BOUNDARY_START",
            WorldConfig.SYMBOL_END:   "BOUNDARY_END",
            " ": "SEPARATOR"
        }
        
        self.last_internal_voice = (None, 0.0)
        self.chance_level = 0.033 

    def receive_motor_feedback(self, content, confidence):
        self.last_internal_voice = (content, confidence)

    def _analyze_structure(self, char):
        if char == WorldConfig.SYMBOL_NOISE: return "NOISE_ARTIFACT"
        if char in self.struct_map: return self.struct_map[char]
        if char.isdigit(): return "CONTENT_DIGIT"
        if char.isupper(): return "CONTENT_UPPER"
        if char.islower(): return "CONTENT_LOWER"
        if char in ['.', '!', '?', '。', '！', '？']: return "PUNCT_STOP"
        return "PUNCT_SYMBOL"

    def _mix_auditory_channels(self, ground_truth_char):
        """
        [混音逻辑]
        Self vs Parent
        """
        # 1. 获取内在语
        self_char, self_conf = self.last_internal_voice
        if not self_char: self_conf = 0.0
        
        # 2. 模拟父母语 (Parental Voice)
        # 父母看到什么读什么 (Ground Truth)
        # 并不是一直读，有时会停顿
        parent_char = WorldConfig.SYMBOL_VOID
        parent_conf = 0.0
        
        if random.random() < WorldConfig.PARENTAL_INTERVENTION_RATE:
            parent_char = ground_truth_char
            parent_conf = WorldConfig.PARENTAL_VOICE_CONFIDENCE
            
        # 3. 竞争与混合 (Shared Channel)
        # 如果自己在说话 (Conf > Chance)，则听不见父母 (Self-Dominance)
        # 或者是简单的 Max() 逻辑？
        # 这里的逻辑是：内在语优先级更高（因为它在头骨内部），会物理掩盖外部声音
        
        if self_conf > self.chance_level:
            return self_char # 听到自己
        elif parent_conf > 0:
            return parent_char # 听到父母
        else:
            return WorldConfig.SYMBOL_SILENCE # 啥也没听到

    def transduce(self, world_events: List[Dict]) -> SensoryInput:
        if not world_events: return None
        
        event = world_events[0]
        original_photon = event.get('content', '')
        
        # Visual
        self.retina_buffer.append(original_photon)
        visual_frame = list(self.retina_buffer)
        foveal_stimulus = visual_frame[WorldConfig.FOVEA_INDEX]
        struct_info = self._analyze_structure(foveal_stimulus)
        
        # Auditory (Mixing)
        # 传入 foveal_stimulus 作为父母朗读的参考
        auditory_signal = self._mix_auditory_channels(foveal_stimulus)
        
        return SensoryInput(visual_frame, struct_info, auditory_signal)
