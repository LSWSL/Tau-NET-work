from typing import List, Dict
from collections import deque
import random
from config import WorldConfig

class SensoryInput:
    def __init__(self, visual_field, inner_field, outer_field, foveal_structure):
        self.visual = visual_field     
        self.inner = inner_field
        self.outer = outer_field
        self.structure = foveal_structure

class VatInterface:
    def __init__(self):
        self.retina_buffer = deque([WorldConfig.SYMBOL_VOID]*WorldConfig.RECEPTIVE_FIELD, maxlen=WorldConfig.RECEPTIVE_FIELD)
        self.inner_buffer = deque([WorldConfig.SYMBOL_SILENCE]*WorldConfig.RECEPTIVE_FIELD, maxlen=WorldConfig.RECEPTIVE_FIELD)
        self.outer_buffer = deque([WorldConfig.SYMBOL_SILENCE]*WorldConfig.RECEPTIVE_FIELD, maxlen=WorldConfig.RECEPTIVE_FIELD)
        
        self.struct_map = {
            WorldConfig.SYMBOL_START: "BOUNDARY_START",
            WorldConfig.SYMBOL_END:   "BOUNDARY_END",
            " ": "SEPARATOR"
        }
        
        self.last_internal_voice = (None, 0.0)
        
        # [原子相位表]
        # 定义每个原子符号在惯性下的"自然流向"
        # Consonant -> Vowel
        # Vowel -> Vowel Drift
        # Math/Punct -> Silence
        self.atomic_phase_map = {
            'b': 'a', 'p': 'a', 'm': 'a', 'd': 'a', 't': 'a',
            'a': 'o', 'o': 'e', 'e': 'i', 'i': 'u', 'u': 'a',
            '.': '.', ' ': '.',
        }

    def receive_motor_feedback(self, content, confidence):
        self.last_internal_voice = (content, confidence)

    def _analyze_structure(self, char):
        if char == WorldConfig.SYMBOL_NOISE: return "NOISE_ARTIFACT"
        if char in self.struct_map: return self.struct_map[char]
        if char in WorldConfig.DIGITS: return "CONTENT_DIGIT"
        if char in WorldConfig.MATHS: return "CONTENT_MATH"
        if char.isalpha(): return "CONTENT_ALPHA"
        return "PUNCT_SYMBOL"

    def _get_next_atomic_phase(self, char):
        """
        [原子相位推演]
        给定当前的一个原子符号，预测下一个最自然的生理原子符号。
        """
        # 默认流向：如果是未知符号，流向静音
        return self.atomic_phase_map.get(char, '.')

    def _construct_phase_locked_stream(self, history_buffer, current_center):
        fovea_idx = WorldConfig.FOVEA_INDEX 
        stream = [WorldConfig.SYMBOL_SILENCE] * WorldConfig.RECEPTIVE_FIELD
        
        # Past (从 Buffer 获取真实历史)
        if len(history_buffer) >= 1: stream[fovea_idx-1] = history_buffer[-1]
        if len(history_buffer) >= 2: stream[fovea_idx-2] = history_buffer[-2]
        
        # Center (强制锁定)
        stream[fovea_idx] = current_center
        
        # Future (基于 Center 的原子推演)
        stream[fovea_idx+1] = self._get_next_atomic_phase(current_center)
        stream[fovea_idx+2] = self._get_next_atomic_phase(stream[fovea_idx+1])
        
        return stream

    def transduce(self, world_events: List[Dict], explicit_guide=None) -> SensoryInput:
        if not world_events: return None
        
        event = world_events[0]
        original_photon = event.get('content', '')
        # 强制截断为单字符 (原子性保证)
        if len(original_photon) > 1: original_photon = original_photon[0]
        
        # Visual
        self.retina_buffer.append(original_photon)
        visual_frame = list(self.retina_buffer)
        foveal_stimulus = visual_frame[WorldConfig.FOVEA_INDEX]
        
        # --- Center Locking ---
        
        # Outer
        center_outer = explicit_guide if explicit_guide else "."
        if len(center_outer) > 1: center_outer = center_outer[0] # 原子性保证
        if center_outer == ' ': center_outer = '.'
        
        # Inner
        self_char, _ = self.last_internal_voice
        center_inner = self_char if self_char else "."
        if len(center_inner) > 1: center_inner = center_inner[0] # 原子性保证
        if center_inner == ' ': center_inner = '.'
        
        # Build Streams
        outer_frame = self._construct_phase_locked_stream(self.outer_buffer, center_outer)
        self.outer_buffer.append(center_outer)
        
        inner_frame = self._construct_phase_locked_stream(self.inner_buffer, center_inner)
        self.inner_buffer.append(center_inner)
        
        struct_info = self._analyze_structure(foveal_stimulus)
        
        return SensoryInput(visual_frame, inner_frame, outer_frame, struct_info)
