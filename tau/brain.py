import os
import json
from collections import deque, Counter
import random
import math
from interface import SensoryInput
from config import WorldConfig
from body import VocalTract

class SymbolMapper:
    def __init__(self):
        self.chars = WorldConfig.VALID_SYMBOLS
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.dim_n = len(self.chars)
        self.void_idx = self.char_to_idx.get(WorldConfig.SYMBOL_VOID, 0)
    def encode(self, char): return self.char_to_idx.get(char, self.void_idx)
    def decode(self, idx): return self.idx_to_char.get(idx, WorldConfig.SYMBOL_VOID)

class Hippocampus:
    def __init__(self):
        self.episodes = [] 
        self.capacity = 5000 
        self.stm_associator = {}
        self.stm_capacity = 200

    def check_resonance(self, prediction, reality):
        if not prediction or not reality: return False
        forbidden = [WorldConfig.SYMBOL_VOID, WorldConfig.SYMBOL_START, WorldConfig.SYMBOL_END, WorldConfig.SYMBOL_SILENCE, WorldConfig.SYMBOL_NOISE]
        if prediction in forbidden or reality in forbidden: return False
        return prediction.lower() == reality.lower()

    def store_experience(self, target_idx, context_indices):
        if len(self.episodes) >= self.capacity: self.episodes.pop(0)
        self.episodes.append({"target": target_idx, "context": context_indices})
        
        short_ctx = tuple(context_indices[:3]) 
        if short_ctx not in self.stm_associator:
            self.stm_associator[short_ctx] = Counter()
            if len(self.stm_associator) > self.stm_capacity:
                self.stm_associator.pop(next(iter(self.stm_associator)))
        self.stm_associator[short_ctx][target_idx] += 1

    def query_stm(self, context_indices):
        short_ctx = tuple(context_indices[:3])
        if short_ctx in self.stm_associator:
            counter = self.stm_associator[short_ctx]
            total = sum(counter.values())
            return {k: v/total for k, v in counter.items()}
        return {}
    
    def replay(self): return self.episodes
    def clear(self): 
        self.episodes = []
        self.stm_associator = {}

class Cortex:
    def __init__(self):
        self.mapper = SymbolMapper()
        self.layers = WorldConfig.LTM_LAYERS
        self.dim = self.mapper.dim_n
        init_w = WorldConfig.INITIAL_SYNAPTIC_WEIGHT
        self.tensor = [
            [[init_w] * self.dim for _ in range(self.dim)]
            for _ in range(self.layers)
        ]
        self.bio_path = os.path.join(WorldConfig.BIOLOGY_DIR, "cortex_tensor.json")
        self._load()
        self.history_buffer = deque([self.mapper.void_idx] * (self.layers + 1), maxlen=(self.layers + 1))

    def _load(self):
        if os.path.exists(self.bio_path):
            try:
                with open(self.bio_path, 'r') as f:
                    data = json.load(f)
                    if len(data) == self.layers: self.tensor = data
            except: pass

    def push_history(self, char):
        idx = self.mapper.encode(char)
        self.history_buffer.append(idx)
        return idx

    def get_current_context_indices(self):
        hist = list(self.history_buffer)
        indices = []
        for k in range(self.layers):
            if k + 1 <= len(hist): indices.append(hist[-(k+1)])
            else: indices.append(self.mapper.void_idx)
        return indices

    def predict_distribution_from_context(self, context_indices):
        activations = [0.0] * self.dim
        for k, idx in enumerate(context_indices):
            if idx == self.mapper.void_idx: continue
            row = self.tensor[k][idx]
            temporal_decay = 1.0 / (k + 1)
            for t_idx, w in enumerate(row):
                activations[t_idx] += w * temporal_decay
        
        forbidden = [WorldConfig.SYMBOL_START, WorldConfig.SYMBOL_END, WorldConfig.SYMBOL_VOID, WorldConfig.SYMBOL_NOISE, WorldConfig.SYMBOL_SILENCE]
        for char in forbidden:
            idx = self.mapper.encode(char)
            activations[idx] = 0.0

        total = sum(activations)
        if total <= 0: return {}
        dist = {}
        for idx, s in enumerate(activations):
            if s > 0: dist[self.mapper.decode(idx)] = s / total
        return dist

    def _aggregate_hippocampus_memory(self, memories):
        agg = Counter()
        for mem in memories:
            target_idx = mem['target']
            context_indices = mem['context']
            for k in range(WorldConfig.HIPPOCAMPUS_DEPTH):
                if k >= len(context_indices): break
                past_idx = context_indices[k]
                if past_idx == self.mapper.void_idx: continue
                agg[(k, past_idx, target_idx)] += 1
        total = sum(agg.values())
        if total == 0: return {}
        scale_factor = WorldConfig.NEUROPLASTICITY_RATE
        return {k: v * scale_factor for k, v in agg.items()}

    def learn_offline(self, memories):
        suggestions = self._aggregate_hippocampus_memory(memories)
        if not suggestions: return 0
        suggestion_keys = list(suggestions.keys())
        updates_performed = 0
        
        for _ in range(WorldConfig.SLEEP_REPLAY_CYCLES):
            key = random.choice(suggestion_keys)
            layer_k, pre_idx, post_idx = key
            hippo_strength = suggestions[key]
            current_strength = self.tensor[layer_k][pre_idx][post_idx]
            
            if hippo_strength > current_strength:
                target_new_weight = current_strength + (hippo_strength * 0.1) 
                if target_new_weight > 0.99: target_new_weight = 0.99
                self.tensor[layer_k][pre_idx][post_idx] = target_new_weight
                updates_performed += 1
                
                remaining_budget = 1.0 - target_new_weight
                sum_others = 0.0
                for l in range(self.layers):
                    for p in range(self.dim):
                        if l == layer_k and p == pre_idx: continue
                        sum_others += self.tensor[l][p][post_idx]
                
                if sum_others > 0:
                    scale_factor = remaining_budget / sum_others
                    for l in range(self.layers):
                        for p in range(self.dim):
                            if l == layer_k and p == pre_idx: continue
                            self.tensor[l][p][post_idx] *= scale_factor
        return updates_performed

    def save(self):
        try:
            with open(self.bio_path, 'w', encoding='utf-8') as f: json.dump(self.tensor, f)
        except: pass

class WorkingMemorySystem:
    def __init__(self):
        self.time_depth = WorldConfig.MEMORY_CAPACITY
        self.num_channels = WorldConfig.WM_CHANNELS
        self.space_width = WorldConfig.RECEPTIVE_FIELD
        self.content_tensor = [[[WorldConfig.SYMBOL_VOID]*self.space_width for _ in range(self.num_channels)] for _ in range(self.time_depth)]
        self.strength_tensor = [[[0.0]*self.space_width for _ in range(self.num_channels)] for _ in range(self.time_depth)]
        self.auditory_buffer = deque([WorldConfig.SYMBOL_VOID]*self.space_width, maxlen=self.space_width)
        self.attention_mask = WorldConfig.VISUAL_ATTENTION_MASK

    def update(self, visual_frame, auditory_stimulus):
        decay = WorldConfig.MEMORY_DECAY_FACTOR
        thresh = WorldConfig.MEMORY_FORGET_THRESHOLD
        for t in range(self.time_depth):
            for c in range(self.num_channels):
                for s in range(self.space_width):
                    curr = self.strength_tensor[t][c][s]
                    if curr > 0:
                        self.strength_tensor[t][c][s] = curr * decay if (curr * decay) > thresh else 0.0
                        if self.strength_tensor[t][c][s] == 0.0: self.content_tensor[t][c][s] = WorldConfig.SYMBOL_VOID

        self.content_tensor.pop(); self.strength_tensor.pop()
        vis_c = list(visual_frame); vis_s = list(self.attention_mask)
        self.auditory_buffer.append(auditory_stimulus)
        buff = list(self.auditory_buffer)
        aud_c = [WorldConfig.SYMBOL_VOID]*self.space_width; aud_s = [0.0]*self.space_width
        aud_c[2] = buff[-1]; aud_s[2] = 0.8 if buff[-1]!=WorldConfig.SYMBOL_VOID else 0.0
        aud_c[1] = buff[-2]; aud_s[1] = 0.5 if buff[-2]!=WorldConfig.SYMBOL_VOID else 0.0
        aud_c[0] = buff[-3]; aud_s[0] = 0.2 if buff[-3]!=WorldConfig.SYMBOL_VOID else 0.0
        self.content_tensor.insert(0, [vis_c, aud_c])
        self.strength_tensor.insert(0, [vis_s, aud_s])

class VatBrain:
    def __init__(self):
        self.age = 0
        self.state = "AWAKE"
        self.sleep_timer = 0
        self.cortex = Cortex()
        self.hippocampus = Hippocampus()
        self.wm = WorkingMemorySystem() 
        self.body = VocalTract()
        self.content_types = {"CONTENT_ALPHA", "CONTENT_UPPER", "CONTENT_LOWER", "CONTENT_DIGIT", "NOISE_ARTIFACT"}
        self.current_motor_output = (None, 0.0)
        self.current_attention_focus = (None, 0.0)
        self.experiences_today = 0
        self.last_entropy = 0.0
        
        self.inhibited_symbols = {
            WorldConfig.SYMBOL_START, WorldConfig.SYMBOL_END, 
            WorldConfig.SYMBOL_VOID, WorldConfig.SYMBOL_NOISE,
            WorldConfig.SYMBOL_SILENCE
        }
        
        # [新增] 沉默计数器
        self.silence_streak = 0
        
        print(f"[Subject] Passive Mode Enabled: Auto-silence after 7 ticks.")

    def _calculate_entropy(self, distribution):
        if not distribution: return 0.0
        entropy = 0.0
        for p in distribution.values():
            if p > 0:
                entropy -= p * math.log(p, 2)
        return entropy

    def _sample_with_temperature(self, distribution, temperature=0.5):
        if not distribution: return None, 0.0
        chars = list(distribution.keys())
        probs = list(distribution.values())
        logits = []
        for p in probs:
            if p <= 0: logits.append(-100.0)
            else: logits.append(math.log(p) / temperature)
        exp_logits = [math.exp(x) for x in logits]
        sum_exp = sum(exp_logits)
        if sum_exp == 0: return None, 0.0
        new_probs = [x / sum_exp for x in exp_logits]
        chosen_char = random.choices(chars, weights=new_probs, k=1)[0]
        original_prob = distribution[chosen_char]
        return chosen_char, original_prob

    def perceive(self, signal: SensoryInput):
        if self.state == "ASLEEP":
            self.process_sleep()
            return

        self.age += 1
        if not signal: return

        # 检查是否因为新的开始而重置
        if signal.structure == "BOUNDARY_START":
            self.silence_streak = 0

        self.wm.update(signal.visual, signal.auditory)
        context_indices = self.cortex.get_current_context_indices()
        
        # [逻辑变更] 沉默判定
        # 如果连续沉默超过 7 次，进入"只听不说"模式 (Passive Listening)
        if self.silence_streak >= 7:
            best_char = None
            best_prob = 0.0
            # 注意：我们依然继续下面的海马体记录逻辑 (Imprinting)，因为还在学习
            # 只是跳过了预测分布计算 (省算力) 和 articulate
        else:
            # 正常预测流程
            dist_ltm = self.cortex.predict_distribution_from_context(context_indices)
            dist_stm_indices = self.hippocampus.query_stm(context_indices)
            dist_stm = {}
            for idx, score in dist_stm_indices.items():
                dist_stm[self.cortex.mapper.decode(idx)] = score
                
            alpha_stm = 0.8
            final_dist = {}
            all_chars = set(dist_ltm.keys()) | set(dist_stm.keys())
            for char in all_chars:
                p_ltm = dist_ltm.get(char, 0.0)
                p_stm = dist_stm.get(char, 0.0)
                final_dist[char] = (p_ltm * (1 - alpha_stm)) + (p_stm * alpha_stm)
            
            for forbidden in [WorldConfig.SYMBOL_START, WorldConfig.SYMBOL_END, WorldConfig.SYMBOL_VOID]:
                if forbidden in final_dist:
                    del final_dist[forbidden]

            self.last_entropy = self._calculate_entropy(final_dist)
            best_char, best_prob = self._sample_with_temperature(final_dist, temperature=0.5)
        
        self.current_attention_focus = (best_char, best_prob)
        
        # Imprinting (始终进行，除非在被动模式下为了极致优化也可以跳过? 
        # 不，"听"是学习的关键，必须保留)
        fovea_char = self.wm.content_tensor[0][0][WorldConfig.FOVEA_INDEX]
        fovea_struct = signal.structure
        is_resonant = self.hippocampus.check_resonance(best_char, fovea_char)
        
        if fovea_struct in self.content_types:
            target_idx = self.cortex.mapper.encode(fovea_char)
            self.hippocampus.store_experience(target_idx, context_indices)
            self.cortex.push_history(fovea_char)
            self.experiences_today += 1
        else:
            pass 

        motor_char = None
        motor_conf = 0.0
        
        if best_char:
            if best_char in self.inhibited_symbols:
                output_str = None
            else:
                output_str = self.body.articulate(best_char, best_prob)
            
            if output_str:
                self.current_motor_output = (output_str, best_prob)
                self.silence_streak = 0 # 成功说话，重置沉默计数
                if is_resonant:
                    motor_char = output_str
                    motor_conf = best_prob
            else:
                self.current_motor_output = (None, 0.0)
                self.silence_streak += 1
        else:
            self.body.articulate("", 0.0)
            self.current_motor_output = (None, 0.0)
            self.silence_streak += 1
            
        return motor_char, motor_conf

    def trigger_sleep(self):
        self.state = "ASLEEP"
        self.sleep_timer = WorldConfig.SLEEP_DURATION
        self.experiences_today = 0
        self.silence_streak = 0 # 睡醒重置

    def process_sleep(self):
        if self.sleep_timer > 0:
            self.sleep_timer -= 1
            if self.sleep_timer == WorldConfig.SLEEP_DURATION - 1:
                memories = self.hippocampus.replay()
                brain_updates = self.cortex.learn_offline(memories)
                self.cortex.save()
            self.body.rest()
            self.current_motor_output = (None, 0.0)
        else:
            self.hippocampus.clear()
            self.state = "AWAKE"

    def save(self):
        self.cortex.save()
