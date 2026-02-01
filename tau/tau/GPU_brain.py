import os
import json
import random
import math
from collections import deque, Counter
import torch  # [核心升级] 引入 PyTorch
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
    """
    [海马体 - 快速情景记忆]
    负责暂存短期的具体经历 (Episodes)。
    不需要 GPU 加速，因为主要是列表操作。
    """
    def __init__(self):
        self.episodes = [] 
        # 这里的容量只是为了防止无限增长，具体训练策略由外部控制
        self.capacity = 10000 
        self.stm_associator = {}
        self.stm_capacity = 200

    def check_resonance(self, prediction, reality):
        if not prediction or not reality: return False
        forbidden = [WorldConfig.SYMBOL_VOID, WorldConfig.SYMBOL_START, WorldConfig.SYMBOL_END, WorldConfig.SYMBOL_SILENCE, WorldConfig.SYMBOL_NOISE]
        if prediction in forbidden or reality in forbidden: return False
        return prediction.lower() == reality.lower()

    def store_experience(self, target_idx, context_indices):
        # 仅存储索引整数，非常轻量
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append({"target": target_idx, "context": context_indices})
        
    def replay(self): 
        # 返回当前所有记忆供皮层学习
        return self.episodes
    
    def clear(self): 
        self.episodes = []
        self.stm_associator = {}

# =========================================================
# [核心修改] Cortex 类 - GPU 加速版
# =========================================================
class Cortex:
    """
    [大脑皮层 - 长期语义记忆]
    使用 PyTorch Tensor 替代 Python List。
    实现并行化的赫布学习 (Parallel Hebbian Learning)。
    """
    def __init__(self):
        self.mapper = SymbolMapper()
        self.layers = WorldConfig.LTM_LAYERS
        self.dim = self.mapper.dim_n
        
        # 1. 硬件检测与初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Cortex] Hardware Acceleration: {self.device}")
        
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            print(f"  -> GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
            print(f"  -> Synaptic Matrix: {self.layers} Layers x {self.dim}x{self.dim}")

        # 2. 初始化核心突触矩阵 (The Synaptic Matrix)
        # Shape: [Layers, Pre_Dim, Post_Dim]
        # 这是一个巨大的 3D 张量，始终驻留在显存中。
        # 使用极小的随机数初始化 (打破对称性，防止死锁)
        # requires_grad=False 确保我们不使用反向传播
        self.tensor = torch.rand((self.layers, self.dim, self.dim), device=self.device) * 0.0001
        
        # 历史缓冲区依然在 CPU 上维护 (操作太频繁且数据量小，无需 GPU)
        self.history_buffer = deque([self.mapper.void_idx] * (self.layers + 1), maxlen=(self.layers + 1))
        
        # 尝试加载已有的大脑
        self.bio_path = os.path.join(WorldConfig.BIOLOGY_DIR, "cortex_tensor.pt")
        self._load()

    def _load(self):
        if os.path.exists(self.bio_path):
            try:
                print(f"[Cortex] Loading synaptic weights from {self.bio_path}...")
                self.tensor = torch.load(self.bio_path, map_location=self.device)
            except Exception as e:
                print(f"[Cortex] Load failed: {e}. Starting with fresh brain.")

    def push_history(self, char):
        idx = self.mapper.encode(char)
        self.history_buffer.append(idx)
        return idx

    def get_current_context_indices(self):
        hist = list(self.history_buffer)
        indices = []
        for k in range(self.layers):
            # 取最近的 k+1 个符号作为上下文
            if k + 1 <= len(hist): indices.append(hist[-(k+1)])
            else: indices.append(self.mapper.void_idx)
        return indices

    def predict_distribution_from_context(self, context_indices):
        """
        [GPU 推理]
        并行查询所有层的上下文，聚合预测结果。
        """
        # 1. 预处理：过滤无效上下文
        valid_layers = []
        valid_indices = []
        
        for k, idx in enumerate(context_indices):
            if idx != self.mapper.void_idx and k < self.layers:
                valid_layers.append(k)
                valid_indices.append(idx)
                
        if not valid_layers: return {}

        # 2. 传输到 GPU (这是推理阶段唯一的开销)
        t_layers = torch.tensor(valid_layers, device=self.device, dtype=torch.long)
        t_indices = torch.tensor(valid_indices, device=self.device, dtype=torch.long)

        # 3. 高级索引 (Advanced Indexing) - 核心加速点
        # 直接提取所有相关层的预测向量
        # selected_rows shape: [num_valid_inputs, dim]
        selected_rows = self.tensor[t_layers, t_indices]
        
        # 4. 聚合 (Sum Pooling)
        # 将各层的意见叠加
        activations = torch.sum(selected_rows, dim=0) 
        
        # 5. 计算概率 (简单归一化)
        total = torch.sum(activations)
        if total <= 1e-6: return {}
        
        # 传输回 CPU
        probs = (activations / total).cpu().numpy()
        
        dist = {}
        for i, p in enumerate(probs):
            if p > 1e-4: # 忽略极小概率，减少数据传输和处理
                dist[self.mapper.decode(i)] = float(p)
        return dist

    def learn_offline(self, memories):
        """
        [GPU 学习] 
        实现并行的赫布学习。
        一次性处理成千上万条记忆更新。
        """
        if not memories: return
        
        # --- 阶段 1: CPU 数据组装 ---
        batch_layers = []
        batch_pre = []
        batch_post = []
        
        for mem in memories:
            target_idx = mem['target']
            context_indices = mem['context']
            
            for k, pre_idx in enumerate(context_indices):
                # 过滤无效连接
                if k < self.layers and pre_idx != self.mapper.void_idx:
                    batch_layers.append(k)
                    batch_pre.append(pre_idx)
                    batch_post.append(target_idx)
        
        if not batch_layers: return

        # --- 阶段 2: 传输到 GPU ---
        t_layers = torch.tensor(batch_layers, device=self.device, dtype=torch.long)
        t_pre = torch.tensor(batch_pre, device=self.device, dtype=torch.long)
        t_post = torch.tensor(batch_post, device=self.device, dtype=torch.long)
        
        # --- 阶段 3: 并行计数 (Parallel Counting) ---
        # 将 3D 坐标扁平化为 1D 索引
        dim = self.dim
        flat_indices = t_layers * (dim * dim) + t_pre * dim + t_post
        
        # 统计每条突触需要强化的次数
        unique_indices, counts = torch.unique(flat_indices, return_counts=True)
        
        # --- 阶段 4: 赫布更新 (Hebbian Update) ---
        # 计算增量：次数 * 学习率
        update_values = counts.float() * WorldConfig.NEUROPLASTICITY_RATE
        
        # 原位更新大张量
        flat_tensor = self.tensor.view(-1)
        flat_tensor.index_add_(0, unique_indices, update_values)
        
        # (可选) 权重衰减/归一化可以在这里添加，防止数值爆炸
        # 但对于短时实验，纯累加通常没问题

    def save(self):
        # 使用 torch.save 保存张量，极快
        torch.save(self.tensor, self.bio_path)
        # print("[Cortex] Brain state saved.")

class WorkingMemorySystem:
    def __init__(self):
        self.time_depth = WorldConfig.MEMORY_CAPACITY
        self.num_channels = WorldConfig.WM_CHANNELS
        self.space_width = WorldConfig.RECEPTIVE_FIELD
        # 这里的张量只是个名字，实际上还是 List，因为是在 CPU 上做简单的逻辑处理
        self.content_tensor = [[[WorldConfig.SYMBOL_VOID]*self.space_width for _ in range(self.num_channels)] for _ in range(self.time_depth)]
        
    def update(self, visual_frame, auditory_stimulus=None):
        # 简单队列逻辑
        self.content_tensor.pop()
        
        vis_c = list(visual_frame)
        aud_c = [WorldConfig.SYMBOL_VOID] * self.space_width
        
        # 如果有听觉输入，放在中央凹位置
        if auditory_stimulus:
             aud_c[WorldConfig.FOVEA_INDEX] = auditory_stimulus
             
        self.content_tensor.insert(0, [vis_c, aud_c])

class VatBrain:
    def __init__(self):
        self.age = 0
        self.state = "AWAKE"
        self.sleep_timer = 0
        
        # 初始化组件
        self.cortex = Cortex() # 这里的 Cortex 已经是 GPU 版的了
        self.hippocampus = Hippocampus()
        self.wm = WorkingMemorySystem() 
        self.body = VocalTract()
        
        self.inhibited_symbols = {
            WorldConfig.SYMBOL_START, WorldConfig.SYMBOL_END, 
            WorldConfig.SYMBOL_VOID, WorldConfig.SYMBOL_NOISE,
            WorldConfig.SYMBOL_SILENCE
        }
        self.current_motor_output = (None, 0.0)
        self.silence_streak = 0
        
        print(f"[Subject] Brain initialized. Passive Mode logic enabled.")

    def perceive(self, signal: SensoryInput):
        if self.state == "ASLEEP":
            self.process_sleep()
            return None, 0.0

        self.age += 1
        if not signal: return None, 0.0

        if hasattr(signal, 'structure') and signal.structure == "BOUNDARY_START":
            self.silence_streak = 0

        # 1. 接收感知
        self.wm.update(signal.visual, getattr(signal, 'auditory', None))
        context_indices = self.cortex.get_current_context_indices()
        
        # 2. 沉默判定 (Passive Listening)
        # 模仿婴儿：听不懂或预测差时，闭嘴多听
        best_char = None
        best_prob = 0.0
        
        if self.silence_streak < 7:
            # 只有在非静默模式下才进行预测 (节省算力)
            dist = self.cortex.predict_distribution_from_context(context_indices)
            
            # 简单贪婪采样 (Greedy Sampling)
            if dist:
                best_char = max(dist, key=dist.get)
                best_prob = dist[best_char]
        
        # 3. 刻印 (Imprinting) - 永远进行
        # 这是学习的关键：无论我说不说，我都要记住我看到了什么
        fovea_char = self.wm.content_tensor[0][0][WorldConfig.FOVEA_INDEX]
        fovea_struct = getattr(signal, 'structure', 'CONTENT_ALPHA')
        
        # 过滤掉不需要学习的噪声
        if fovea_struct not in ["BOUNDARY_START", "BOUNDARY_END"]:
            target_idx = self.cortex.mapper.encode(fovea_char)
            self.hippocampus.store_experience(target_idx, context_indices)
            self.cortex.push_history(fovea_char)

        # 4. 发声 (Articulation)
        motor_char = None
        motor_conf = 0.0
        
        if best_char and best_char not in self.inhibited_symbols:
            output_str = self.body.articulate(best_char, best_prob)
            
            if output_str:
                self.current_motor_output = (output_str, best_prob)
                self.silence_streak = 0 
                
                # 检查共鸣 (Resonance)
                if self.hippocampus.check_resonance(best_char, fovea_char):
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
        self.silence_streak = 0 

    def process_sleep(self):
        if self.sleep_timer > 0:
            self.sleep_timer -= 1
            # 在睡梦的最后一刻，进行深度学习
            if self.sleep_timer == 0:
                memories = self.hippocampus.replay()
                if memories:
                    # print(f"[Brain] REM Sleep: Consolidating {len(memories)} episodes...")
                    self.cortex.learn_offline(memories)
                    self.cortex.save()
            self.body.rest()
            self.current_motor_output = (None, 0.0)
        else:
            self.hippocampus.clear()
            self.state = "AWAKE"

    def save(self):
        self.cortex.save()