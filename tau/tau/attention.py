from config import WorldConfig
from collections import deque

class AttentionSystem:
    def __init__(self, brain_ref):
        self.brain = brain_ref
        # [适配N阶] 使用双端队列作为滑动窗口
        self.window_size = WorldConfig.N_ORDER
        self.context_buffer = deque(maxlen=self.window_size)

    def predict(self, current_atom):
        """
        Input: 当前看到的原子
        Output: 预测下一个原子 (及置信度)
        """
        # 1. 更新上下文窗口 (将当前原子推入)
        # 注意：Attention 的预测逻辑是基于[当前及之前的历史]来预测[未来]
        self.context_buffer.append(current_atom)
        
        # 2. 如果窗口未满，无法形成 N-gram，暂时无法预测
        if len(self.context_buffer) < self.window_size:
            return None, 0.0
            
        # 3. 打包上下文
        context_list = list(self.context_buffer)
        
        # 4. 编码 (调用 Brain 的新接口，接受 list)
        context_idx = self.brain.encode_context(context_list)
        
        # 5. 查询大脑
        pred_idx, prob = self.brain.query_distribution(context_idx)
        
        if pred_idx is not None:
            return self.brain.decode(pred_idx), prob
        
        return None, 0.0
    
    def reset(self):
        """清空上下文 (用于切换文章或测试用例时)"""
        self.context_buffer.clear()
