import sys
import time
import torch
from config import WorldConfig
from brain import SimpleBrain
from world import ExternalWorld
from hippocampus import Hippocampus
from attention import AttentionSystem

def evaluate_surprise(text, attention, brain):
    """
    计算一段文本的平均惊奇度。
    """
    attention.reset()
    total_surprise = 0.0
    valid_predictions = 0
    
    # 预热：先填满窗口，因为 N-gram 需要 N 个字才能开始预测
    burn_in = WorldConfig.N_ORDER
    
    for i, char in enumerate(text):
        # 严格过滤：测试时也要遵守白名单，否则测试本身就不公平
        if char not in WorldConfig.VALID_SYMBOLS: continue
        
        # 1. 如果上下文不够长，只积累，不打分
        if len(attention.context_buffer) < WorldConfig.N_ORDER:
            attention.context_buffer.append(char)
            continue
            
        # 2. 预测阶段 (Peek)
        context_list = list(attention.context_buffer)
        context_idx = brain.encode_context(context_list)
        
        # 查询大脑最期待的下一个字
        pred_char_idx, prob = brain.query_distribution(context_idx)
        pred_char = brain.decode(pred_char_idx) if pred_char_idx is not None else None
        
        # 3. 验证阶段 (Compare)
        step_surprise = 1.0 # 默认完全惊讶
        
        if pred_char == char:
            # 预测对了：惊奇度 = 1 - 置信度
            # (非常自信且对了 -> 惊奇度 0%)
            # (不太自信但蒙对了 -> 惊奇度 60%)
            step_surprise = 1.0 - prob
        else:
            # 预测错了：完全惊讶 (100%)
            step_surprise = 1.0
            
        total_surprise += step_surprise
        valid_predictions += 1
        
        # 4. 更新阶段 (Update)
        attention.context_buffer.append(char)
        
    if valid_predictions == 0: return 0.0
    return (total_surprise / valid_predictions) * 100

def run_clean_anomaly_detector():
    print(f">>> SECURITY SYSTEM: CLEAN FILTERED MODE")
    print(f">>> DATA: Strict Allowlist (Only Valid Symbols)")
    
    try:
        world = ExternalWorld()
        brain = SimpleBrain()           
        hippo = Hippocampus(brain)
        attention = AttentionSystem(brain)
    except Exception as e:
        print(f"[FATAL] Init Failed: {e}")
        sys.exit(1)

    # --- 阶段一：纯净训练 ---
    # 训练 100,000 步
    TRAIN_STEPS = 100000
    print(f"\n[PHASE 1] Learning from Clean Data ({TRAIN_STEPS} chars)...")
    
    sys.stdout.write("[")
    for tick in range(TRAIN_STEPS):
        input_atom, target_atom = world.manifest()
        
        # 只有有效的原子才会触发学习
        if input_atom != WorldConfig.SYMBOL_VOID:
            hippo.consolidate(input_atom, target_atom)
        
        if tick % (TRAIN_STEPS // 50) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
    print("] Done.\n")
    
    # --- 阶段二：异常检测 ---
    print("[PHASE 2] Detecting Anomalies")
    print(f"{'TYPE':<15} | {'TEXT SAMPLE':<30} | {'ANOMALY SCORE'}")
    print("-" * 75)
    
    test_cases = [
        # 1. 训练数据中大量存在的 (预期: 极低惊奇度)
        ("NORMAL", "alice was beginning to get"),
        ("NORMAL", "down the rabbit-hole"),
        
        # 2. 符合英语语法，但可能未见过 (预期: 中等惊奇度，或者是正常的)
        ("ENG-STRUCT", "the cat sat on the mat"),
        
        # 3. 注入攻击 (预期: 高惊奇度)
        ("SQL-INJECT", "SELECT * FROM users WHERE"),
        
        # 4. 代码结构 (预期: 高惊奇度)
        ("CODE", "function main() { return 0; }"),
        
        # 5. 纯乱码 (预期: 极高惊奇度)
        ("NOISE", "xkq zjw qqz 883 a#$"),
    ]
    
    for label, text in test_cases:
        score = evaluate_surprise(text, attention, brain)
        
        bar_len = int(score / 5)
        bar = "█" * bar_len
        
        # 严格的判定标准
        if score > 80:   status = "🚨 CRITICAL"
        elif score > 50: status = "⚠️ SUSPICIOUS"
        else:            status = "✅ NORMAL"
        
        print(f"{label:<15} | {text[:30]:<30} | {score:5.1f}% {bar} {status}")

if __name__ == "__main__":
    run_clean_anomaly_detector()
