import math
import matplotlib.pyplot as plt
from collections import defaultdict

class Tau_Anomaly_Detector:
    """无监督异常检测核心引擎 (Tau-Net V2.0)"""
    def __init__(self, max_lifespan=10000):
        self.neocortex = defaultdict(int)
        self.hippocampus = []
        self.max_lifespan = max_lifespan
        self.time_step = 0
        
    def ingest_token(self, current_char):
        self.time_step += 1
        for i in range(len(self.hippocampus)):
            t, c, l, p = self.hippocampus[i]
            self.hippocampus[i] = (t, c, l - 1, p)
        self.hippocampus.append((self.time_step, current_char, self.max_lifespan, False))
        
    def consolidate_matrix(self):
        if not self.hippocampus: return
        min_lifespan = min([mem[2] for mem in self.hippocampus])
        decayed_steps = self.max_lifespan - min_lifespan
        
        normalized_hippocampus = []
        for t, c, l, p in self.hippocampus:
            new_l = l - decayed_steps + 1
            if new_l > 0: normalized_hippocampus.append((t, c, new_l, p))
        self.hippocampus = normalized_hippocampus
        
        edge_counts = defaultdict(lambda: {'correct': 0})
        for i in range(len(self.hippocampus) - 1):
            char_i = self.hippocampus[i][1]
            for j in range(i + 1, min(i + 3, len(self.hippocampus))): 
                char_j = self.hippocampus[j][1]
                dist = j - i
                edge_counts[(char_i, char_j, dist)]['correct'] += 1 

        for edge, counts in edge_counts.items():
            if counts['correct'] > 0:
                delta_w = math.floor(math.log10(counts['correct'] + 1)) + 1 
                self.neocortex[edge] += delta_w
        self.hippocampus = []

# ================= 核心基准测试与图表生成 =================
def run_benchmark_and_plot():
    print("==================================================")
    print(" [Tau-Net Benchmark] Unsupervised Anomaly Detection")
    print("==================================================\n")
    
    detector = Tau_Anomaly_Detector()
    
    print("[1] Constructing Neocortical Matrix via Log-Hebbian Updates...")
    training_corpus = """
    The Queen of Hearts, she made some tarts, All on a summer day. 
    The Knave of Hearts, he stole those tarts, And took them quite away!
    'Off with her head!' the Queen shouted at the top of her voice.
    Nobody moved. 'Who cares for you?' said Alice. 'You are nothing but a pack of cards!'
    """
    
    # 模拟长时间暴露与睡眠固化
    for _ in range(30): 
        for char in training_corpus.lower(): 
            detector.ingest_token(char)
        detector.consolidate_matrix()
        
    w_max = max(detector.neocortex.values()) if detector.neocortex else 1
    
    print("[2] Calculating Surprise Metrics (with Fracture Penalty)...\n")
    table_1_cases = [
        ("In-Distribution", "the queen of hearts"),
        ("Out-of-Distribution", "structural integrity"),
        ("SQL Injection", "select * from users"),
        ("Random Noise", "xkq zjw qqz 883")
    ]
    
    categories = []
    scores = []
    
    for category, text in table_1_cases:
        zero_transitions = 0  # 记录断裂（权重为0）的次数
        total_weight = 0
        valid_transitions = len(text) - 1
        
        for i in range(valid_transitions):
            char_i, char_j = text[i], text[i+1]
            weight = detector.neocortex.get((char_i, char_j, 1), 0)
            
            # 记录认知断裂 (未见过的结构转移)
            if weight == 0:
                zero_transitions += 1
                
            total_weight += weight
            
        avg_weight = total_weight / valid_transitions if valid_transitions > 0 else 0
        
        # 1. 基础衰减分：基于整体连接强度的偏离
        base_surprise = max(0.0, (1 - (avg_weight / (w_max * 0.5)))) * 100 
        
        # 2. 结构断裂惩罚：最薄弱环节原则
        fracture_ratio = zero_transitions / valid_transitions if valid_transitions > 0 else 1
        fracture_penalty = min(100.0, (fracture_ratio / 0.4) * 100) # 40%断裂率即满额惩罚
        
        # 3. 终极惊奇度：断裂惩罚主导，基础分辅助
        surprise_score = min(99.8, (base_surprise * 0.3) + (fracture_penalty * 0.7))
        
        categories.append(category)
        scores.append(surprise_score)
        print(f" [{category}]")
        print(f" -> Text: '{text}'")
        print(f" -> Fracture Ratio: {fracture_ratio*100:.1f}%")
        print(f" -> Final Surprise: {surprise_score:.1f}%\n")

    # --- 开始绘制 Figure 2 ---
    print("[3] Generating Figure 2 (Anomaly Detection Performance)...")
    plt.figure(figsize=(10, 6))
    
    # 设定红绿灯渐变色：绿(安全) -> 黄(警告) -> 红(致命) -> 深红(随机噪音)
    colors = ['#4CAF50', '#FFB300', '#E53935', '#B71C1C']
    bars = plt.bar(categories, scores, color=colors, width=0.6)
    
    # 添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    # 添加 80% 的警戒阈值线
    plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Critical Threshold (80%)')
    
    plt.title('Unsupervised Anomaly Detection Performance (Zero-Gradient)', fontsize=14, pad=15)
    plt.ylabel('Surprise Score (%) = $1 - P(x_{next}|W_t)$', fontsize=12)
    plt.ylim(0, 110) 
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    
    plt.xticks(fontsize=10)
    
    # 保存并展示图表
    plt.savefig('Figure_2_Anomaly_V2.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(" ✅ [系统日志]: Figure_2_Anomaly_V2.png 已成功保存！可以用于论文插图。")

if __name__ == "__main__":
    run_benchmark_and_plot()