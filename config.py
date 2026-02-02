import os

class WorldConfig:
    """
    [Tau-Net 宇宙常数]
    适配论文：Natural Numbers, Addition, and Subtraction are All You Need
    """
    
    BASE_DIR = os.path.expanduser("~/tau")
    # 请确认这个路径是否依然有效，或者改回您的本地路径
    DATA_DIR = "/home/liushengwei/yushulsm/data/processed/nlp_jsonl/EN/"
    BIOLOGY_DIR = os.path.join(BASE_DIR, "biology")
    
    FILE_EXTENSION = ".jsonl"
    SEEK_BUFFER = 1000
    ENCODING = 'utf-8'
    
    # --- 符号定义 ---
    SYMBOL_START = "▼" 
    SYMBOL_END =   "▲"
    SYMBOL_VOID =  "∅" 
    SYMBOL_NOISE = "?" 
    SYMBOL_SILENCE="~" 
    
    VALID_SYMBOLS = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,!?;:'\"-()[]"
        f"{SYMBOL_START}{SYMBOL_END}{SYMBOL_VOID}{SYMBOL_NOISE}{SYMBOL_SILENCE}"
    )
    
    # --- Tau-Net 核心参数 ---
    # [新增] 上下文窗口大小 (N-gram) - 论文中提到的滑动窗口
    CONTEXT_WINDOW = 4
    
    # [新增] 哈希表大小 (M_size) - 决定了记忆容量和冲突率
    TAU_MEMORY_SIZE = 500000
    
    # [新增] 突触注入常数 (Delta) - 整数加法学习率
    TAU_DELTA = 10
    
    # --- 视觉参数 ---
    INPUT_CHANNELS = 3 
    RECEPTIVE_FIELD = 5 
    FOVEA_INDEX = 2 
    WM_CHANNELS = 2
    MEMORY_CAPACITY = 7 
    
    # --- 睡眠与父母参数 ---
    # Tau-Net 的睡眠是随机衰减过程
    SLEEP_DURATION = 300 
    
    # 父母干预 (用于 body.py)
    PARENTAL_VOICE_CONFIDENCE = 0.8  
    PARENTAL_INTERVENTION_RATE = 0.6 
    
    VISUAL_ATTENTION_MASK = [0.3, 0.5, 1.0, 0.5, 0.3]
    MEMORY_DECAY_FACTOR = 0.85 
    MEMORY_FORGET_THRESHOLD = 0.1
    VISUAL_NOISE_LEVEL = 0.0
    
    # --- 仿真时间参数 ---
    TIME_ATOM_UNIT = 1
    TIME_START = 0
    TIME_END = 10000000 
    MAX_EVENTS = 5000

    @staticmethod
    def ensure_directories():
        if not os.path.exists(WorldConfig.DATA_DIR):
            print(f"[Warning] Physics engine cannot find matter at: {WorldConfig.DATA_DIR}")
        if not os.path.exists(WorldConfig.BIOLOGY_DIR):
            os.makedirs(WorldConfig.BIOLOGY_DIR, exist_ok=True)