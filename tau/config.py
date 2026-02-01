import os

class WorldConfig:
    """
    [宇宙常数 - 神经动力学版]
    Constraint: Sum(Weights_to_Neuron_Y) = 1.0
    Sleep: 3000 Cycles
    """
    
    BASE_DIR = os.path.expanduser("~/tau")
    DATA_DIR = "/home/liushengwei/yushulsm/data/processed/nlp_jsonl/EN/"
    BIOLOGY_DIR = os.path.join(BASE_DIR, "biology")
    
    FILE_EXTENSION = ".jsonl"
    SEEK_BUFFER = 1000
    
    COMPLEXITY_N = 1 
    ENCODING = 'utf-8'
    
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
    
    INPUT_CHANNELS = 3 
    RECEPTIVE_FIELD = 5 
    FOVEA_INDEX = 2 
    WM_CHANNELS = 2
    MEMORY_CAPACITY = 7 
    
    # --- 神经参数 ---
    LTM_LAYERS = 24
    DIM_N = len(VALID_SYMBOLS)
    
    # 一个神经元接受的传入连接总数 (24层 * 81个符号)
    INCOMING_CONNECTIONS = LTM_LAYERS * DIM_N
    
    # [神经动力学初始化]
    # 归一化条件：Sum = 1.0
    # 初始权重 = 1.0 / 传入连接总数 (~0.0005)
    INITIAL_SYNAPTIC_WEIGHT = 1.0 / INCOMING_CONNECTIONS
    
    # [动力学参数]
    # 睡眠时的重放循环次数
    SLEEP_REPLAY_CYCLES = 3000
    # 学习力度 (每次注入多少能量)
    NEUROPLASTICITY_RATE = 0.1 
    
    # [梯度与父母]
    BASE_LEARNING_RATE = 1.0
    HIPPOCAMPUS_DEPTH = 3
    PARENTAL_VOICE_CONFIDENCE = 0.8  
    PARENTAL_INTERVENTION_RATE = 0.6 
    
    SLEEP_DURATION = 300 
    
    VISUAL_ATTENTION_MASK = [0.3, 0.5, 1.0, 0.5, 0.3]
    MEMORY_DECAY_FACTOR = 0.85 
    MEMORY_FORGET_THRESHOLD = 0.1
    VISUAL_NOISE_LEVEL = 0.0
    EXTERNAL_AUDIO_BASE_CONFIDENCE = 0.01 
    
    TIME_ATOM_UNIT = 1
    PLANCK_TICK = 0.005
    TIME_START = 0
    
    TIME_END = 10000000 
    MAX_EVENTS = 5000

    @staticmethod
    def ensure_directories():
        if not os.path.exists(WorldConfig.DATA_DIR):
            print(f"[Warning] Physics engine cannot find matter at: {WorldConfig.DATA_DIR}")
        if not os.path.exists(WorldConfig.BIOLOGY_DIR):
            os.makedirs(WorldConfig.BIOLOGY_DIR)
