import os

class WorldConfig:
    # --- 1. 核心超参数 (修复点) ---
    N_ORDER = 5             # 5阶注意力 (看过去5个字)
    MEMORY_SIZE = 500000    # 稀疏哈希槽位数量

    # --- 2. 时空设置 ---
    TIME_START = 0
    TIME_END = 10000000       
    MAX_EVENTS = 10000        
    
    # 请确认这个路径是正确的
    DATA_DIR = "/home/liushengwei/yushulsm/data/processed/nlp_jsonl/EN/"
    OUTPUT_DIR = os.path.expanduser("~/tau/output")
    FILE_EXTENSION = ".jsonl" 

    # --- 3. 符号集 ---
    ALPHAS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    DIGITS = "0123456789"
    PUNCTS = ".,!?;:'\"()[]{}<>-+*=/@#%&|_^~`$€£¥\\"
    CONTROL = " \n"
    _RAW = ALPHAS + DIGITS + PUNCTS + CONTROL
    VALID_SYMBOLS = "".join(dict.fromkeys(_RAW))
    SYMBOL_VOID = "∅"

    @staticmethod
    def ensure_directories():
        os.makedirs(WorldConfig.OUTPUT_DIR, exist_ok=True)
