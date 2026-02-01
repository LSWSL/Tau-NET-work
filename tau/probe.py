import time
from world import ExternalWorld
from interface import VatInterface
from config import WorldConfig

def probe_data_quality(samples=50):
    print("="*60)
    print(f"  DATA QUALITY PROBE (N={samples})")
    print("  Checking for: Void events & Abnormal endings")
    print("="*60)

    # 1. 初始化
    world = ExternalWorld()
    interface = VatInterface()
    
    if not world.positions:
        print("[Error] No positions found.")
        return

    void_count = 0
    clean_endings = 0
    abnormal_endings = 0
    
    print(f"\n[Probing started] Sampling {samples} events...\n")

    for i in range(samples):
        # 驱动世界和接口
        world.update_time()
        raw_events = world.manifest()
        sensory_signal = interface.transduce(raw_events)
        
        # 获取 N=1 的那个通道数据
        if not sensory_signal.channels:
            continue
            
        # 数据格式: ▼内容▲
        framed_data = sensory_signal.channels[0]['data']
        
        # 剥离外壳，检查核心
        # 移除首尾符号
        core_content = framed_data.replace(WorldConfig.SYMBOL_START, '').replace(WorldConfig.SYMBOL_END, '')
        
        # --- 检查 1: 是否为空 ---
        if not core_content.strip():
            print(f"  [!] Sample {i+1}: VOID DETECTED (Empty or Whitespace)")
            void_count += 1
            continue

        # --- 检查 2: 结尾字符 ---
        # 去除末尾换行符后，看最后一个字符
        clean_text = core_content.strip()
        last_char = clean_text[-1]
        
        # 定义什么是"正常结尾" (句子通常以标点结束)
        normal_punctuations = {'.', '!', '?', '"', "'", '”', '’'}
        
        is_normal = last_char in normal_punctuations
        status = "OK" if is_normal else "ABNORMAL"
        
        if is_normal:
            clean_endings += 1
        else:
            abnormal_endings += 1
            
        # 打印异常或随机抽样打印正常样本
        if not is_normal or (i % 10 == 0):
            # 显示最后10个字符
            snippet = clean_text[-15:].replace('\n', '\\n')
            print(f"  [{status}] Sample {i+1} ends with: [...{snippet}] (Char: '{last_char}')")

    # 统计报告
    print("\n" + "="*60)
    print("  PROBE REPORT")
    print("="*60)
    print(f"  Total Samples:    {samples}")
    print(f"  Void (Empty):     {void_count}  ({(void_count/samples)*100:.1f}%)")
    print(f"  Normal Endings:   {clean_endings}  ({(clean_endings/samples)*100:.1f}%)")
    print(f"  Abrupt Endings:   {abnormal_endings}  ({(abnormal_endings/samples)*100:.1f}%)")
    
    if void_count > 0:
        print("\n  [ADVICE] We should filter out empty strings in position.py.")
    if abnormal_endings > 0:
        print("  [NOTE] Abrupt endings might be Titles, Headers, or JSON artifacts.")

if __name__ == "__main__":
    probe_data_quality()
