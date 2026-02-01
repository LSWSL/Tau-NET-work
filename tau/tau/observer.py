import sys
from interface import SensoryInput
from config import WorldConfig

class StreamObserver:
    """
    [流式观察器]
    不思考，只观察。
    逻辑：Start(▼) -> Print -> Stop(▲)
    """
    def __init__(self):
        self.active = False
        print("[Observer] Online. Waiting for signal stream...")

    def perceive(self, signal: SensoryInput):
        if not signal.channels:
            return

        # 获取当前的时间原子 (字符)
        char = signal.channels[0]['data']

        # --- 状态机逻辑 ---

        # 1. 检测到开始符：激活
        if char == WorldConfig.SYMBOL_START:
            self.active = True
            print(f"\n[Event] ", end='', flush=True)

        # 2. 如果处于激活状态：打印
        if self.active:
            # 打印字符，不换行，且强制刷新缓冲区以保证实时性
            print(char, end='', flush=True)

        # 3. 检测到结束符：关闭
        if char == WorldConfig.SYMBOL_END:
            self.active = False
            # 打印一个完成标记，便于肉眼区分
            print(" <EOF>", flush=True)

    def sleep(self):
        print("\n[Observer] Offline.")
