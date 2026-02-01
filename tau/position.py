import os
import random
import json
from collections import deque
from config import WorldConfig

class Position:
    def __init__(self, file_path):
        self.file_path = file_path
        self.location_id = os.path.basename(file_path)
        self.atom_buffer = deque()
        self.current_atom = None
        # 放宽一点标点限制，防止只有句号才算
        self.valid_endings = ('.', '!', '?', '"', "'", '”', '’', '…', ';')

    def _reload_buffer(self):
        """
        [Fix] 优先读取 'content' 字段
        """
        attempts = 0
        while attempts < 10: # 增加尝试次数
            attempts += 1
            try:
                with open(self.file_path, 'r', encoding=WorldConfig.ENCODING, errors='ignore') as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    if size == 0: return

                    safe_limit = max(0, size - WorldConfig.SEEK_BUFFER)
                    random_pos = random.randint(0, safe_limit)
                    
                    f.seek(random_pos)
                    if random_pos != 0: f.readline()
                    
                    line = f.readline()
                    if not line:
                        f.seek(0)
                        line = f.readline()

                    if line:
                        try:
                            data = json.loads(line.strip())
                            
                            # [关键修复] 这里改为优先获取 'content'
                            # 如果 content 为空，再尝试 text，最后尝试 body
                            content = data.get('content') or data.get('text') or data.get('body') or ""
                            content = content.strip()
                            
                            # 调试日志：如果读到了内容但被过滤了，显示原因
                            if not content:
                                # print(f"[DEBUG] Empty content in {self.location_id}")
                                continue
                                
                            if len(content) <= 10:
                                continue
                                
                            if not content.endswith(self.valid_endings):
                                # print(f"[DEBUG] Invalid ending: {content[-1]}")
                                continue
                            
                            # 加载成功！
                            self.atom_buffer.append(WorldConfig.SYMBOL_START)
                            for char in content:
                                self.atom_buffer.append(char)
                            self.atom_buffer.append(WorldConfig.SYMBOL_END)
                            
                            # print(f"[System] Loaded: {content[:15]}...")
                            return

                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        
        # print(f"[DEBUG] {self.location_id}: Retry limit reached.")

    def evolve(self):
        if not self.atom_buffer:
            self._reload_buffer()
            
        if self.atom_buffer:
            char = self.atom_buffer.popleft()
            self.current_atom = {
                "content": char,
                "location_id": self.location_id,
                "type": "atom"
            }
        else:
            self.current_atom = None

    def get_state(self):
        return self.current_atom
