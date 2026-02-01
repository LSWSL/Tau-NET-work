import time
from config import WorldConfig
from world import ExternalWorld
from interface import VatInterface
from brain import VatBrain

def run_training_marathon():
    WorldConfig.ensure_directories()
    
    world = ExternalWorld()
    interface = VatInterface()
    brain = VatBrain()
    
    chance = brain.body.chance_level
    
    start = WorldConfig.TIME_START
    end = WorldConfig.TIME_END # Now very large
    max_events = WorldConfig.MAX_EVENTS
    
    print("\n" + "="*60)
    print(f"  BRAIN IN A VAT: LONG-TERM TRAINING")
    print(f"  Target: {max_events} Events")
    print(f"  Strategy: Hippocampal Gating & Gradient Descent")
    print("="*60 + "\n")

    is_recording = False
    event_count = 0
    
    b_retina, b_atten, b_out = [], [], []

    try:
        for tick in range(start, end):
            brain_state = brain.state
            
            if brain_state == "AWAKE":
                world.update_time()
                raw_atom = world.manifest()
                if not raw_atom: continue
                
                atom = raw_atom[0]
                char = atom.get('content', '')
                
                if char == WorldConfig.SYMBOL_START:
                    is_recording = True
                    event_count += 1
                    b_retina, b_atten, b_out = [], [], []
                    
                    # 进度提示
                    if event_count % 10 == 1:
                        print(f"\n[Event #{event_count}/{max_events}] Reading...", end="", flush=True)

                if is_recording:
                    # 1. Transduce
                    signal = interface.transduce(raw_atom)
                    
                    # 2. Perceive
                    gated_char, gated_conf = brain.perceive(signal)
                    
                    # 3. Feedback
                    interface.receive_motor_feedback(gated_char, gated_conf)
                    
                    # Logging (只收集数据，是否打印取决于频率)
                    if event_count % 50 == 0: # 每50次打印一次详细日志，避免刷屏
                        raw_out_str, raw_out_conf = brain.current_motor_output
                        tgt, score = brain.current_attention_focus
                        
                        b_retina.append(signal.visual[WorldConfig.FOVEA_INDEX].replace('\n',' '))
                        b_atten.append(tgt if tgt and score > chance else ".")
                        b_out.append(raw_out_str[0] if raw_out_str else ".")

                if char == WorldConfig.SYMBOL_END:
                    is_recording = False
                    
                    # 只有特定时刻才打印详细输出
                    if event_count % 50 == 0:
                        print(f"\n--- Event #{event_count} Log ---")
                        print(f"[RETINA] {''.join(b_retina)}")
                        print(f"[ATTEN ] {''.join(b_atten)}")
                        print(f"[OUTPUT] {''.join(b_out)}")
                    elif event_count % 10 == 1:
                        print(" Done.", flush=True)
                    
                    # 触发睡眠
                    brain.trigger_sleep()
                    
                    # 检查是否完成任务
                    if event_count >= max_events:
                        print(f"\n[System] Target of {max_events} events reached. Terminating simulation.")
                        break
            
            elif brain_state == "ASLEEP":
                brain.perceive(None)

    except KeyboardInterrupt: 
        print("\n[System] Training interrupted by user.")
    finally:
        print("[System] Saving final brain state...")
        brain.save()

if __name__ == "__main__":
    run_training_marathon()
