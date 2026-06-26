import pandas as pd
import os
import re
import numpy as np
import resampy
import soundfile as sf
from supertonic import TTS

# Paths
ORIG_AUDIO_ROOT = "/media/hdd1/amkapoor/N400/N400Stimset_manuscriptdata/N400Stimset_manuscriptdata/stimuli"
TARGET_FS = 22050
RESULTS_DIR = "results_f"
os.makedirs(RESULTS_DIR, exist_ok=True)

files = [
    "filelists/n400/filelist_train_text_ipa.txt",
    "filelists/n400/filelist_test_unseen_audio_text_ipa.txt",
    "filelists/n400/filelist_test_unseen_both_text_ipa.txt",
    "filelists/n400/filelist_test_unseen_subject_text_ipa.txt"
]

dfs = []
for f in files:
    df = pd.read_csv(f, sep=r"\|\|", names=["orig_path", "text", "ipa"], engine="python")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
# Keep original path for lookup, but create a clean filename for saving
data["clean_filename"] = data["orig_path"].apply(os.path.basename).str.replace(r"^sub-\d+-_-", "", regex=True)
data = data.drop_duplicates(subset="clean_filename", keep="first").reset_index(drop=True)

tts = TTS(auto_download=True)
style = tts.get_voice_style(voice_name="F4")

duration_logs = []

for i in range(len(data)):
    text = data["text"].values[i]
    orig_rel_path = data["orig_path"].values[i]
    clean_name = data["clean_filename"].values[i]
    
    orig_audio_path = os.path.join(ORIG_AUDIO_ROOT, f"{clean_name}.wav")
    
    if not os.path.exists(orig_audio_path):
        print(f"Warning: Original audio not found at {orig_audio_path}. Skipping duration matching.")
        orig_duration = None
    else:
        orig_info = sf.info(orig_audio_path)
        orig_duration = orig_info.duration

    # Duration matching loop
    current_speed = 0.7 # Starting speed from original script
    best_wav = None
    best_duration = 0
    
    if orig_duration:
        max_iters = 5
        for attempt in range(max_iters):
            wav, syn_duration = tts.synthesize(text, voice_style=style, speed=current_speed, silence_duration=0.3)
            syn_duration = float(syn_duration)
            
            print(f"[{clean_name}] Attempt {attempt+1}: speed={current_speed:.3f}, duration={syn_duration:.3f}s (target={orig_duration:.3f}s)")
            
            # Update best_wav: we want the longest duration that is still <= orig_duration
            # If we haven't found anything <= orig_duration, we take the one closest to it (which will be the one at max speed)
            if best_wav is None or (syn_duration <= orig_duration and syn_duration > best_duration) or (best_duration > orig_duration and syn_duration < best_duration):
                best_wav = wav
                best_duration = syn_duration

            if syn_duration > orig_duration:
                # Too slow, increase speed
                current_speed *= (syn_duration / orig_duration)
            elif syn_duration < 0.98 * orig_duration:
                # Too fast, decrease speed slightly if we have room
                current_speed *= (syn_duration / orig_duration)
            else:
                # Close enough
                break
            
            # Clamp speed to library limits [0.7, 2.0]
            current_speed = max(0.7, min(2.0, current_speed))
    else:
        # Fallback if original not found
        best_wav, best_duration = tts.synthesize(text, voice_style=style, speed=current_speed, silence_duration=0.3)
        best_duration = float(best_duration)

    # Process and save
    wav = best_wav.squeeze()
    orig_fs = tts.sample_rate
    wav_22k = resampy.resample(wav, orig_fs, TARGET_FS)

    out_file = os.path.join(RESULTS_DIR, f"{clean_name}.wav")
    sf.write(out_file, wav_22k, TARGET_FS)

    duration_logs.append({
        "filename": clean_name,
        "original_duration": orig_duration,
        "recreated_duration": best_duration,
        "duration_difference": (best_duration - orig_duration) if orig_duration is not None else None,
        "final_speed": current_speed
    })

    target_dur_str = f"{orig_duration:.3f}s" if orig_duration is not None else "N/A"
    print(f"Saved {out_file} | Final Duration: {float(best_duration):.3f}s | Target: {target_dur_str}")

# Save duration logs
log_df = pd.DataFrame(duration_logs)
log_df.to_csv(os.path.join(RESULTS_DIR, "duration_log.csv"), index=False)
print(f"Duration log saved to {os.path.join(RESULTS_DIR, 'duration_log.csv')}")
