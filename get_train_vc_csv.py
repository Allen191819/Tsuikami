import os
import pandas as pd
import random
import glob
from pathlib import Path
import argparse
from tqdm import tqdm

def generate_voice_conversion_csv(dataset_dirs, output_csv, vc_path, target_count=200):
    """
    生成语音转换的CSV文件，遍历所有说话人作为源说话人，支持多个数据集目录
    Args:
        dataset_dirs: 音频数据集根目录列表
        output_csv: 输出CSV文件路径
        vc_path: 转换后音频的存放根目录
        target_count: 每个源说话人对应的目标说话人数量（默认为200）
    """
    # 获取所有说话人ID和对应目录
    speaker_dirs = {}
    for dataset_dir in dataset_dirs:
        for d in os.listdir(dataset_dir):
            spk_dir = os.path.join(dataset_dir, d)
            if os.path.isdir(spk_dir):
                if d not in speaker_dirs:
                    speaker_dirs[d] = []
                speaker_dirs[d].append(spk_dir)

    speaker_ids = list(speaker_dirs.keys())
    print(f"Found {len(speaker_ids)} unique speakers across {len(dataset_dirs)} datasets")

    if len(speaker_ids) < target_count + 1:
        raise ValueError(f"Not enough speakers: need at least {target_count + 1}, found {len(speaker_ids)}")

    data = []
    used_pairs = set()

    for src_id in tqdm(speaker_ids, desc="Processing source speakers"):
        # 获取源说话人的所有音频文件（遍历所有数据集目录）
        src_audio_files = []
        for spk_dir in speaker_dirs[src_id]:
            src_audio_files.extend(glob.glob(os.path.join(spk_dir, "*.flac")))
            src_audio_files.extend(glob.glob(os.path.join(spk_dir, "*.wav")))
        if not src_audio_files:
            print(f"Warning: No audio files found for speaker {src_id}")
            continue

        remaining_speakers = [sid for sid in speaker_ids if sid != src_id]
        tar_speakers = random.sample(remaining_speakers, min(target_count, len(remaining_speakers)))

        for tar_id in tar_speakers:
            tar_audio_files = []
            for spk_dir in speaker_dirs[tar_id]:
                tar_audio_files.extend(glob.glob(os.path.join(spk_dir, "*.flac")))
                tar_audio_files.extend(glob.glob(os.path.join(spk_dir, "*.wav")))
            if len(tar_audio_files) < 2:
                print(f"Warning: Speaker {tar_id} has fewer than 2 audio files, skipping")
                continue

            src_audio = random.choice(src_audio_files)

            attempts = 0
            max_attempts = 10
            while attempts < max_attempts:
                tar_audio, ref_audio = random.sample(tar_audio_files, 2)
                pair = (src_audio, tar_audio)
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    break
                attempts += 1

            if attempts >= max_attempts:
                print(f"Warning: Could not find unique pair for src {src_id} and tar {tar_id}")
                continue

            src_name = Path(src_audio).stem
            tar_name = Path(tar_audio).stem
            vc_subdir = f"{src_id}-{tar_id}"
            vc_audio = os.path.join(vc_path, vc_subdir, f"{src_name}-{tar_name}.wav")

            os.makedirs(os.path.dirname(vc_audio), exist_ok=True)

            data.append({
                "ID": len(data) + 1,
                "src_id": src_id,
                "tar_id": tar_id,
                "src_audio": os.path.abspath(src_audio),
                "tar_audio": os.path.abspath(tar_audio),
                "ref_audio": os.path.abspath(ref_audio),
                "vc_audio": os.path.abspath(vc_audio),
                "valid": False
            })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Generated CSV with {len(df)} entries saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Generate voice conversion CSV file")
    parser.add_argument('--dataset_dirs', type=str, nargs='+', required=True,
                        help='List of audio dataset directories')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save the output CSV file')
    parser.add_argument('--vc_path', type=str, required=True,
                        help='Root directory for storing converted audio files')
    parser.add_argument('--target_count', type=int, default=200,
                        help='Number of target speakers per source speaker (default: 200)')

    args = parser.parse_args()

    try:
        generate_voice_conversion_csv(args.dataset_dirs, args.output_csv, args.vc_path, args.target_count)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
