import os
import pandas as pd
import random
import argparse

def build_vc_csv(dataset_path: str, vc_path: str, output_csv: str, num_samples: int = 20):
    """
    构建变声语音的 CSV 文件。
    
    参数:
    - dataset_path: 语音数据集路径，包含若干说话人子文件夹
    - vc_path: 变声语音保存路径
    - output_csv: 输出 CSV 文件路径
    - num_samples: 每个说话人选择的样本数量，默认 20
    """
    # 获取所有说话人 ID（子文件夹名称）
    speaker_ids = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if len(speaker_ids) < 2:
        raise ValueError("数据集必须至少包含 2 个说话人")

    # 初始化结果列表
    data = []

    # 遍历所有可能的 src_id 和 tar_id 组合
    for src_id in speaker_ids:
        src_folder = os.path.join(dataset_path, src_id)
        src_files = [f for f in os.listdir(src_folder) if f.endswith(('.wav', '.flac'))]
        # print(src_files)
        if len(src_files) < num_samples:
            print(f"Warning: {src_id} has fewer than {num_samples} files, using all available")
            src_samples = src_files
        else:
            src_samples = random.sample(src_files, num_samples)

        for tar_id in speaker_ids:
            if src_id == tar_id:  # 跳过 src_id == tar_id 的情况
                continue
            tar_folder = os.path.join(dataset_path, tar_id)
            tar_files = [f for f in os.listdir(tar_folder) if f.endswith(('.wav', '.flac'))]
            # print(tar_files)
            if len(tar_files) < num_samples + 1:  # +1 因为需要额外的 ref_audio
                print(f"Warning: {tar_id} has fewer than {num_samples + 1} files, skipping or adjusting")
                continue
            
            # 选择 tar_audio 和 ref_audio
            tar_samples = random.sample(tar_files, num_samples + 1)  # 多选一条用于 ref_audio
            tar_audios = tar_samples[:-1]  # 前 num_samples 条作为 tar_audio
            ref_audio = tar_samples[-1]   # 最后一条作为 ref_audio

            # 按索引组合 src_audio 和 tar_audio
            for idx in range(num_samples):
                src_audio = src_samples[idx]
                tar_audio = tar_audios[idx]
                
                # 提取文件名（去掉后缀）
                src_name = os.path.splitext(src_audio)[0]
                tar_name = os.path.splitext(tar_audio)[0]
                
                # 构建 vc_subdir 和 vc_audio
                vc_subdir = f"{src_id}-{tar_id}"
                vc_audio = os.path.join(vc_path, vc_subdir, f"{src_name}-{tar_name}.wav")
                
                # 添加记录
                data.append({
                    "ID": len(data) + 1,  # 自增 ID
                    "src_id": src_id,
                    "tar_id": tar_id,
                    "src_audio": os.path.join(dataset_path, src_id, src_audio),
                    "tar_audio": os.path.join(dataset_path, tar_id, tar_audio),
                    "ref_audio": os.path.join(dataset_path, tar_id, ref_audio),
                    "vc_audio": vc_audio,
                    "valid": False
                })

    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv} with {len(df)} entries")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Build CSV for Voice Conversion Dataset")
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help="Path to the original audio dataset with speaker subfolders")
    parser.add_argument('--vc_path', type=str, required=True, 
                        help="Path to save voice-converted audio files")
    parser.add_argument('--output_csv', type=str, default="vc_dataset.csv", 
                        help="Path to save the output CSV file (default: vc_dataset.csv)")
    parser.add_argument('--num_samples', type=int, default=20, 
                        help="Number of samples per speaker (default: 20)")

    args = parser.parse_args()

    # 确保 vc_path 存在
    if not os.path.exists(args.vc_path):
        os.makedirs(args.vc_path)

    # 运行构建函数
    build_vc_csv(
        dataset_path=args.dataset_path,
        vc_path=args.vc_path,
        output_csv=args.output_csv,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
