import torch
import random
import torch.nn as nn

class CutMixAudioAugmenter(nn.Module):
    def __init__(self, max_len, min_len, max_frames, mix_rate, device):
        super(CutMixAudioAugmenter, self).__init__()
        self.max_len = max_len  # 每段的最大长度
        self.min_len = min_len  # 每段的最小长度
        self.max_frames = max_frames  # 最终音频长度
        self.mix_rate = mix_rate  # 混合比例
        self.device = device  # 目标设备

    def forward(self, sig_a, sig_b, lens):
        """
        Args:
            sig_a (Tensor): Processed audio signal A, shape [1, num_frames]
            sig_b (Tensor): Processed audio signal B, shape [1, num_frames]

        Returns:
            Tensor: Mixed audio signal, shape [1, max_frames]
        """
        # Step 1: 计算每段的长度范围
        segment_length = random.randint(self.min_len, self.max_len)
        
        # 计算音频可以分成多少段
        num_segments = sig_a.size(-1) // segment_length
        
        # Step 2: 切分音频 A 和 B 成多个段落
        segments_a = [sig_a[:, i * segment_length: (i + 1) * segment_length] for i in range(num_segments)]
        segments_b = [sig_b[:, i * segment_length: (i + 1) * segment_length] for i in range(num_segments)]

        # Step 3: 按照 mix_rate 随机选择需要替换的段落
        num_replace = int(num_segments * self.mix_rate)  # 需要替换的段落数量
        replace_indices = random.sample(range(num_segments), num_replace)  # 随机选择替换的段落索引

        mixed_segments = []

        for i in range(num_segments):
            if i in replace_indices:
                mixed_segments.append(segments_b[i])  # 用 B 中对应段落替换
            else:
                mixed_segments.append(segments_a[i])  # 保留原始 A 中的段落

        # Step 4: 合并所有段落
        mixed_signal = torch.cat(mixed_segments, dim=-1)

        # Step 5: 确保音频长度为 max_frames
        if mixed_signal.size(-1) < self.max_frames:
            mixed_signal = torch.nn.functional.pad(mixed_signal, (0, self.max_frames - mixed_signal.size(-1)))
        elif mixed_signal.size(-1) > self.max_frames:
            mixed_signal = mixed_signal[:, :self.max_frames]

        # Step 6: 转移到目标设备
        mixed_signal = mixed_signal.to(self.device)

        return mixed_signal, lens
