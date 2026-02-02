import os
import torch
import torchaudio
import torch.nn.functional as F
import random
import pickle
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import logit
from models.DNN_PLDA import NeuralPLDA
from sklearn.manifold import TSNE
from tabulate import tabulate
from tqdm import tqdm
from collections import defaultdict
from utils.simulate import simulate_telephone_audio_torch
from scipy.stats import bootstrap
from statsmodels.stats.proportion import proportion_confint
from sklearn.isotonic import IsotonicRegression
from utils.SBEncoder import *
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda")

def to_json_serializable(obj):

    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return float(obj.item())
    else:
        return obj


class Tester:
    def __init__(self, encoder, args, vc_folder: str, orig_folder: str, output_dir: str,
                 resume: bool = False, batch_size: int = 60, sample_rate: int = 16000,
                 duration: int = 6, phone_channel=False, num_orig_samples: int = 1, add_noise: bool = False, noise_file: str = None, channel: str = 'G.711', channel_enhance: bool = False, plda: bool = False):
        self.device = device
        self.encoder = encoder
        self.vc_folder = vc_folder
        self.orig_folder = orig_folder
        self.output_dir = output_dir
        self.resume = resume
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.duration = duration
        self.vc_embeddings = {}
        self.orig_embeddings = {}
        self.test_pairs = []
        self.args = args
        self.phone_channel = phone_channel
        self.num_orig_samples = num_orig_samples
        self.channel = channel
        self.channel_enhance = channel_enhance
        self.add_noise = add_noise
        self.noise_file = noise_file
        self.w_en = args.w_en
        if plda:
            self.plda = NeuralPLDA(emb_dim=192).to(self.device)
            self.plda.load_state_dict(torch.load(os.path.join(self.args.model_src,"neural_plda.pth"), map_location=self.device))
        else:
            self.plda = None
        if self.add_noise:
            self.noise_wav, sr = torchaudio.load(self.noise_file)
            print(f"Loaded noise file {self.noise_file} with sample rate {sr}")
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                self.noise_wav = resampler(self.noise_wav)
            if self.noise_wav.shape[0] > 1:
                self.noise_wav = self.noise_wav.mean(dim=0, keepdim=True)
        else:
            self.noise_wav = None
        if phone_channel:
            print(f"Test with simulate {self.channel} telephone audio...")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.vc_has_subdirs = self._check_vc_folder_structure()

    def _check_vc_folder_structure(self):
        items = os.listdir(self.vc_folder)
        for item in items:
            if os.path.isdir(os.path.join(self.vc_folder, item)):
                return True
        return False

    def _get_vc_files(self):
        if not self.vc_has_subdirs:
            return [os.path.join(self.vc_folder, f) for f in os.listdir(self.vc_folder) if f.endswith('.wav')]
        else:
            vc_files = []
            for subdir in os.listdir(self.vc_folder):
                subdir_path = os.path.join(self.vc_folder, subdir)
                if os.path.isdir(subdir_path):
                    vc_files.extend([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.wav')])
            return vc_files

    def _parse_speaker_ids(self, vc_file):
        basename = os.path.basename(vc_file)
        if self.vc_has_subdirs:
            subdir = os.path.basename(os.path.dirname(vc_file))
            src_id, tar_id = subdir.split('-')
        else:
            parts = basename.split('_to_')
            src_id = parts[0].split('_')[0]
            tar_id = parts[1].split('_')[0]
        return src_id, tar_id

    def preprocess_audio(self, file_paths: list, target_sample_rate: int, target_duration: int, phone_channel=False, channel: str = 'G.711', add_noise: bool = False):
        num_samples = target_sample_rate * target_duration
        waveforms = []
       
        for file_path in file_paths:
            if file_path.lower() == "none" or not os.path.exists(file_path):
                waveform = torch.zeros(1, num_samples)
            else:
                waveform, sample_rate = torchaudio.load(file_path)
                if phone_channel:
                    waveform, sample_rate = simulate_telephone_audio_torch(waveform, sample_rate, channel=channel)
                if sample_rate != target_sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                    waveform = resampler(waveform)
                if add_noise:
                    if self.noise_wav is None:
                        raise ValueError("Noise waveform is None, please check the noise file path.")
                    if self.noise_wav.shape[1] > waveform.shape[1]:
                        noise_start = random.randint(0, self.noise_wav.shape[1] - waveform.shape[1])
                        noise_segment = self.noise_wav[:, noise_start:noise_start + waveform.shape[1]]
                    else:
                        repeat_count = (waveform.shape[1] // self.noise_wav.shape[1]) + 1
                        noise_segment = self.noise_wav.repeat(1, repeat_count)[:, :waveform.shape[1]]
                    signal_power = waveform.pow(2).mean()
                    noise_power = noise_segment.pow(2).mean()
                    target_snr_db = 0
                    target_noise_power = signal_power / (10 ** (target_snr_db / 10))
                    scaling_factor = (target_noise_power / (noise_power + 1e-9)).sqrt()
                    noise_scaled = noise_segment * scaling_factor
                    waveform = waveform + noise_scaled
                if waveform.shape[1] > num_samples:
                    waveform = waveform[:, :num_samples]
                if waveform.shape[1] < num_samples:
                    repeat_count = (num_samples // waveform.shape[1]) + 1
                    waveform = waveform.repeat(1, repeat_count)
                    waveform = waveform[:, :num_samples]
            waveforms.append(waveform)
       
        waveforms = torch.stack(waveforms).squeeze(1) # [batch_size, time]
        return waveforms

    @torch.no_grad()
    def compute_vc_embeddings(self):
        if self.resume and os.path.exists(os.path.join(self.output_dir, "vc_embeddings.pkl")):
            with open(os.path.join(self.output_dir, "vc_embeddings.pkl"), 'rb') as f:
                self.vc_embeddings = pickle.load(f)
            print(f"Loaded VC embeddings from {self.output_dir}")
            return
       
        vc_files = self._get_vc_files()
        for i in tqdm(range(0, len(vc_files), self.batch_size), desc="Computing VC embeddings"):
            batch_files = vc_files[i:i + self.batch_size]
            ref_files = []
            for vc_file in batch_files:
                _, tar_id = self._parse_speaker_ids(vc_file)
                speaker_folder = os.path.join(self.orig_folder, tar_id)
                if os.path.exists(speaker_folder):
                    orig_files = [f for f in os.listdir(speaker_folder) if f.endswith('.flac') or f.endswith('.wav')]
                    ref_file = os.path.join(speaker_folder, random.choice(orig_files))
                else:
                    ref_file = "none"
                ref_files.append(ref_file)
           
            vc_waveforms = self.preprocess_audio(batch_files, self.sample_rate, self.duration, self.phone_channel, self.channel)
            ref_waveforms = self.preprocess_audio(ref_files, self.sample_rate, self.duration, False, self.channel)
           
            embeddings = self.encoder.encode_vc_audio(vc_waveforms.to(self.encoder.device),
                                                      ref_waveforms.to(self.encoder.device))
            embeddings = embeddings / torch.linalg.norm(embeddings, dim=-1, keepdim=True)
           
            for file_path, embedding in zip(batch_files, embeddings):
                self.vc_embeddings[file_path] = embedding.cpu()
       
        with open(os.path.join(self.output_dir, "vc_embeddings.pkl"), 'wb') as f:
            pickle.dump(self.vc_embeddings, f)
        print(f"Saved VC embeddings to {self.output_dir}")

    def preprocess_and_embed(self, wav_tensor, sr_in):
        resampler = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=self.sample_rate)
        wav_resampled = resampler(wav_tensor)
        num_samples = self.sample_rate * self.duration
        if wav_resampled.shape[1] > num_samples:
            wav_resampled = wav_resampled[:, :num_samples]
        if wav_resampled.shape[1] < num_samples:
            repeat_count = (num_samples // wav_resampled.shape[1]) + 1
            wav_resampled = wav_resampled.repeat(1, repeat_count)
            wav_resampled = wav_resampled[:, :num_samples]
        embedding = self.encoder.encode_ori_audio(wav_resampled.to(self.encoder.device))
        embedding = embedding / torch.linalg.norm(embedding, dim=-1, keepdim=True)
        return embedding.squeeze(0)

    def bandpass_filter(self, wav_tensor, sr, low, high):
        f_center = (low + high) / 2
        bandwidth = high - low
        Q = f_center / bandwidth
        return torchaudio.functional.bandpass_biquad(
            waveform=wav_tensor,
            sample_rate=sr,
            central_freq=f_center,
            Q=Q)

    def add_white_noise(self, waveform, std=0.005):
        noise = torch.randn_like(waveform) * std
        return torch.clamp(waveform + noise, -1, 1)

    @torch.no_grad()
    def compute_enhanced_embedding(self, wav, sr):
        if not self.channel_enhance:
            return self.preprocess_and_embed(wav, sr)
       
        all_embeddings = []
        weights = []
       
        # 原始A
        emb_A = self.preprocess_and_embed(wav, sr)
        all_embeddings.append(emb_A)
        weights.append(self.w_en)
       
        # 增强B：300-3400Hz → 8000Hz → u-law 8bit
        bandpass_B = self.bandpass_filter(wav, sr, 300, 3400)
        resample_B = torchaudio.transforms.Resample(sr, 8000)(bandpass_B)
        ulaw_B = torchaudio.functional.mu_law_encoding(resample_B, quantization_channels=256)
        wav_B = torchaudio.functional.mu_law_decoding(ulaw_B, quantization_channels=256)
        wav_B = self.add_white_noise(wav_B, std=0.003)
        emb_B = self.preprocess_and_embed(wav_B, 8000)
        all_embeddings.append(emb_B)
        weights.append(1.0)
       
        # 增强C：50-7000Hz → 16000Hz → 10bit线性
        bandpass_C = self.bandpass_filter(wav, sr, 50, 7000)
        resample_C = torchaudio.transforms.Resample(sr, 16000)(bandpass_C)
        max_val = 2 ** 9 - 1
        wav_C = (resample_C * max_val).round() / max_val
        wav_C = self.add_white_noise(wav_C, std=0.003)
        emb_C = self.preprocess_and_embed(wav_C, 16000)
        all_embeddings.append(emb_C)
        weights.append(1.0)
       
        # 增强D：50-4000Hz → 8000Hz → 24kbps
        bandpass_D = self.bandpass_filter(wav, sr, 50, 4000)
        resample_D = torchaudio.transforms.Resample(sr, 8000)(bandpass_D)
        ulaw_D = torchaudio.functional.mu_law_encoding(resample_D, quantization_channels=16)
        wav_D = torchaudio.functional.mu_law_decoding(ulaw_D, quantization_channels=16)
        wav_D = self.add_white_noise(wav_D, std=0.003)
        emb_D = self.preprocess_and_embed(wav_D, 8000)
        all_embeddings.append(emb_D)
        weights.append(1.0)
       
        # 增强E：50-8000Hz → 16000Hz → 32kbps
        bandpass_E = self.bandpass_filter(wav, sr, 50, 8000)
        resample_E = torchaudio.transforms.Resample(sr, 16000)(bandpass_E)
        ulaw_E = torchaudio.functional.mu_law_encoding(resample_E, quantization_channels=32)
        wav_E = torchaudio.functional.mu_law_decoding(ulaw_E, quantization_channels=32)
        wav_E = self.add_white_noise(wav_E, std=0.002)
        emb_E = self.preprocess_and_embed(wav_E, 16000)
        all_embeddings.append(emb_E)
        weights.append(1.0)
       
        # 加权平均
        embeddings_stack = torch.stack(all_embeddings)
        weights_tensor = torch.tensor(weights, device=embeddings_stack.device).unsqueeze(1)
        weighted_sum = torch.sum(embeddings_stack * weights_tensor, dim=0)
        emb = weighted_sum / weights_tensor.sum()
        emb = emb / torch.linalg.norm(emb, dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def compute_orig_embeddings(self):
        print(f"Computing original embeddings with {self.num_orig_samples} samples per speaker...")
        enhance_str = "with" if self.channel_enhance else "without"
        # if self.resume and os.path.exists(os.path.join(self.output_dir, f"orig_embeddings_{self.num_orig_samples}_{enhance_str}_channel_enhance.pkl")):
        #     with open(os.path.join(self.output_dir, f"orig_embeddings_{self.num_orig_samples}_{enhance_str}_channel_enhance.pkl"), 'rb') as f:
        #         self.orig_embeddings = pickle.load(f)
        #     print(f"Loaded original embeddings from {self.output_dir}")
        #     return
        speaker_ids = [d for d in os.listdir(self.orig_folder) if os.path.isdir(os.path.join(self.orig_folder, d))]
        for speaker_id in tqdm(speaker_ids, desc="Computing original embeddings"):
            speaker_folder = os.path.join(self.orig_folder, speaker_id)
            orig_files = [os.path.join(speaker_folder, f) for f in os.listdir(speaker_folder) if f.endswith('.flac') or f.endswith('.wav')]
            selected_files = random.sample(orig_files, min(self.num_orig_samples, len(orig_files)))
            all_embeddings = []
            for file_path in selected_files:
                wav, sr = torchaudio.load(file_path)
                emb = self.compute_enhanced_embedding(wav, sr)
                all_embeddings.append(emb)
            embeddings_stack = torch.stack(all_embeddings)
            mean_embedding = torch.mean(embeddings_stack, dim=0)
            mean_embedding = mean_embedding / torch.linalg.norm(mean_embedding, dim=-1, keepdim=True)
            self.orig_embeddings[speaker_id] = mean_embedding.cpu()
       
        enhance_str = "with" if self.channel_enhance else "without"
        with open(os.path.join(self.output_dir, f"orig_embeddings_{self.num_orig_samples}_{enhance_str}_channel_enhance.pkl"), 'wb') as f:
            pickle.dump(self.orig_embeddings, f)
        print(f"Saved original embeddings to {self.output_dir}")

    def compute_topk_accuracy(self, k_values=[1, 2, 5, 10]):
        correct_counts = {k: 0 for k in k_values}
        total = len(self.vc_embeddings)
        correct_per_class = defaultdict(int)
        incorrect_per_class = defaultdict(int)
        for vc_file, vc_emb in tqdm(self.vc_embeddings.items(), desc="Computing Top-K accuracy"):
            src_id, _ = self._parse_speaker_ids(vc_file)
            if self.plda is not None:
                similarities = {spk_id: self.plda(orig_emb.to(self.device).unsqueeze(0), vc_emb.to(self.device).unsqueeze(0)).item()
                                for spk_id, orig_emb in self.orig_embeddings.items()}
            else:
                similarities = {spk_id: F.cosine_similarity(vc_emb, orig_emb, dim=0).item()
                                for spk_id, orig_emb in self.orig_embeddings.items()}
            sorted_speakers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
           
            for k in k_values:
                top_k_speakers = [spk for spk, _ in sorted_speakers[:k]]
                if src_id in top_k_speakers:
                    correct_counts[k] += 1
                if k == 1:
                    if src_id in top_k_speakers:
                        correct_per_class[src_id] += 1
                    else:
                        incorrect_per_class[src_id] += 1
        table_data = [(spk, correct_per_class.get(spk, 0), incorrect_per_class.get(spk, 0))
              for spk in set(correct_per_class.keys()).union(set(incorrect_per_class.keys()))]
        print("\nPer-Class Accuracy:")
        table_str = tabulate(table_data, headers=["Speaker ID", "Correct", "Incorrect"], tablefmt="grid")
        print(table_str)
        with open(os.path.join(self.output_dir, f'per_class_accuracy_{self.num_orig_samples}.txt'), 'w') as f:
            f.write("Per-Class Accuracy:\n")
            f.write(table_str)
        accuracies = {}
        for k in k_values:
            acc = correct_counts[k] / total
            ci_low, ci_high = proportion_confint(correct_counts[k], total, alpha=0.05, method='wilson')
            accuracies[k] = {
                "acc": acc,
                "95CI_low": ci_low,
                "95CI_high": ci_high
            }
        return accuracies

    def construct_test_pairs(self, num_pos=5, num_neg=5):
        if self.resume and os.path.exists(os.path.join(self.output_dir, "test_pairs.pkl")):
            with open(os.path.join(self.output_dir, "test_pairs.pkl"), 'rb') as f:
                self.test_pairs = pickle.load(f)
            print(f"Loaded test pairs from {self.output_dir}")
            return
       
        all_speakers = list(self.orig_embeddings.keys())
        for vc_file in tqdm(self.vc_embeddings.keys(), desc="Constructing test pairs"):
            src_id, _ = self._parse_speaker_ids(vc_file)
            pos_files = [f for f in os.listdir(os.path.join(self.orig_folder, src_id))
                        if f.endswith('.flac') or f.endswith('.wav')]
            pos_files = random.sample(pos_files, min(num_pos, len(pos_files)))
            for pos_file in pos_files:
                self.test_pairs.append((vc_file, os.path.join(self.orig_folder, src_id, pos_file), 1))
           
            neg_speakers = random.sample([s for s in all_speakers if s != src_id], num_neg)
            for neg_speaker in neg_speakers:
                neg_files = [f for f in os.listdir(os.path.join(self.orig_folder, neg_speaker))
                            if f.endswith('.flac') or f.endswith('.wav')]
                neg_file = random.choice(neg_files)
                self.test_pairs.append((vc_file, os.path.join(self.orig_folder, neg_speaker, neg_file), 0))
       
        with open(os.path.join(self.output_dir, "test_pairs.pkl"), 'wb') as f:
            pickle.dump(self.test_pairs, f)
        print(f"Saved test pairs to {self.output_dir}")

    def _compute_tar_at_far(self, scores: torch.Tensor, labels: torch.Tensor, target_far: float = 0.001) -> dict:
        """
        计算 TAR @ FAR ≤ target_far，并给出95%置信区间
        """
        scores = scores.cpu()
        labels = labels.cpu()

        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]

        if len(neg_scores) == 0 or len(pos_scores) == 0:
            return {"TAR": 0.0, "threshold": float('inf'), "actual_FAR": 0.0, "TAR_95CI_low": 0.0, "TAR_95CI_high": 0.0}

        neg_sorted, _ = torch.sort(neg_scores, descending=True)
        n_neg = len(neg_sorted)

        max_allowed_fp = int(target_far * n_neg)  # 允许的最大假正例数（向下取整）

        if max_allowed_fp == 0:
            threshold = neg_sorted[0].item() if n_neg > 0 else float('inf')
            actual_far = 0.0
        else:
            threshold = neg_sorted[max_allowed_fp - 1].item()
            actual_far = max_allowed_fp / n_neg

        tar = (pos_scores >= threshold).float().mean().item()

        n_pos = len(pos_scores)
        successes = int((pos_scores >= threshold).sum())
        ci_low, ci_high = proportion_confint(successes, n_pos, alpha=0.05, method='wilson')

        return {
            "TAR": tar,
            "threshold": threshold,
            "actual_FAR": actual_far,
            "TAR_95CI_low": ci_low,
            "TAR_95CI_high": ci_high
        }

    @torch.no_grad()
    def get_eer_from_register(self):
        scores, labels = [], []
        vc_files = list(self.vc_embeddings.keys())
        genuine_trials = 0
        impostor_trials = 0
        for vc_file in tqdm(vc_files, desc="Computing scores against register embeddings"):
            vc_emb = self.vc_embeddings[vc_file].to(self.encoder.device)
            src_id, _ = self._parse_speaker_ids(vc_file)
            for spk_id, reg_emb in self.orig_embeddings.items():
                reg_emb = reg_emb.to(self.encoder.device)
                if self.plda is not None:
                    score = self.plda(reg_emb.unsqueeze(0).to(self.device), vc_emb.unsqueeze(0).to(self.device)).item()
                else:
                    score = F.cosine_similarity(vc_emb, reg_emb, dim=0).item()
                scores.append(score)
                if spk_id == src_id:
                    labels.append(1)
                    genuine_trials += 1
                else:
                    labels.append(0)
                    impostor_trials += 1
        scores = torch.tensor(scores)
        labels = torch.tensor(labels)

        # ----------------- 计算主 EER 和 ROC -----------------
        scores_np = scores.cpu().numpy().ravel()
        labels_np = labels.cpu().numpy().astype(np.int64).ravel()
        pairs = np.zeros(len(scores_np), dtype=[('score', 'f8'), ('label', 'i4')])
        pairs['score'] = scores_np
        pairs['label'] = labels_np
        # ----------------- EER 的 bootstrap CI（完全 vectorized） -----------------
        def eer_from_scores(scores, labels):
            idx = np.argsort(scores)[::-1]
            labels = labels[idx]

            P = np.sum(labels == 1)
            N = np.sum(labels == 0)

            fp = np.cumsum(labels == 0)
            tp = np.cumsum(labels == 1)

            far = fp / N
            frr = 1 - tp / P

            i = np.argmin(np.abs(far - frr))
            return far[i]

        def compute_eer_bootstrap(pairs):
            scores = pairs['score']
            labels = pairs['label']
            return eer_from_scores(scores, labels)


        def eer_threshold_from_scores(scores, labels):
            idx = np.argsort(scores)[::-1]
            scores = scores[idx]
            labels = labels[idx]

            P = np.sum(labels == 1)
            N = np.sum(labels == 0)

            fp = np.cumsum(labels == 0)
            tp = np.cumsum(labels == 1)

            far = fp / (N + 1e-12)
            frr = 1.0 - tp / (P + 1e-12)

            i = np.argmin(np.abs(far - frr))
            return scores[i]

        boot_res = bootstrap(
            (pairs,),
            compute_eer_bootstrap,
            vectorized=False,
            n_resamples=100,
            method='percentile',
            random_state=42
        )
        eer = eer_from_scores(scores_np, labels_np)
        eer_threshold = eer_threshold_from_scores(scores_np, labels_np)

        eer_ci_low, eer_ci_high = boot_res.confidence_interval
        print(f"EER: {eer:.4f} (95% CI: [{eer_ci_low:.4f}, {eer_ci_high:.4f}])")

        # ----------------- 计算 TAR@FAR -----------------
        tar_far_0_1 = self._compute_tar_at_far(scores, labels, target_far=0.001)
        tar_far_0_01 = self._compute_tar_at_far(scores, labels, target_far=0.0001)
        print(f"TAR @ FAR=0.1%: {tar_far_0_1['TAR']:.4f} (95% CI: [{tar_far_0_1['TAR_95CI_low']:.4f}, {tar_far_0_1['TAR_95CI_high']:.4f}])")
        print(f"TAR @ FAR=0.01%: {tar_far_0_01['TAR']:.4f} (95% CI: [{tar_far_0_01['TAR_95CI_low']:.4f}, {tar_far_0_01['TAR_95CI_high']:.4f}])")

        # ----------------- 计算 Cllr / minCllr -----------------
        scores_np_2d = scores_np.reshape(-1, 1)
        labels_np_int = labels_np.astype(np.int64)

        lr_model = LogisticRegression(solver='lbfgs')
        lr_model.fit(scores_np_2d, labels_np_int)
        loglr = lr_model.decision_function(scores_np_2d)
        s_tar = loglr[labels_np_int == 1]
        s_non = loglr[labels_np_int == 0]
        cllr = (np.mean(np.log1p(np.exp(-s_tar))) + np.mean(np.log1p(np.exp(s_non)))) / (2 * np.log(2))
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        calibrated_probs = iso_reg.fit_transform(scores_np, labels_np_int)
        eps = 1e-6
        calibrated_probs = np.clip(calibrated_probs, eps, 1 - eps)
        calibrated_loglr = logit(calibrated_probs)
        c_tar = calibrated_loglr[labels_np_int == 1]
        c_non = calibrated_loglr[labels_np_int == 0]
        mincllr = (np.mean(np.log1p(np.exp(-c_tar))) + np.mean(np.log1p(np.exp(c_non)))) / (2 * np.log(2))
        print(f"Cllr: {cllr:.4f}, minCllr: {mincllr:.4f}")

        # Cllr 和 minCllr 的 CI（vectorized=False，但 200 次很快）
        def compute_cllr(pairs):
            sl_scores_np = pairs['score'].reshape(-1, 1)
            sl_labels_np = pairs['label'].astype(np.int64)
            lr = LogisticRegression(solver='lbfgs')
            lr.fit(sl_scores_np, sl_labels_np)
            sl_loglr = lr.decision_function(sl_scores_np)
            sl_tar = sl_loglr[sl_labels_np == 1]
            sl_non = sl_loglr[sl_labels_np == 0]
            return ( np.mean(np.log1p(np.exp(-sl_tar))) +
                        np.mean(np.log1p(np.exp(sl_non)))
                    ) / (2 * np.log(2))

        def compute_mincllr(pairs):
            sl_scores_np = pairs['score'].ravel()
            sl_labels_np = pairs['label'].astype(np.int64)
            iso = IsotonicRegression(out_of_bounds='clip')
            cal_probs = iso.fit_transform(sl_scores_np, sl_labels_np)
            cal_probs = np.clip(cal_probs, eps, 1 - eps)
            cal_loglr = logit(cal_probs)
            c_tar = cal_loglr[sl_labels_np == 1]
            c_non = cal_loglr[sl_labels_np == 0]
            return (np.mean(np.log1p(np.exp(-c_tar))) + np.mean(np.log1p(np.exp(c_non)))) / (2 * np.log(2))

        boot_cllr = bootstrap(
            (pairs,),
            compute_cllr,
            vectorized=False,
            n_resamples=100,
            method='percentile',
            random_state=42
        )
        boot_mincllr = bootstrap(
            (pairs,),
            compute_mincllr,
            vectorized=False,
            n_resamples=100,
            method='percentile',
            random_state=42
        )
        cllr_ci_low, cllr_ci_high = boot_cllr.confidence_interval
        mincllr_ci_low, mincllr_ci_high = boot_mincllr.confidence_interval

        print(f"Cllr: {cllr:.4f} (95% CI: [{cllr_ci_low:.4f}, {cllr_ci_high:.4f}])")
        print(f"minCllr: {mincllr:.4f} (95% CI: [{mincllr_ci_low:.4f}, {mincllr_ci_high:.4f}])")

        results = {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "eer_95CI_low": eer_ci_low,
            "eer_95CI_high": eer_ci_high,
            "genuine_trials": genuine_trials,
            "impostor_trials": impostor_trials,
            "TAR@FAR=0.1%": tar_far_0_1["TAR"],
            "threshold@FAR=0.1%": tar_far_0_1["threshold"],
            "actual_FAR@0.1%": tar_far_0_1["actual_FAR"],
            "TAR_95CI_low@FAR=0.1%": tar_far_0_1["TAR_95CI_low"],
            "TAR_95CI_high@FAR=0.1%": tar_far_0_1["TAR_95CI_high"],
            "TAR@FAR=0.01%": tar_far_0_01["TAR"],
            "threshold@FAR=0.01%": tar_far_0_01["threshold"],
            "actual_FAR@0.01%": tar_far_0_01["actual_FAR"],
            "TAR_95CI_low@FAR=0.01%": tar_far_0_01["TAR_95CI_low"],
            "TAR_95CI_high@FAR=0.01%": tar_far_0_01["TAR_95CI_high"],
            "Cllr": cllr,
            "Cllr_95CI_low": cllr_ci_low,
            "Cllr_95CI_high": cllr_ci_high,
            "minCllr": mincllr,
            "minCllr_95CI_low": mincllr_ci_low,
            "minCllr_95CI_high": mincllr_ci_high
        }
        return results

    def plot_tsne(self, selected_spkids, perplexity=10, random_state=42):
        embeddings = []
        labels = []
        types = []
        for spkid in selected_spkids:
            if spkid in self.orig_embeddings:
                embeddings.append(self.orig_embeddings[spkid].cpu().numpy())
                labels.append(spkid)
                types.append("orig")
           
            for vc_file, vc_emb in self.vc_embeddings.items():
                src_id, _ = self._parse_speaker_ids(vc_file)
                if src_id == spkid:
                    embeddings.append(vc_emb.cpu().numpy())
                    labels.append(spkid)
                    types.append("vc")
        if not embeddings:
            print("未找到匹配的 spkid 数据！")
            return
        embeddings = np.vstack(embeddings)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        unique_spkids = set(labels)
        colors = plt.cm.get_cmap("tab10", len(unique_spkids))
        markers = {"orig": "o", "vc": "^"}
        for i, spkid in enumerate(unique_spkids):
            indices_vc = [j for j, lbl in enumerate(labels) if lbl == spkid and types[j] == "vc"]
            if indices_vc:
                plt.scatter(embeddings_2d[indices_vc, 0], embeddings_2d[indices_vc, 1],
                            label=f"{spkid} (vc)", alpha=0.7,
                            color=colors(i), marker=markers["vc"], zorder=2)
        for i, spkid in enumerate(unique_spkids):
            indices_orig = [j for j, lbl in enumerate(labels) if lbl == spkid and types[j] == "orig"]
            if indices_orig:
                plt.scatter(embeddings_2d[indices_orig, 0], embeddings_2d[indices_orig, 1],
                            label=f"{spkid} (orig)", alpha=1.0,
                            color=colors(i), marker=markers["orig"],
                            edgecolors="black", linewidth=1.5, zorder=3)
        plt.title("t-SNE Visualization of Speaker Embeddings")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.savefig(os.path.join(self.output_dir, f"tsne_plot_{self.num_orig_samples}.png"))

    def run(self):
        self.compute_vc_embeddings()
        self.compute_orig_embeddings()
       
        accuracies = self.compute_topk_accuracy()
        print(f"Top-K Accuracies with CI: {accuracies}")
        # self.plot_tsne(selected_spkids=sorted(list(self.orig_embeddings.keys()))[:10], perplexity=10, random_state=42)
        # self.construct_test_pairs()
        perf = self.get_eer_from_register() # 返回一个 dict，包含 EER 和 TAR@FAR 结果
        print("\n[Performance Summary]")
        print(f"EER = {perf['eer']*100:.2f}% @ threshold = {perf['eer_threshold']:.4f}, 95% CI: {perf['eer_95CI_low']*100:.2f}%–{perf['eer_95CI_high']*100:.2f}%")
        print(f"Total genuine trials: {perf['genuine_trials']}, impostor trials: {perf['impostor_trials']}")
        print(f"FAR = 0.10% (actual {perf['actual_FAR@0.1%']:.4f}) → threshold ≈ {perf['threshold@FAR=0.1%']:.4f}, "
              f"TAR = {perf['TAR@FAR=0.1%']*100:.2f}%, 95% CI ≈ {perf['TAR_95CI_low@FAR=0.1%']*100:.2f}%–{perf['TAR_95CI_high@FAR=0.1%']*100:.2f}%")
        print(f"FAR = 0.01% (actual {perf['actual_FAR@0.01%']:.4f}) → threshold ≈ {perf['threshold@FAR=0.01%']:.4f}, "
              f"TAR = {perf['TAR@FAR=0.01%']*100:.2f}%, 95% CI ≈ {perf['TAR_95CI_low@FAR=0.01%']*100:.2f}%–{perf['TAR_95CI_high@FAR=0.01%']*100:.2f}%")
        print(f"Cllr = {perf['Cllr']:.4f}, 95% CI: {perf['Cllr_95CI_low']:.4f}–{perf['Cllr_95CI_high']:.4f}")
        print(f"minCllr = {perf['minCllr']:.4f}, 95% CI: {perf['minCllr_95CI_low']:.4f}–{perf['minCllr_95CI_high']:.4f}")

        test_results = {
            "model": self.args.model,
            "model_src": self.args.model_src,
            "vc_folder": self.args.vc_folder,
            "orig_folder": self.args.orig_folder,
            "output_folder": self.args.output_dir,
            "phone_channel": self.args.phone_channel,
            "channel": self.channel,
            "channel_enhance": self.channel_enhance,
            "num_orig_samples": self.num_orig_samples,
            "add_noise": self.add_noise,
            "noise_file": self.noise_file,
            "accuracies": accuracies,
            **perf # 把所有性能指标字典一起写入
        }
        enhance_str = "with" if self.channel_enhance else "without"
        with open(os.path.join(self.output_dir, f"test_info_{self.num_orig_samples}_{enhance_str}_channel_enhance.json"), "w", encoding="utf-8") as f:
            json.dump(to_json_serializable(test_results), f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Voice Conversion Testing Script")
    parser.add_argument('--model', type=str, default='vctrace', choices=['vctrace', 'revelio', 'midlayer', 'cutmix', 'split_cutmix', 'ecapa'], help="VCTrace model mode")
    parser.add_argument('--model_src', type=str, required=True, help="Path to model checkpoint folder")
    parser.add_argument('--vc_folder', type=str, required=True, help="Path to voice-converted audio folder")
    parser.add_argument('--orig_folder', type=str, required=True, help="Path to original audio folder")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to output directory for saving/loading results")
    parser.add_argument('--resume', action='store_true', help="Resume from saved embeddings and test pairs")
    parser.add_argument('--phone_channel', action='store_true', help="Test with simulated phone channel")
    parser.add_argument('--batch_size', type=int, default=80, help="Batch size for processing")
    parser.add_argument('--sample_rate', type=int, default=16000, help="Target sample rate for audio")
    parser.add_argument('--duration', type=int, default=6, help="Target duration in seconds for audio")
    parser.add_argument('--num_orig_samples', type=int, default=20, help="Number of original samples per speaker for embedding and testing")
    parser.add_argument('--channel', type=str, default='G.711', choices=['G.711', 'G.722', 'GSM-FR', 'AMR', 'AMR-WB', 'Opus-WB'], help="Channel type for simulation")
    parser.add_argument('--channel_enhance', action='store_true', help="Use channel enhancement")
    parser.add_argument('--noise_file', type=str, default=None, help="Noise file for simulation")
    parser.add_argument('--add_noise', action='store_true', help="Add noise to the audio")
    parser.add_argument('--plda', action='store_true', help="Use PLDA scoring or not")
    parser.add_argument('--w_en', type=float, default=4.0, help="Weight for enrollment embedding")
    args = parser.parse_args()
    max_key_len = max(len(k) for k in vars(args))
    for k, v in vars(args).items():
        print(f"{k.ljust(max_key_len)} : {v}")
    model_src = args.model_src
    hparams_src = 'inference.yaml'
    if args.model == "vctrace":
        encoder = VCTraceEncoder.from_hparams(source=model_src, 
                                            hparams_file=hparams_src, 
                                            savedir=model_src,
                                            freeze_params=True, 
                                            run_opts={"device": device, "data_parallel_backend": True})
    elif args.model == "revelio":
        encoder = RevelioEncoder.from_hparams(source=model_src, 
                                            hparams_file=hparams_src, 
                                            savedir=model_src,
                                            freeze_params=True, 
                                            run_opts={"device": device, "data_parallel_backend": True})
    elif args.model == "midlayer":
        encoder = MidlayerEncoder.from_hparams(source=model_src, 
                                            hparams_file=hparams_src, 
                                            savedir=model_src,
                                            freeze_params=True, 
                                            run_opts={"device": device, "data_parallel_backend": True})
    elif args.model == "cutmix":
        encoder = MidlayerCutMixEncoder.from_hparams(source=model_src, 
                                            hparams_file=hparams_src, 
                                            savedir=model_src,
                                            freeze_params=True, 
                                            run_opts={"device": device, "data_parallel_backend": True})
    elif args.model == "split_cutmix":
        encoder = SplitEncoder.from_hparams(source=model_src, 
                                            hparams_file=hparams_src, 
                                            savedir=model_src,
                                            freeze_params=True, 
                                            run_opts={"device": device, "data_parallel_backend": True})
    elif args.model == "ecapa":
        encoder = EcapaEncoder.from_hparams(source=model_src, 
                                            hparams_file=hparams_src, 
                                            savedir=model_src,
                                            freeze_params=True, 
                                            run_opts={"device": device, "data_parallel_backend": True})
    else:
        raise ValueError("Unknown model mode!")
    
    tester = Tester(
        encoder=encoder,
        args=args,
        vc_folder=args.vc_folder,
        orig_folder=args.orig_folder,
        output_dir=args.output_dir,
        resume=args.resume,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        duration=args.duration,
        phone_channel=args.phone_channel,
        num_orig_samples=args.num_orig_samples,
        channel=args.channel,
        channel_enhance=args.channel_enhance,
        noise_file=args.noise_file,
        add_noise=args.add_noise,
        plda=args.plda
    )
    tester.run()

if __name__ == "__main__":
    main()
