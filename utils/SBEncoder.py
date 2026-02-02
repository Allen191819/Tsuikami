import os
import torch
import torchaudio
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from speechbrain.inference.interfaces import Pretrained
from typing import Tuple, List

class VCTraceEncoder(Pretrained):
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "extractor"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def encode_vc_audio(self, wavs, tarwavs, wav_lens=None, tarwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        if len(tarwavs.shape) == 1:
            tarwavs = tarwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        if tarwav_lens is None:
            tarwav_lens = torch.ones(tarwavs.shape[0], device=self.device)

        feats1 = self.mods.compute_features(wavs.to(self.device))
        feats2 = self.mods.compute_features(tarwavs.to(self.device))
        feats1 = self.mods.mean_var_norm(feats1, wav_lens)
        feats2 = self.mods.mean_var_norm(feats2, tarwav_lens)
        
        embeddings_1 = self.mods.embedding_model.module.inference(feats1, feats2)

        if normalize:
            embeddings_1 = self.hparams.mean_var_norm_emb(
                embeddings_1, torch.ones(embeddings_1.shape[0], device=self.device)
            )
        return embeddings_1.squeeze(1)


    @torch.no_grad()
    def encode_ori_audio(self, srcwavs, srcwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(srcwavs.shape) == 1:
            srcwavs = srcwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if srcwav_lens is None:
            srcwav_lens = torch.ones(srcwavs.shape[0], device=self.device)

        feats3 = self.mods.compute_features(srcwavs.to(self.device))
        feats3 = self.mods.mean_var_norm(feats3, srcwav_lens)
        
        embeddings_2 = self.mods.extractor(feats3, srcwav_lens)

        if normalize:
            embeddings_2 = self.hparams.mean_var_norm_emb(
                embeddings_2, torch.ones(embeddings_2.shape[0], device=self.device)
            )
        return embeddings_2.squeeze(1)

class MidlayerEncoder(Pretrained):
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "extractor",
        "midlayer",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def encode_vc_audio(self, vcwavs, refwavs, vcwav_lens=None, refwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(vcwavs.shape) == 1:
            vcwavs = vcwavs.unsqueeze(0)
        if len(refwavs.shape) == 1:
            refwavs = refwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if vcwav_lens is None:
            vcwav_lens = torch.ones(vcwavs.shape[0], device=self.device)
        if refwav_lens is None:
            refwav_lens = torch.ones(refwavs.shape[0], device=self.device)

        vc_feats = self.mods.compute_features(vcwavs.to(self.device))
        ref_feats = self.mods.compute_features(refwavs.to(self.device))
        vc_feats = self.mods.mean_var_norm(vc_feats, vcwav_lens)
        ref_feats = self.mods.mean_var_norm(ref_feats, refwav_lens)
        
        vc_mid_feats = self.mods.extractor.module.extract_feature(vc_feats, vcwav_lens)
        ref_mid_feats = self.mods.extractor.module.extract_feature(ref_feats, refwav_lens)
        embeddings_2 = self.mods.midlayer(vc_mid_feats,ref_mid_feats)

        if normalize:
            embeddings_2 = self.hparams.mean_var_norm_emb(
                embeddings_2, torch.ones(embeddings_2.shape[0], device=self.device)
            )
        return embeddings_2.squeeze(1)


    @torch.no_grad()
    def encode_ori_audio(self, srcwavs, srcwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(srcwavs.shape) == 1:
            srcwavs = srcwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if srcwav_lens is None:
            srcwav_lens = torch.ones(srcwavs.shape[0], device=self.device)

        src_feats = self.mods.compute_features(srcwavs.to(self.device))
        src_feats = self.mods.mean_var_norm(src_feats, srcwav_lens)
        
        embeddings_1 = self.mods.extractor(src_feats, srcwav_lens)

        if normalize:
            embeddings_1 = self.hparams.mean_var_norm_emb(
                embeddings_1, torch.ones(embeddings_1.shape[0], device=self.device)
            )
        return embeddings_1.squeeze(1)

        
class RevelioEncoder(Pretrained):
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @torch.no_grad()
    def encode_vc_audio(self, wavs, tarwavs, wav_lens=None, tarwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        if len(tarwavs.shape) == 1:
            tarwavs = tarwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        if tarwav_lens is None:
            tarwav_lens = torch.ones(tarwavs.shape[0], device=self.device)
        if (
            hasattr(self.hparams, "use_tacotron2_mel_spec")
            and self.hparams.use_tacotron2_mel_spec
        ):
            feats1 = self.hparams.compute_features(wavs).to(self.device)
            feats2 = self.hparams.compute_features(tarwavs).to(self.device)
            feats1 = torch.transpose(feats1, 1, 2)
            feats2 = torch.transpose(feats2, 1, 2)
        else:
            feats1 = self.mods.compute_features(wavs)
            feats2 = self.mods.compute_features(tarwavs)
        feats1 = self.mods.mean_var_norm(feats1, wav_lens).to(self.device)
        feats2 = self.mods.mean_var_norm(feats2, tarwav_lens).to(self.device)
        
        embeddings = self.mods.embedding_model.module.inference(feats1, feats2)

        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings.squeeze(1)


    @torch.no_grad()
    def encode_ori_audio(self, wavs, wav_lens=None, normalize=False):
        # Manage single waveforms in input
        tarwavs = torch.zeros_like(wavs,device=self.device)
        tarwav_lens = torch.ones(tarwavs.shape[0], device=self.device)
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        if len(tarwavs.shape) == 1:
            tarwavs = tarwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        if (
            hasattr(self.hparams, "use_tacotron2_mel_spec")
            and self.hparams.use_tacotron2_mel_spec
        ):
            feats1 = self.hparams.compute_features(wavs).to(self.device)
            feats2 = self.hparams.compute_features(tarwavs).to(self.device)
            feats1 = torch.transpose(feats1, 1, 2)
            feats2 = torch.transpose(feats2, 1, 2)
        else:
            feats1 = self.mods.compute_features(wavs)
            feats2 = self.mods.compute_features(tarwavs)
        feats1 = self.mods.mean_var_norm(feats1, wav_lens).to(self.device)
        feats2 = self.mods.mean_var_norm(feats2, tarwav_lens).to(self.device)
        
        embeddings = self.mods.embedding_model.module.inference(feats1, feats2)

        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings.squeeze(1)


class MidlayerCutMixEncoder(Pretrained):
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "extractor",
        "midlayer",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def encode_vc_audio(self, vcwavs, refwavs, vcwav_lens=None, refwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(vcwavs.shape) == 1:
            vcwavs = vcwavs.unsqueeze(0)
        # if len(refwavs.shape) == 1:
             # refwavs = refwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if vcwav_lens is None:
            vcwav_lens = torch.ones(vcwavs.shape[0], device=self.device)
        # if refwav_lens is None:
            # refwav_lens = torch.ones(refwavs.shape[0], device=self.device)

        vc_feats = self.mods.compute_features(vcwavs.to(self.device))
        # ref_feats = self.mods.compute_features(refwavs.to(self.device))
        vc_feats = self.mods.mean_var_norm(vc_feats, vcwav_lens)
        # ref_feats = self.mods.mean_var_norm(ref_feats, refwav_lens)
        
        vc_mid_feats = self.mods.extractor.module.extract_feature(vc_feats, vcwav_lens)
        # ref_mid_feats = self.mods.extractor.module.extract_feature(ref_feats, refwav_lens)
        embeddings_2 = self.mods.midlayer(vc_mid_feats)

        if normalize:
            embeddings_2 = self.hparams.mean_var_norm_emb(
                embeddings_2, torch.ones(embeddings_2.shape[0], device=self.device)
            )
        return embeddings_2.squeeze(1)


    @torch.no_grad()
    def encode_ori_audio(self, srcwavs, srcwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(srcwavs.shape) == 1:
            srcwavs = srcwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if srcwav_lens is None:
            srcwav_lens = torch.ones(srcwavs.shape[0], device=self.device)

        src_feats = self.mods.compute_features(srcwavs.to(self.device))
        src_feats = self.mods.mean_var_norm(src_feats, srcwav_lens)
        
        embeddings_1 = self.mods.extractor(src_feats, srcwav_lens)

        if normalize:
            embeddings_1 = self.hparams.mean_var_norm_emb(
                embeddings_1, torch.ones(embeddings_1.shape[0], device=self.device)
            )
        return embeddings_1.squeeze(1)


class SplitEncoder(Pretrained):
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "extractor"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def encode_vc_audio(self, wavs, tarwavs, wav_lens=None, tarwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # if len(tarwavs.shape) == 1:
        #     tarwavs = tarwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        # if tarwav_lens is None:
        #     tarwav_lens = torch.ones(tarwavs.shape[0], device=self.device)

        feats1 = self.mods.compute_features(wavs.to(self.device))
        # feats2 = self.mods.compute_features(tarwavs.to(self.device))
        feats1 = self.mods.mean_var_norm(feats1, wav_lens)
        # feats2 = self.mods.mean_var_norm(feats2, tarwav_lens)
        
        embeddings_1 = self.mods.embedding_model(feats1, wav_lens)

        if normalize:
            embeddings_1 = self.hparams.mean_var_norm_emb(
                embeddings_1, torch.ones(embeddings_1.shape[0], device=self.device)
            )
        return embeddings_1.squeeze(1)


    @torch.no_grad()
    def encode_ori_audio(self, srcwavs, srcwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(srcwavs.shape) == 1:
            srcwavs = srcwavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if srcwav_lens is None:
            srcwav_lens = torch.ones(srcwavs.shape[0], device=self.device)

        feats3 = self.mods.compute_features(srcwavs.to(self.device))
        feats3 = self.mods.mean_var_norm(feats3, srcwav_lens)
        
        embeddings_2 = self.mods.extractor(feats3, srcwav_lens)

        if normalize:
            embeddings_2 = self.hparams.mean_var_norm_emb(
                embeddings_2, torch.ones(embeddings_2.shape[0], device=self.device)
            )
        return embeddings_2.squeeze(1)

class EcapaEncoder(Pretrained):
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @torch.no_grad()
    def encode_vc_audio(self, wavs, tarwavs, wav_lens=None, tarwav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        if (
            hasattr(self.hparams, "use_tacotron2_mel_spec")
            and self.hparams.use_tacotron2_mel_spec
        ):
            feats1 = self.hparams.compute_features(wavs).to(self.device)
            feats1 = torch.transpose(feats1, 1, 2)
        else:
            feats1 = self.mods.compute_features(wavs)
        feats1 = self.mods.mean_var_norm(feats1, wav_lens).to(self.device)
        
        embeddings = self.mods.embedding_model.module(feats1, wav_lens)

        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings.squeeze(1)


    @torch.no_grad()
    def encode_ori_audio(self, wavs, wav_lens=None, normalize=False):
        # Manage single waveforms in input
        tarwavs = torch.zeros_like(wavs,device=self.device)
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        if (
            hasattr(self.hparams, "use_tacotron2_mel_spec")
            and self.hparams.use_tacotron2_mel_spec
        ):
            feats1 = self.hparams.compute_features(wavs).to(self.device)
            feats1 = torch.transpose(feats1, 1, 2)
        else:
            feats1 = self.mods.compute_features(wavs)
        feats1 = self.mods.mean_var_norm(feats1, wav_lens).to(self.device)
        
        embeddings = self.mods.embedding_model.module(feats1,wav_lens)

        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings.squeeze(1)