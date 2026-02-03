#!/usr/bin/env python3
"""Recipe for training a speaker-id system. The template can use used as a
basic example for any signal classification task such as language_id,
emotion recognition, command classification, etc. The proposed task classifies
28 speakers using Mini Librispeech. This task is very easy. In a real
scenario, you need to use datasets with a larger number of speakers such as
the voxceleb one (see recipes/VoxCeleb). Speechbrain has already some built-in
models for signal classifications (see the ECAPA one in
speechbrain.lobes.models.ECAPA_TDNN.py or the xvector in
speechbrain/lobes/models/Xvector.py)

To run this recipe, do the following:
> python train.py train.yaml

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Mirco Ravanelli 2021
"""
import os
import sys
import torch

import torchaudio
import torch.nn.functional as F
import random
import speechbrain as sb
import random
import swanlab
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm
from utils.loss import MaxMarginLoss, CircleLoss, convert_label_to_similarity, convert_crossmodal_similarity
from torchaudio.functional import resample
from torch import nn
from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech
from speechbrain.utils.parameter_transfer import Pretrainer
from utils.aug import CutMixAudioAugmenter


def build_cache_from_csv(vc_csv_path,src_1_csv_path, src_2_csv_path):
    print("📦 构建缓存中...")
    vc_df = pd.read_csv(vc_csv_path)
    src_1_df = pd.read_csv(src_1_csv_path)
    src_2_df = pd.read_csv(src_2_csv_path)
    cache = {"src":defaultdict(list),"vc":defaultdict(list)}
    for _, row in tqdm(vc_df.iterrows(),total=vc_df.shape[0]):
        cache["vc"][str(row['src_id'])].append(row['vc_audio'])
    for _, row in tqdm(src_1_df.iterrows(),total=src_1_df.shape[0]):
        cache["src"][str(row['src_id'])].append(row['vc_audio'])
    for _, row in tqdm(src_2_df.iterrows(),total=src_2_df.shape[0]):
        cache["src"][str(row['src_id'])].append(row['vc_audio'])
    # Limit to 200 entries per speaker
    for src_id in cache["vc"]:
        if len(cache["vc"][src_id]) > 200:
            cache["vc"][src_id] = random.sample(cache["vc"][src_id], 200)
    for src_id in cache["src"]:
        if len(cache["src"][src_id]) > 200:
            cache["src"][src_id] = random.sample(cache["src"][src_id], 200)

    # 持久化保存
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print("✅ 缓存构建完成并保存到本地。")
    return cache

def load_or_build_cache():
    if os.path.exists(CACHE_FILE):
        print("🔄 正在从本地加载缓存...")
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        print("✅ 加载完成。")
    else:
        cache = build_cache_from_csv(CSV_VC_FILE,CSV_VOXCE_FILE,CSV_LIBRI_FILE)
    return cache

def get_random_audio(spk_id, cache, subset):
    if spk_id not in cache[subset] or not cache[subset][spk_id]:
        return None
    return random.choice(cache[subset][spk_id])


def pad_to_target_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = tensor.shape[1]
    
    if current_length >= target_length:
        return tensor[:, :target_length]
    repeat_times = target_length // current_length
    remaining = target_length % current_length
    expanded_tensor = tensor.repeat(1, repeat_times)
    if remaining > 0:
        expanded_tensor = torch.cat([expanded_tensor, tensor[:, :remaining]], dim=1)

    return expanded_tensor

def probability_true(probability: float) -> bool:
    return random.random() < probability

# Brain class for speech enhancement training
class SpkIdBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        swanlab.init(
                project=self.hparams.wandb_project,
                workspace=self.hparams.wandb_name,
                # resume=True,
                # id=self.hparams.wandb_id
                mode="offline"
            )
        pretrain = Pretrainer(
            loadables = {'model':self.modules.extractor},
            paths = {'model':'./spkrec-ecapa-voxceleb/embedding_model.ckpt'}
        )
        pretrain.collect_files()
        pretrain.load_collected()
        print("loaded ecapa from huggingface successfully!!")
        self.temperature = 4.0
        self.cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.1)
        self.cir_loss = CircleLoss(m=0.15, gamma=40)
        self.cutmixaug = CutMixAudioAugmenter(30000,15000,self.hparams.sample_rate*6,0.3,self.device)

    # def init_optimizers(self):
    #     """Called during ``on_fit_start()``, initialize optimizers
    #     after parameters are fully configured (e.g. DDP, jit).

    #     The default implementation of this method depends on an optimizer
    #     class being passed at initialization that takes only a list
    #     of parameters (e.g., a lambda or a partial function definition).
    #     This creates a single optimizer that optimizes all trainable params.

    #     Override this class if there are multiple optimizers.
    #     """
    #     print(f"Init optaimizer......")
    #     all_params = list(self.modules.midlayer.parameters()) + list(self.modules.classifier.parameters())
    #     self.modules.extractor.requires_grad_(False)
    #     if self.opt_class is not None:
    #         if self.remove_vector_weight_decay:
    #             all_params = rm_vector_weight_decay(self.modules)

    #         self.optimizer = self.opt_class(all_params)

    #         self.optimizers_dict = {"opt_class": self.optimizer}

    #         if self.checkpointer is not None:
    #             self.checkpointer.add_recoverable("optimizer", self.optimizer)
    #     self.print_trainable_parameters()

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : torch.Tensor
            torch.Tensor that contains the posterior probabilities over the N classes.
        """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings, and predictions
        vc_feats, vc_lens = self.prepare_features(batch.vc_sig, batch.vc2_sig, stage)
        src_feats, src_lens = self.prepare_features(batch.src_sig, batch.src2_sig, stage)
        # tar_feats, tar_lens = self.prepare_features(batch.tar_sig, None, stage)
        src_voice_prints = self.modules.extractor(src_feats, src_lens)
        # tar_mid_feats    = self.modules.extractor.module.extract_feature(tar_feats, tar_lens)
        vc_mid_feats   = self.modules.extractor.module.extract_feature(vc_feats, vc_lens)
        vc_embeddings  = self.modules.midlayer(vc_mid_feats)
        # tar_embeddings = self.modules.midlayer(tar_mid_feats)
        src_predictions = self.modules.classifier(src_voice_prints)
        vc_predictions  = self.modules.classifier(vc_embeddings)
        # tar_predictions = self.modules.classifier(tar_embeddings)
        predictions = (vc_predictions, src_predictions) #,tar_predictions)
        return predictions, vc_embeddings, src_voice_prints

    def prepare_features(self, wavs, wav2s, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.

        Returns
        -------
        feats : torch.Tensor
            The prepared features.
        lens : torch.Tensor
            The lengths of the corresponding prepared features.
        """
        wavs, lens = wavs
        if wav2s is not None:
            wav2s, len2s = wav2s

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            if probability_true(0.5) and wav2s is not None:
                wavs, lens = self.cutmixaug(wavs, wav2s, lens)
            wavs, lens = self.hparams.wav_augment(wavs, lens)
        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        return feats, lens

    def compute_objectives(self, predictions, embeddings, voice_prints, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : torch.Tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        _, lens = batch.vc_sig
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            spkid = self.hparams.wav_augment.replicate_labels(spkid)
            lens = self.hparams.wav_augment.replicate_labels(lens)
        # Compute the cost function
        labels = spkid.view(-1)
        loss_cls = (1-self.hparams.gamma) * self.hparams.compute_cost_vc(predictions[0], spkid, lens) + \
            self.hparams.gamma * self.cir_loss(*convert_label_to_similarity(embeddings.squeeze(1),labels))
        loss_ref = (1-self.hparams.gamma) * self.hparams.compute_cost_ref(predictions[1], spkid, lens) + \
            self.hparams.gamma * self.cir_loss(*convert_label_to_similarity(voice_prints.squeeze(1),labels))

        # vps = nn.functional.normalize(torch.cat([embeddings.squeeze(1), voice_prints.squeeze(1)], dim=0))
        # labels = torch.cat([spkid,spkid], dim=0).view(-1)
        # loss_cir = self.cir_loss(*convert_label_to_similarity(vps,labels))
        # Cosine Konwledge distillation Loss
        # loss_ctr = self.cir_loss(*convert_crossmodal_similarity(embeddings.squeeze(1),voice_prints.squeeze(1),labels))
        loss_ctr = self.cos_loss(embeddings.squeeze(1),voice_prints.squeeze(1),torch.ones(embeddings.shape[0]).to(embeddings.device))
        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions[0], spkid, lens, reduction="batch"
        )
        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics_vc.append(batch.id, predictions[0], spkid, lens)
            self.error_metrics_ref.append(batch.id, predictions[1], spkid, lens)
        loss = (loss_cls)*self.hparams.alpha + (loss_ref)*self.hparams.beta + loss_ctr*self.hparams.delta
        if stage == sb.Stage.TRAIN:
            swanlab.log({
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "loss_cls": loss_cls.item(),
                "loss_ref": loss_ref.item(),
                "loss_ctr": loss_ctr.item(),
                # "loss_kd": loss_kd.item(),
                # "loss_tar": loss_tar.item(),
                "total_loss": loss.item(),
                "epoch": self.hparams.epoch_counter.current
            }) 
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics_vc = self.hparams.error_stats()
            self.error_metrics_ref = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_vc": self.error_metrics_vc.summarize("average"),
                "error_ref": self.error_metrics_ref.summarize("average"),
            }
        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
        swanlab.log({
            f"{stage.name.lower()}_loss": stage_loss,
            "epoch": epoch
        })
        if stage == sb.Stage.TEST:
            swanlab.finish()

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        amp = sb.core.AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step)

        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type
                ):
                    outputs, embeddings, voice_prints = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, embeddings, voice_prints, batch, sb.Stage.TRAIN
                    )
            else:
                outputs, embeddings, voice_prints = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(
                    outputs, embeddings, voice_prints, batch, sb.Stage.TRAIN
                )

            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()

        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        self.hparams.lr_annealing.on_batch_end(self.optimizer)
        return loss.detach().cpu()

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        amp = sb.core.AMPConfig.from_name(self.eval_precision)
        if self.use_amp:
            with torch.autocast(
                dtype=amp.dtype, device_type=torch.device(self.device).type
            ):
                out, embeddings, voice_prints = self.compute_forward(batch, stage=stage)
                loss = self.compute_objectives(out, embeddings, voice_prints, batch, stage=stage)
        else:
            out, embeddings, voice_prints = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, embeddings, voice_prints, batch, stage=stage)
        return loss.detach().cpu()


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(hparams["n_classes"])
    num_frames = hparams["sample_rate"]*6
    # Define audio pipeline
    @sb.utils.data_pipeline.takes("vc_audio")
    @sb.utils.data_pipeline.provides("vc_sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`.
        """
        sig,fs = torchaudio.load(wav)
        # if probability_true(0.2):
            # sig,fs = simulate_telephone_audio_torch(sig, fs)
        sig = resample(sig, fs, int(hparams["sample_rate"]))
        if sig.size(-1)>num_frames:
            start = random.randint(0, sig.size(-1)-num_frames)
            sig = sig[:, start:start+num_frames]
        if sig.size(-1)<num_frames:
            sig = pad_to_target_length(sig,num_frames)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    @sb.utils.data_pipeline.takes("src_id")
    @sb.utils.data_pipeline.provides("vc2_sig")
    def vc2_audio_pipeline(src_id):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`.
        """
        wav = get_random_audio(str(src_id),cache,"vc")
        sig,fs = torchaudio.load(wav)
        # if probability_true(0.2):
            # sig,fs = simulate_telephone_audio_torch(sig, fs)
        sig = resample(sig, fs, int(hparams["sample_rate"]))
        if sig.size(-1)>num_frames:
            start = random.randint(0, sig.size(-1)-num_frames)
            sig = sig[:, start:start+num_frames]
        if sig.size(-1)<num_frames:
            sig = pad_to_target_length(sig,num_frames)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    @sb.utils.data_pipeline.takes("src_audio")
    @sb.utils.data_pipeline.provides("src_sig")
    def src_audio_pipeline(wav):
        sig,fs = torchaudio.load(wav)
        # if probability_true(0.2):
            # sig,fs = simulate_telephone_audio_torch(sig, fs)
        sig = resample(sig, fs, int(hparams["sample_rate"]))
        if sig.size(-1)>num_frames:
            start = random.randint(0, sig.size(-1)-num_frames)
            sig = sig[:, start:start+num_frames]
        if sig.size(-1)<num_frames:
            sig = pad_to_target_length(sig,num_frames)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    # @sb.utils.data_pipeline.takes("tar_audio")
    # @sb.utils.data_pipeline.provides("tar_sig")
    # def tar_audio_pipeline(wav):
    #     sig,fs = torchaudio.load(wav)
    #     # if probability_true(0.2):
    #         # sig,fs = simulate_telephone_audio_torch(sig, fs)
    #     sig = resample(sig, fs, int(hparams["sample_rate"]))
    #     if sig.size(-1)>num_frames:
    #         start = random.randint(0, sig.size(-1)-num_frames)
    #         sig = sig[:, start:start+num_frames]
    #     if sig.size(-1)<num_frames:
    #         sig = pad_to_target_length(sig,num_frames)
    #     sig = sig.transpose(0, 1).squeeze(1)
    #     return sig


    @sb.utils.data_pipeline.takes("src_id")
    @sb.utils.data_pipeline.provides("src2_sig")
    def src2_audio_pipeline(src_id):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`.
        """
        wav = get_random_audio(str(src_id),cache,"src")
        sig,fs = torchaudio.load(wav)
        # if probability_true(0.2):
            # sig,fs = simulate_telephone_audio_torch(sig, fs)
        sig = resample(sig, fs, int(hparams["sample_rate"]))
        if sig.size(-1)>num_frames:
            start = random.randint(0, sig.size(-1)-num_frames)
            sig = sig[:, start:start+num_frames]
        if sig.size(-1)<num_frames:
            sig = pad_to_target_length(sig,num_frames)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("src_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        """Defines the pipeline to process the input speaker label."""
        spk_id = str(spk_id)
        yield spk_id
        spk_id_encoded = label_encoder.encode_label_torch(spk_id)
        yield spk_id_encoded

    # # Define label pipeline:
    # @sb.utils.data_pipeline.takes("tar_id")
    # @sb.utils.data_pipeline.provides("tar_id", "tar_id_encoded")
    # def tar_label_pipeline(spk_id):
    #     spk_id = str(spk_id)
    #     """Defines the pipeline to process the input speaker label."""
    #     yield spk_id
    #     spk_id_encoded = label_encoder.encode_label_torch(spk_id)
    #     yield spk_id_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"]
    }
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, vc2_audio_pipeline, src_audio_pipeline,src2_audio_pipeline, label_pipeline],
            output_keys=["id", "vc_sig","vc2_sig", "src_sig", "src2_sig", "spk_id_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="spk_id",
    )
    return datasets


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    run_opts["eval_precision"]="fp16"
    run_opts["precision"]="fp16"
    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    CACHE_FILE = hparams["data_cache"]
    CSV_VC_FILE = hparams["vc_annotation"]
    CSV_VOXCE_FILE = hparams["voxceleb_annotation"]
    CSV_LIBRI_FILE = hparams["librispeech_annotation"]
    cache = load_or_build_cache()
    cache_vc = cache["vc"]
    cache_src = cache["src"]
    print(f"Total Speaker:{len(cache_vc)}")
    print(f"Total Speaker:{len(cache_src)}")

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])

    # Initialize the Brain object to prepare for mask training.
    spk_id_brain = SpkIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    spk_id_brain.fit(
        epoch_counter=spk_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
