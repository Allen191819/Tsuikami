result_dir="results/eval_result/overall/"
cuda_divce="1"
batch_size=160


vc_model="test_ckpt/CKPT+2025-10-19+09-58-03+00_circle_diffhvc"
vc_audio="Dataset/audio/Diff_VC_Librispeech_test_clean"
CUDA_VISIBLE_DEVICES=$cuda_divce python test_tsuikami.py --model cutmix --model_src $vc_model  --vc_folder $vc_audio --orig_folder Dataset/audio/LibriSpeech/LibriSpeech_test-clean_40spk_flac --output_dir $result_dir"DiffH_VC_Circle" --num_orig_samples 5 --resume --batch_size $batch_size
 
vc_model="test_ckpt/CKPT+2025-10-31+10-07-52+00_circle_freevc"
vc_audio="Dataset/audio/Free_VC_Librispeech_test_clean"
CUDA_VISIBLE_DEVICES=$cuda_divce python test_tsuikami.py --model cutmix --model_src $vc_model  --vc_folder $vc_audio --orig_folder Dataset/audio/LibriSpeech/LibriSpeech_test-clean_40spk_flac --output_dir $result_dir"Free_VC_Circle" --num_orig_samples 5 --resume --batch_size $batch_size
  
vc_model="test_ckpt/CKPT+2025-11-07+10-23-03+00_circle_againvc"
vc_audio="Dataset/audio/Again_test_40spk/wav"
CUDA_VISIBLE_DEVICES=$cuda_divce python test_tsuikami.py --model cutmix --model_src $vc_model  --vc_folder $vc_audio --orig_folder Dataset/audio/LibriSpeech/LibriSpeech_test-clean_40spk_flac --output_dir $result_dir"Again_VC_Circle" --num_orig_samples 5 --resume --batch_size $batch_size

vc_model="test_ckpt/CKPT+2025-10-17+13-14-12+00_circle_dddmvc"
vc_audio="Dataset/audio/DDDM_VC_Librispeech_test_clean"
CUDA_VISIBLE_DEVICES=$cuda_divce python test_tsuikami.py --model cutmix --model_src $vc_model  --vc_folder $vc_audio --orig_folder Dataset/audio/LibriSpeech/LibriSpeech_test-clean_40spk_flac --output_dir $result_dir"DDDM_VC_Circle" --num_orig_samples 5 --resume --batch_size $batch_size
 
vc_model="test_ckpt/CKPT+2025-09-29+22-45-06+00_circle_triaanvc"
vc_audio="Dataset/audio/TriAANVC_librispeech_test_clean"
CUDA_VISIBLE_DEVICES=$cuda_divce python test_tsuikami.py --model cutmix --model_src $vc_model  --vc_folder $vc_audio --orig_folder Dataset/audio/LibriSpeech/LibriSpeech_test-clean_40spk_flac --output_dir $result_dir"TriANN_VC_Circle" --num_orig_samples 5 --resume --batch_size $batch_size
