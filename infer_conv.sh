accelerate launch infer_conv.py --dataset redial --split train --tokenizer ../utils/tokenizer/dialogpt-small --model microsoft/DialoGPT-small --text_tokenizer ../utils/tokenizer/roberta-base --text_encoder roberta-base --n_prefix_conv 20 --prompt_encoder /mnt/wangxiaolei/crs/prompt/dialogpt_prompt-pre_prefix-20_redial_1e-4/final --per_device_eval_batch_size 64 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32
accelerate launch infer_conv.py --dataset redial --split valid --tokenizer ../utils/tokenizer/dialogpt-small --model microsoft/DialoGPT-small --text_tokenizer ../utils/tokenizer/roberta-base --text_encoder roberta-base --n_prefix_conv 20 --prompt_encoder /mnt/wangxiaolei/crs/prompt/dialogpt_prompt-pre_prefix-20_redial_1e-4/final --per_device_eval_batch_size 64 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32
accelerate launch infer_conv.py --dataset redial --split test --tokenizer ../utils/tokenizer/dialogpt-small --model microsoft/DialoGPT-small --text_tokenizer ../utils/tokenizer/roberta-base --text_encoder roberta-base --n_prefix_conv 20 --prompt_encoder /mnt/wangxiaolei/crs/prompt/dialogpt_prompt-pre_prefix-20_redial_1e-4/final --per_device_eval_batch_size 64 --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32