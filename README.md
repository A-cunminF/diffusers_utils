# diffusers_utils

## Useage
```shell
python convert_diffusers_lora_bin_to_webui.py \
  --lora_bin_path=/PATH/TO/pytorch_lora_weights.bin \
  --output_path=/PATH/TO/pytorch_lora_weights.safetensors
  
python convert_diffusers_lora_safetensors_to_bin.py \
  --sd_model_path=/PATH/TO/STABLE/DIFFUSION/MODEL/WHEN/YOU/TRAIN/THIS/LORA \
  --lora_safetensors_path=/PATH/TO/pytorch_lora_weights.bin \
  --output_path=/PATH/TO/pytorch_lora_weights.safetensors
```
