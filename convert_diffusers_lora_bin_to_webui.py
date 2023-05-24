import argparse
import torch
from safetensors.torch import load_file, save_file

def convert_name_to_bin(name):
    
    # down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up
    new_name = name.replace(LORA_PREFIX_UNET+'_', '')
    new_name = new_name.replace('.weight', '')
    
    # ['down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q', 'lora.up']
    parts = new_name.split('.')
    
    #parts[0] = parts[0].replace('_0', '')
    if 'out' in parts[0]:
        parts[0] = "_".join(parts[0].split('_')[:-1])
    parts[1] = parts[1].replace('_', '.')
    
    # ['down', 'blocks', '0', 'attentions', '0', 'transformer', 'blocks', '0', 'attn1', 'to', 'q']
    # ['mid', 'block', 'attentions', '0', 'transformer', 'blocks', '0', 'attn2', 'to', 'out']
    sub_parts = parts[0].split('_')

    # down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q_
    new_sub_parts = ""
    for i in range(len(sub_parts)):
        if sub_parts[i] in ['block', 'blocks', 'attentions'] or sub_parts[i].isnumeric() or 'attn' in sub_parts[i]:
            if 'attn' in sub_parts[i]:
                new_sub_parts += sub_parts[i] + ".processor."
            else:
                new_sub_parts += sub_parts[i] + "."
        else:
            new_sub_parts += sub_parts[i] + "_"
    
    # down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.up
    new_sub_parts += parts[1]
    
    new_name =  new_sub_parts + '.weight'
    
    return new_name


def safetensors_to_bin(safetensor_path, bin_path):
    
    bin_state_dict = {}
    safetensors_state_dict = load_file(safetensor_path)
        
    for key_safetensors in safetensors_state_dict:
        # these if are required  by current diffusers' API
        # remove these may have negative effect as not all LoRAs are used
        if 'text' in key_safetensors:
            continue
        if 'unet' not in key_safetensors:
            continue
        if 'transformer_blocks' not in key_safetensors:
            continue
        if 'ff_net' in key_safetensors or 'alpha' in key_safetensors:
            continue
        key_bin = convert_name_to_bin(key_safetensors)
        bin_state_dict[key_bin] = safetensors_state_dict[key_safetensors]
    
    torch.save(bin_state_dict, bin_path)

    
def convert_name_to_safetensors(name):
    
    # ['down_blocks', '0', 'attentions', '0', 'transformer_blocks', '0', 'attn1', 'processor', 'to_q_lora', 'up', 'weight']
    parts = name.split('.')
    
    # ['down_blocks', '_0', 'attentions', '_0', 'transformer_blocks', '_0', 'attn1', 'processor', 'to_q_lora', 'up', 'weight']
    for i in range(len(parts)):
        if parts[i].isdigit():
            parts[i] = '_' + parts[i]
        if "to" in parts[i] and "lora" in parts[i]:
            parts[i] = parts[i].replace('_lora', '.lora')
        
    new_parts = []
    for i in range(len(parts)):
        if i == 0:
            new_parts.append(LORA_PREFIX_UNET + '_' + parts[i])
        elif i == len(parts) - 2:
            new_parts.append(parts[i] + '_to_' + parts[i+1])
            new_parts[-1] = new_parts[-1].replace('_to_weight', '')
        elif i == len(parts) - 1:
            new_parts[-1] += '.' + parts[i]
        elif parts[i] != 'processor':
            new_parts.append(parts[i])
    new_name = '_'.join(new_parts)
    new_name = new_name.replace('__', '_')
    new_name = new_name.replace('_to_out.', '_to_out_0.')
    return new_name


def main(bin_path, safetensor_path):
    
    bin_state_dict = torch.load(bin_path)
    safetensors_state_dict = {}
    
    for key_bin in bin_state_dict:
        key_safetensors = convert_name_to_safetensors(key_bin)
        safetensors_state_dict[key_safetensors] = bin_state_dict[key_bin]
    
    save_file(safetensors_state_dict, safetensor_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lora_bin_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--output_path", default=None, type=str, required=True, help="Path to the output model.")

    args = parser.parse_args()

    LORA_PREFIX_UNET = 'lora_unet'

    main(args.lora_bin_path,args.output_path)