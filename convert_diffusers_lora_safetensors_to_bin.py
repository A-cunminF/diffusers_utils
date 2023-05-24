from diffusers import DiffusionPipeline
import torch
import argparse

def main(repo_id,lora_path,output_path):
    generator = DiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    generator.unet.load_attn_procs(lora_path)
    generator.unet.save_attn_procs(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model_path", default=None, type=str, required=True, help="Path to the sd model when you train.")
    parser.add_argument("--lora_safetensors_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--output_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args.sd_model_path,args.lora_safetensors_path,args.output_path)