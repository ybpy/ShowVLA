import argparse
import os
import os.path as osp
import json
import torch
import sys

from utils import get_config, path_to_llm_name, load_state_dict, load_xvla_modules, replace_model_parameters, remove_trailing_digits
from models.misc import get_text_tokenizer, get_weight_type
from models import Showo2Qwen2_5

from omegaconf import OmegaConf
from transformers import Qwen2MoeConfig
from peft import LoraConfig, get_peft_model

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config = get_config()

    # parser = argparse.ArgumentParser(description="Launch ShowVLA inference FastAPI server")
    # parser.add_argument("--model_path", type=str, required=True,
    #                     help="Path to the pretrained ShowVLA model directory")
    # parser.add_argument('--LoRA_path', type=str, default=None)
    # parser.add_argument("--output_dir", type=str, default="./logs",
    #                     help="Directory to save runtime info (info.json)")
    # parser.add_argument("--device", type=str, default="cuda",
    #                     help="Device to load model on (cuda / cpu / auto)")
    # parser.add_argument("--port", default=8000, type=int,
    #                     help="Port number for FastAPI server")
    # parser.add_argument("--host", default="0.0.0.0", type=str,
    #                     help="Host address for FastAPI server")
    # args = parser.parse_args()

    os.makedirs(config.output_dir, exist_ok=True)

    print("🚀 Starting ShowVLA Inference Server...")
    print(f"🔹 Model Path  : {config.model_path}")
    print(f"🔹 Output Dir  : {config.output_dir}")
    print(f"🔹 Device Arg  : {config.device}")
    print(f"🔹 Port        : {config.port}")

    # --------------------------------------------------------------------------
    # Select device automatically
    # --------------------------------------------------------------------------
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    print(f"🧠 Using device: {device}")

    # --------------------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------------------
    print("\n📦 Loading ShowVLA model from pretrained checkpoint...")
    weight_type = get_weight_type(config)
    try:
        # VQ model for processing image into discrete tokens
        if config.model.vae_model.type == 'wan21':
            from models import WanVAE
            vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
        else:
            raise NotImplementedError

        # Initialize Show-o model
        pred_act = config.model.showo.pred_act if 'pred_act' in config.model.showo else False 
        text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True,
                                                            return_showo_token_ids=True,
                                                            llm_name=path_to_llm_name[config.model.showo.llm_model_path],
                                                            add_return_act_token_ids=pred_act)
        config.model.showo.llm_vocab_size = len(text_tokenizer)

        print(config.model.showo)
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        # Drop-upcycling if needed
        if config.model.showo.drop_upcycling:
            logger.info("Dropping upcycling modules...")
            # Create MoE config from yaml settings
            config.model.showo.moe_config.vocab_size = config.model.showo.llm_vocab_size
            moe_config_dict = OmegaConf.to_container(config.model.showo.moe_config, resolve=True)
            target_config = Qwen2MoeConfig(**moe_config_dict)
            model.showo = replace_model_parameters(
                logger=logger,
                source_model=model.showo,
                target_config=target_config,
                num_experts=config.model.showo.moe_config.num_experts,
                num_layers=config.model.showo.moe_config.num_hidden_layers,
                seed=config.training.seed,
                init_method=config.model.showo.init_method,
                ffn_init_ratio=config.model.showo.ffn_init_ratio,
            ).to(device)
            logger.info("Drop-upcycling completed. Model converted to MoE architecture.")
        
        # Load XVLA action modules
        xvla_checkpoint = config.model.showo.get('xvla_ckpt_path', None)
        if xvla_checkpoint is not None and config.model.showo.xvla_hidden_size is not None:
            logger.info("Loading XVLA action modules...")
            success = load_xvla_modules(
                logger,
                model, 
                xvla_checkpoint,
                module_names=config.model.showo.get('xvla_modules_to_load', 
                    ['action_encoder', 'action_decoder', 'norm', 'pos_emb', 'soft_prompt_hub']),
                source_prefix=config.model.showo.get('source_prefix', 'transformer'),
                target_prefix=config.model.showo.get('target_prefix', None),
            )
            if not success:
                logger.error("Failed to load XVLA modules! Please check:")
            else:
                logger.info("XVLA action modules loaded successfully!")

        use_lora = config.training.get('use_lora', False)
        lr_multipler = config.training.get('lr_multipler', 1.0)
        if use_lora:
            exclude_modules = ["time_embed"]
            suffix_of_modules_to_save = [
                "mlp.gate",
                # "mlp.experts",
                "lm_head",
                "image_embedder_und",
                "image_embedder_gen",
                "position_embedding",
                # "fusion_proj",
                # "time_embed",
                "diff_proj",
                "time_embed_proj",
                "diffusion_head_b",
            ]
            modules_to_save = ["norm"]
            if config.model.showo.xvla_hidden_size is not None:
                modules_to_save = [
                    "project_xvla_encode",
                    "project_xvla_decode",
                    "pos_emb",
                    "norm",
                    "action_encoder",
                    "action_decoder",
                    "soft_prompt_hub",
                ]
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.ModuleList) or isinstance(module, torch.nn.Sequential):
                    continue
                if any((name.endswith(x) or remove_trailing_digits(name).endswith(x)) for x in suffix_of_modules_to_save): 
                    modules_to_save.append(name)
            for name in modules_to_save:
                logger.info(f"[modules_to_save] {name}")
            
            lora_config = LoraConfig(
                lora_alpha=48,
                r=24,
                bias="none",
                target_modules="all-linear",
                exclude_modules=exclude_modules,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, lora_config)

        state_dict = load_state_dict(config.model_path)
        # Unwrap model manually to match the state_dict structure
        unwrapped_model = model
        while hasattr(unwrapped_model, "_orig_mod"):
            unwrapped_model = unwrapped_model._orig_mod
        if hasattr(unwrapped_model, "base_model"):
            unwrapped_model = unwrapped_model.base_model.model
        unwrapped_model.load_state_dict(state_dict, strict=False if config.model.showo.params_not_load is not None else True)
        del state_dict
        if use_lora:
            model = model.merge_and_unload()
        model.to(weight_type)
        model.eval()

        print("✅ Model successfully loaded and moved to device.")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # --------------------------------------------------------------------------
    # SLURM environment detection
    # --------------------------------------------------------------------------
    node_list = os.environ.get("SLURM_NODELIST")
    job_id = os.environ.get("SLURM_JOB_ID", "none")

    if node_list:
        print("\n🖥️  SLURM Environment Detected:")
        print(f"   Node list : {node_list}")
        print(f"   Job ID    : {job_id}")

        # Extract host
        try:
            host = ".".join(node_list.split("-")[1:]) if "-" in node_list else node_list
        except Exception:
            host = config.host
    else:
        print("\n⚠️  No SLURM environment detected, defaulting to 0.0.0.0")
        host = config.host

    # --------------------------------------------------------------------------
    # Write info.json for bookkeeping (safe version)
    # --------------------------------------------------------------------------
    info_path = osp.join(config.output_dir, "info.json")
    infos = {
        "host": host,
        "port": config.port,
        "job_id": job_id,
        "node_list": node_list or "none",
    }

    # --- Check existence before writing ---
    if osp.exists(info_path):
        print(f"❌ Error: {info_path} already exists. "
            f"This usually means another server is still running or the previous job did not clean up properly.")
        print("👉 Please remove it manually or use a different --output_dir.")
        sys.exit(1)

    # --- Write safely ---
    try:
        with open(info_path, "w") as f:
            json.dump(infos, f, indent=4)
        print(f"📝 Server info written to {info_path}")
    except Exception as e:
        print(f"⚠️ Failed to write {info_path}: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Launch FastAPI server
    # --------------------------------------------------------------------------
    print(f"\n🌐 Launching FastAPI service at http://{host}:{config.port} ...")
    try:
        if hasattr(model, "run"):
            model.run(config=config, vae_model=vae_model, text_tokenizer=text_tokenizer, showo_token_ids=showo_token_ids, 
                      host=host, port=config.port)
        else:
            print("❌ The loaded model does not implement `.run()` (FastAPI entrypoint).")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped manually.")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")


if __name__ == "__main__":
    main()