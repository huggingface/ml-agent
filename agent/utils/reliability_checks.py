"""Reliability checks for job submissions and other operations"""


def check_training_script_save_pattern(script: str) -> str | None:
    """Check if a training script properly saves models."""
    # MLX fine-tuning check (takes priority for mlx-lm scripts)
    has_mlx_train = "mlx_lm" in script
    if has_mlx_train:
        has_mlx_save = "save_adapters" in script or "adapter_path" in script
        if not has_mlx_save:
            return (
                "\n\033[91mWARNING: MLX training detected but no adapter save found. "
                "Add adapter_path= to train() or call save_adapters() — "
                "adapters are lost when the process exits.\033[0m"
            )
        return "\n\033[92mMLX adapters will be saved.\033[0m"

    # HF Transformers / TRL check
    has_from_pretrained = "from_pretrained" in script
    has_push_to_hub = "push_to_hub" in script

    if has_from_pretrained and not has_push_to_hub:
        return "\n\033[91mWARNING: No model save detected in this script. Ensure this is intentional.\033[0m"
    elif has_from_pretrained and has_push_to_hub:
        return "\n\033[92mModel will be pushed to hub after training.\033[0m"

    return None
