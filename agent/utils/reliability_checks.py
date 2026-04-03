"""Reliability checks for job submissions and other operations"""


def check_training_script_save_pattern(script: str) -> str | None:
    """Check if a training script properly saves models."""
    has_from_pretrained = "from_pretrained" in script
    has_push_to_hub = "push_to_hub" in script

    if has_from_pretrained and not has_push_to_hub:
        return "\n\033[91mWARNING: No model save detected in this script. Ensure this is intentional.\033[0m"
    elif has_from_pretrained and has_push_to_hub:
        return "\n\033[92mModel will be pushed to hub after training.\033[0m"

    return None
