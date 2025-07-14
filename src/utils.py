import torch.nn.functional as F

def pad_to_target(tensor, target_shape=(24, 18)):
    current_height, current_width = tensor.shape
    target_height, target_width = target_shape

    # Berechne das benötigte Padding
    pad_height = target_height - current_height
    pad_width = target_width - current_width

    if pad_height < 0 or pad_width < 0:
        raise ValueError("Target shape must be larger than the current shape.")

    # Padding: (left, right, top, bottom)
    padding = (
        pad_width // 2, pad_width - pad_width // 2,
        pad_height // 2, pad_height - pad_height // 2
    )

    padded_tensor = F.pad(tensor, padding, mode='constant', value=0)
    return padded_tensor