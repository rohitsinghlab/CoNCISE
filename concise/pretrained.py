import torch

def concise_moodeng(version = "pretrained_concise_v2"):
    return torch.hub.load("rohitsinghlab/CoNCISE", version, pretrained=True)
