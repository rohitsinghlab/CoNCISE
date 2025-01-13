import torch



def concise_moodeng():
    return torch.hub.load("rohitsinghlab/CoNCISE", "pretrained_concise_v1", pretrained=True)
