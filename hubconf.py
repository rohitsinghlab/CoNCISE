import torch
from concise.model.concise import Concise, ConciseSA


def pretrained_concise_v1(pretrained = True, progress = True):
    url = "https://zenodo.org/records/14613538/files/pretrained.sav?download=1"

    checkpoint = torch.hub.load_state_dict_from_url(url, progress=progress,
                                                       map_location = torch.device("cpu"))
    if pretrained:
        model = Concise(
                    [[32], [32], [32]],
                    ligand_dim=2048,
                    residue_dim=1280,
                    drug_dim=256,
                    proj_dim=256,
                    nheads=64,
                    activation="gelu",
                    cosine_prediction=True,
                )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def pretrained_concise_v2(pretrained = True, progress = True):
    
    url = "https://zenodo.org/records/14890265/files/feb4-retrained.sav?download=1"
    checkpoint = torch.hub.load_state_dict_from_url(url, progress=progress,
                                                       map_location = torch.device("cpu"))

    model = ConciseSA(
        drug_layers=[[32], [32], [32]],
        ligand_dim = 2048,
        residue_dim= 1280,
        drug_dim   = 256,
        proj_dim   = 256,
        nheads     = 64,
        activation = "gelu",
        cosine_prediction=True,
    )    
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
