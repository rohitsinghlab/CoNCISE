
import os
import sqlite3
import subprocess
import tempfile
import pandas as pd
import concise.pretrained
import numpy as np
import json
import einops 
import torch
from torch.utils.data import DataLoader

from typing import List, Tuple, Dict, Any
from Bio import SeqIO

# concise-specific imports
from ..pretrained import concise_moodeng
from ..dataset import construct_receptor_h5
from ..dataset import LigandDataset, ScoreDataset
from ..dataset import get_fingerprints

from .protein_to_codes import score_proteins

MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- test ---
def main():
    smiles = [
        "CCC\C=C\C=C\C(=O)O[C@H]1\C(C[C@H]2C[C@@H](OC(=O)C[C@H](O)C[C@@H]3C[C@H](OC(C)=O)C(C)(C)[C@](O)(C[C@@H]4C\C(C[C@@H](O4)\C=C\C(C)(C)[C@]1(O)O2)=C\C(=O)OC)O3)[C@@H](C)O)=C\C(=O)OC",
        "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(O)=O)C(O)=O"
    ]

    protein  = "MTMDKSELVQKAKLAEQAERYDDMAAAMKAVTEQGHELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTERNEKKQQMGKEYREKIEAELQDICNDVLELLDKYLIPNATQPESKVFYLKMKGDYFRYLSEVASGDNKQTTVSNSQQAYQEAFEISKKEMQPTHPIRLGLALNFSVFYYEILNSPEKACSLAKTAFDEAIAELDTLNEESYKDSTLIMQLLRDNLTLWTSENQGDEGDAGEGEN"

    print(score(protein, smiles))
    

# --- main scoring function ---
def score(protein, ligands, device=None, model=None):
    if device is None:
        device = torch.device(DEVICE)
    if model is None:
        model = load_model(device)

    smiles2codes = smiles_to_codes(ligands, model, device)
    codes = grab_codes(protein, device=device, threshold=0.0)

    codes2smiles = {}
    for smiles, code in smiles2codes.items():
        if code in codes2smiles:
            codes2smiles[code].append(smiles)
        else:
            codes2smiles[code] = [smiles]

    # output dataset 
    out = []
    for code, score in codes:
        if code in codes2smiles:
            for smiles in codes2smiles[code]:
                out.append({"smiles": smiles, "concise_code": code, "score": score})
    
    if len(out) == 0:
        return pd.DataFrame(columns=["smiles", "concise_code", "score"])

    out = pd.DataFrame(out)
    out = out.sort_values(by="score", ascending=False).reset_index(drop=True)

    return out


# --- function to map smiles to codes ---
def smiles_to_codes(smiles: List[str], model: Any, device: str) -> Dict[str, str]:
    """
    Given a list of SMILES strings, returns a dictionary mapping each SMILES to its corresponding code.
    """
    if device is None:
        device = torch.device(DEVICE)
    valid_features, valid_ids = get_fingerprints(smiles)
    codes = model.codes(torch.tensor(valid_features).to(device))
    smiles2codes = {}
    for i in range(len(valid_ids)):
        drug = smiles[valid_ids[i]]
        code = "-".join([str(x) for x in codes[i].cpu().numpy().tolist()])
        smiles2codes[drug] = code
    return smiles2codes


# --- helper functions ---
def pack3_32(tok1, tok2, tok3):
    alltoks = [tok1, tok2, tok3]
    assert max(alltoks) < 32 and min(alltoks) >= 0
    return np.uint16(tok1 << 10 | tok2 << 5 | tok3)


def grab_codes(protein: str, device: str, threshold: float = 0.0) -> List[Tuple[str, float]]:
    """
    given a protein will return a list of codes that binds to that protein.
    """
    model = concise_moodeng().eval().to(device)
    # Create temporary files for processing
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False
    ) as tf1, tempfile.NamedTemporaryFile(mode="w", delete=False) as tf2:

        # Process the ligand
        construct_receptor_h5([protein], tf1.name)
        pd.DataFrame({"Target Sequence": [protein]}).to_csv(tf2.name)
        dataset = ScoreDataset(tf2.name, tf1.name)
        dl = DataLoader(dataset, batch_size=1)

        print(concise.pretrained.__file__)
        scores = score_proteins(model, dl, device)

        if len(scores) == 0:
            return list()

        scores = [
            ("-".join(map(str, code.tolist())), score.item())
            for (score, code) in zip(*scores[0])
        ]

        scores = [(code, score) for code, score in scores if score > threshold]

    return scores


def load_model(device: str = None):
    if device is None:
        device = torch.device(DEVICE)
    global MODEL 
    if MODEL is None:
        MODEL = concise_moodeng().eval().to(device)
    return MODEL


if __name__ == "__main__":  
    main()
