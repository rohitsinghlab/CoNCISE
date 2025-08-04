
import os
import sys
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

# for command-line use
import typer
import traceback

MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- main ---
def main(
        protein: str = typer.Option('', help="fasta file with protein sequences"),
        drugs: str = typer.Option('', help="text file with drug smiles"),
        output: str = typer.Option('', help="name of output file"),
        device: str = typer.Option('', help="run predictions on GPU (`cuda`) or CPU (`cpu`)"),
        test: bool = typer.Option(False, help="run test function")
    ):
    if test:
        test_main()
    else:
        # error checking for specified fasta file
        if not (protein and drugs and output):
            if not protein:
                typer.echo("❌ Missing required arguments. Provide the name of the protein fasta file using keyword --protein", err=False)
            if not drugs:
                typer.echo("❌ Missing required arguments. Provide the drugs as line-separated smiles text file using keyword --drugs", err=False)
            if not output:
                typer.echo("❌ Missing required arguments. Provide the name of the output file using keyword --output", err=False)
            raise typer.Exit(code=1)
        try:
            # read in fasta sequences 
            proteins = {r.description: str(r.seq) for r in SeqIO.parse(protein, 'fasta')}

            # read in drugs 
            smiles = set()
            with open(drugs, 'r') as f:
                for line in f:
                    smiles.add(line.strip())
            smiles = list(smiles)

            print('device:', device)
            # set device 
            if not device:
                device = torch.device(DEVICE)

            # run all-vs-all prediction 
            results = {}
            for protein_id, protein in proteins.items():
                result = score(protein, smiles, device)
                if len(result) > 0:
                    results[protein_id] = result

            # merge predictions 
            df_out = pd.concat(results, names=['protein'], ignore_index=False).reset_index(level=1, drop=True)
            df_out.to_csv(output)
        except Exception as e:
            print(traceback.format_exc())
            sys.exit(1)


# --- test ---
def test_main():
    smiles = [
        "CCC\C=C\C=C\C(=O)O[C@H]1\C(C[C@H]2C[C@@H](OC(=O)C[C@H](O)C[C@@H]3C[C@H](OC(C)=O)C(C)(C)[C@](O)(C[C@@H]4C\C(C[C@@H](O4)\C=C\C(C)(C)[C@]1(O)O2)=C\C(=O)OC)O3)[C@@H](C)O)=C\C(=O)OC",
        "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(O)=O)C(O)=O"
    ]

    protein  = "MTMDKSELVQKAKLAEQAERYDDMAAAMKAVTEQGHELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTERNEKKQQMGKEYREKIEAELQDICNDVLELLDKYLIPNATQPESKVFYLKMKGDYFRYLSEVASGDNKQTTVSNSQQAYQEAFEISKKEMQPTHPIRLGLALNFSVFYYEILNSPEKACSLAKTAFDEAIAELDTLNEESYKDSTLIMQLLRDNLTLWTSENQGDEGDAGEGEN"

    import atexit 
    protein_input = 'foo.fasta'
    smiles_input = 'bar.txt'
    atexit.register(lambda: os.remove(protein_input) if os.path.isfile(protein_input) else None)
    atexit.register(lambda: os.remove(smiles_input) if os.path.isfile(smiles_input) else None)
    with open(protein_input, 'w') as f:
        f.write(f'>protein1\n{protein}\n')
        f.write(f'>protein2\n{protein}\n')
    with open(smiles_input, 'w') as f:
        for smile in smiles:
            f.write(f'{smile}\n')

    output = 'test.easy_query2.txt'
    atexit.register(lambda: os.remove(output) if os.path.isfile(output) else None)

    import subprocess
    subprocess.run(['concise', 'easy_query2', '--protein', protein_input, '--drugs', smiles_input, '--output', output])

    with open('test.easy_query2.txt', 'r') as fout:
        print(fout.read())
    

# --- scoring function ---
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

    # output data for leftover ligands 
    bad_ligands = set(ligands) - set(codes2smiles.keys())
    for ligand in bad_ligands:
        out.append({"smiles": ligand, "concise_code": smiles2codes.get(ligand, np.nan), "score": 0.0})

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
