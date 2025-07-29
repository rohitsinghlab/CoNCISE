
import typer
import hydra
import os
import logging
import pandas as pd
import sqlite3
from Bio import SeqIO
from Bio import Seq
from Bio import SeqRecord
from pathlib import Path
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import tempfile 
import shutil 
import atexit

# --- temporary directory handling ---
from ..configs import get_config_path
DEFAULT_CONFIG_PATH = get_config_path("easy_query2.yaml")
TEMPDIR_NAME = tempfile.gettempdir()
print(f"Using the following temporary directory for easy_query: {TEMPDIR_NAME}")


def main():
    url = f'https://rest.uniprot.org/uniprotkb/P31946.fasta'
    import requests 
    from io import StringIO
    handle = requests.get(url)
    protein = str(SeqIO.read(StringIO(handle.text), 'fasta').seq)
    ligands = ["CCC\C=C\C=C\C(=O)O[C@H]1\C(C[C@H]2C[C@@H](OC(=O)C[C@H](O)C[C@@H]3C[C@H](OC(C)=O)C(C)(C)[C@](O)(C[C@@H]4C\C(C[C@@H](O4)\C=C\C(C)(C)[C@]1(O)O2)=C\C(=O)OC)O3)[C@@H](C)O)=C\C(=O)OC",  
            "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(O)=O)C(O)=O"]
    score(protein, ligands)


def score(protein, ligands, config_path=DEFAULT_CONFIG_PATH):
    from yaml import safe_load, safe_dump

    print("FLAG 67: changed code", __file__)
    @atexit.register
    def cleanup_tempdir():
        tempdir = Path(TEMPDIR_NAME) / "easy_query_temp"
        if tempdir.exists():
            print(f"Cleaning up temporary directory: {tempdir}")
            shutil.rmtree(tempdir)
        else:
            print(f"No temporary directory found at: {tempdir}")
    
    
    def create_tempdir():
        tempdir = Path(TEMPDIR_NAME) / "easy_query_temp"
        if not tempdir.exists():
            tempdir.mkdir(parents=True, exist_ok=True)
        return tempdir

    # change default config to use a temporary file 
    with open(config_path, 'r') as f:
        cfg = safe_load(f)
    for key, value in cfg['paths'].items():
        cfg[key] = str(create_tempdir() / value)
    config_path = create_tempdir() / "easy_query_temp.yaml"
    with open(config_path, 'w') as f:
        safe_dump(cfg, f)
    print("config_path", config_path)
    print("cfg", cfg)


    # --- create the necessary temporary files  ---
    # protein
    seq = SeqRecord.SeqRecord(
        Seq.Seq(protein),
        id="user_protein",
        description="Temporary protein sequence for scoring"
    )
    temp_protein_file = create_tempdir() / cfg["fasta_file"]
    temp_protein_file.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(seq, temp_protein_file, "fasta")

    # ligands
    ligands_df = pd.DataFrame({"SMILES": ligands})
    temp_ligand_file = create_tempdir() / cfg["ligand_file"]
    temp_ligand_file.parent.mkdir(parents=True, exist_ok=True)
    ligands_df.to_csv(temp_ligand_file, index=False)

    
    main2(config_path)


def main2(config_path: Path = typer.Option(..., help="Path to config file")):
    from concise.dataset import construct_receptor_h5, construct_ligand_h5
    from concise.pretrained import concise_moodeng
    from .protein_to_codes import score_proteins
    from .smiles_to_codes import assign_codes
    from .query import get_top_codes_and_smiles, process_results

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    logger.info("Loading the model")
    model = concise_moodeng()

    # if the h5 files are not present, create it
    create_receptor_emb = not os.path.exists(cfg["protein_dataset"]["rec_embed_file"])
    create_ligand_emb = not os.path.exists(cfg["ligand_dataset"]["lig_embed_file"])

    if create_receptor_emb:
        seqs = list(SeqIO.parse(cfg["fasta_file"], "fasta"))
        all_seqs = list(set([str(x.seq) for x in seqs]))
        receptor_path = cfg["rec_embed_file"]
        construct_receptor_h5(all_seqs, receptor_path)
    if create_ligand_emb:
        smiles = pd.read_csv(cfg["ligand_file"])["SMILES"]
        ligand_path = cfg["lig_embed_file"]
        construct_ligand_h5(smiles, ligand_path)

    logger.info("Loading the fasta dataloader.")

    protein_dataloader = hydra.utils.instantiate(cfg["protein_dataset"]["protein_dataloader"])
    ligand_dataloader = hydra.utils.instantiate(cfg["ligand_dataset"]["ligand_dataloader"])
    device = hydra.utils.instantiate(cfg["device"])

    model.to(device)
    protein_scores = score_proteins(model, protein_dataloader, device)
    ligand_scores = assign_codes(model, ligand_dataloader, device)
    dataset = hydra.utils.instantiate(cfg["protein_dataset"]["protein_dataset"]).seqdata

    score_data = []
    for idx, (score, combination) in enumerate(protein_scores):
        for i in range(len(score)):
            binding_score = score[i].item()
            # if binding_score < 0.5:
                # continue
            score_data.append(
                {
                    "ID": str(dataset[idx].id),
                    "Code": "-".join([str(x) for x in combination[i].tolist()]),
                    "Score": binding_score,
                }
            )

    dataset = hydra.utils.instantiate(cfg["ligand_dataset"]["ligand_dataset"])
    ligand_scores = ligand_scores.cpu().detach().numpy().tolist()
    ligand_scores = ["-".join([str(x) for x in code]) for code in ligand_scores]

    # match smiles to codes
    smiles = dataset.ligands
    smiles_codes = pd.DataFrame(
        {
            "SMILES": smiles,
            "Code": ligand_scores,
        }
    )
    score_df = pd.DataFrame(score_data)

    # conn = sqlite3.connect(":memory:")
    # smiles_codes.to_sql("codes", conn, index=False)
    # score_df.to_sql("protein_scores", conn, index=False)

    score_df['Code'] = score_df['Code'].astype(str)
    smiles_codes['Code'] = smiles_codes['Code'].astype(str)
    merge_df = score_df.merge(smiles_codes, on='Code', how='inner')
    return merge_df

    # try:
        # results_df = get_top_codes_and_smiles(conn, cfg["num_codes_per_protein"], cfg["num_smiles_per_code"])
        # save_path = Path(cfg["save_path"])
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # results_df.to_csv(save_path, index=False)
        
    # finally:
        # conn.close()


if __name__ == "__main__":
    main()
