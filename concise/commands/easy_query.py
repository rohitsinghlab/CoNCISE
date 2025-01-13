import typer
import hydra
import os
import logging
import pandas as pd
import sqlite3
from Bio import SeqIO
from pathlib import Path
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(config_path: Path = typer.Option(..., help="Path to config file")):
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
            if binding_score < 0.5:
                continue
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

    conn = sqlite3.connect(":memory:")
    smiles_codes.to_sql("codes", conn, index=False)
    score_df.to_sql("protein_scores", conn, index=False)

    try:
        results_df = get_top_codes_and_smiles(conn, cfg["num_codes_per_protein"], cfg["num_smiles_per_code"])
        save_path = Path(cfg["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        
    finally:
        conn.close()



if __name__ == "__main__":
    main()
