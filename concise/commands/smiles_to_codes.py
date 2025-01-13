import typer
import hydra
import os
from omegaconf import OmegaConf
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def assign_codes(model, dataloader, device) -> list:
    """

    model.score takes a single raygun embedding of size 50x1280
        and returns scores for every possible code combination in sorted order
        for a model that quantizes ligands into 32x32x32 codes
        size of the score vector is 32^3 = 32768
    """
    import torch
    model.eval()
    all_codes = []
    with torch.no_grad():
        for idx, rec in tqdm(enumerate(dataloader), total=len(dataloader)):
            codes = model.codes(rec.to(device))
            all_codes.append(codes.cpu().detach())
    all_codes = torch.cat(all_codes, dim=0)
    return all_codes


def main(config_path: Path = typer.Option(..., help="Path to config file")):
    from concise.dataset import construct_ligand_h5
    from concise.pretrained import concise_moodeng
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    logger.info("Loading the model")
    model = concise_moodeng()

    # if the h5 files are not present, create it
    create_ligand_emb = not os.path.exists(cfg["dataset"]["lig_embed_file"])

    if create_ligand_emb:
        smiles = pd.read_csv(cfg["ligand_file"])["SMILES"].unique().tolist()
        lig_emb_path = cfg["dataset"]["lig_embed_file"]
        construct_ligand_h5(smiles, lig_emb_path)

    logger.info("Loading the fasta dataloader.")

    dataloader = hydra.utils.instantiate(cfg["dataset"]["ligand_dataloader"])

    device = hydra.utils.instantiate(cfg["device"])

    model.to(device)
    ligand_codes = assign_codes(model, dataloader, device)


    dataset = hydra.utils.instantiate(cfg["dataset"]["ligand_dataset"])
    ligand_codes = ligand_codes.cpu().detach().numpy().tolist()
    ligand_codes = ["-".join([str(x) for x in code]) for code in ligand_codes]

    # match smiles to codes
    smiles = dataset.ligands
    smiles_codes = pd.DataFrame(
        {
            "SMILES": smiles,
            "Code": ligand_codes,
        }
    )

    save_path = Path(cfg["save_path"])
    smiles_codes.to_csv(save_path, index=False)

    save_as_sqlite = cfg["save_as_sqlite"]
    if save_as_sqlite:
        import sqlite3
        save_path = save_path.with_suffix(".sqlite")
        conn = sqlite3.connect(save_path)
        smiles_codes.to_sql("codes", conn, index=False)



if __name__ == "__main__":
    main()
