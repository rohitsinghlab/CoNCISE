import typer
import hydra
import os
import logging
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from pathlib import Path
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def score_proteins(model, dataloader, device) -> list:
    """

    model.score takes a single raygun embedding of size 50x1280
        and returns scores for every possible code combination in sorted order
        for a model that quantizes ligands into 32x32x32 codes
        size of the score vector is 32^3 = 32768
    """
    import torch
    model.eval()
    all_scores = []
    with torch.no_grad():
        for idx, rec in tqdm(enumerate(dataloader), total=len(dataloader)):
            scores, codes = model.score(rec.to(device), b_size=32**2)
            all_scores.append((scores, codes))
    return all_scores


def main(config_path: Path = typer.Option(..., help="Path to config file")):

    from concise.dataset import construct_receptor_h5
    from concise.pretrained import concise_moodeng
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    logger.info("Loading the model")
    model = concise_moodeng()

    # if the h5 files are not present, create it
    create_receptor_emb = not os.path.exists(cfg["dataset"]["rec_embed_file"])

    if create_receptor_emb:
        seqs = list(SeqIO.parse(cfg["dataset"]["fasta_file"], "fasta"))
        all_seqs = list(set([str(x.seq) for x in seqs]))
        receptor_path = cfg["dataset"]["rec_embed_file"]
        construct_receptor_h5(all_seqs, receptor_path)

    logger.info("Loading the fasta dataloader.")

    dataloader = hydra.utils.instantiate(cfg["dataset"]["protein_dataloader"])

    device = hydra.utils.instantiate(cfg["device"])

    model.to(device)
    protein_scores = score_proteins(model, dataloader, device)

    save_path = Path(cfg["save_path"])
    save_prefix = save_path.parent
    os.makedirs(save_prefix, exist_ok=True)

    dataset = hydra.utils.instantiate(cfg["dataset"]["protein_dataset"]).seqdata

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

    score_df = pd.DataFrame(score_data)

    if save_path.suffix == ".csv.bz2":
        score_df.to_csv(save_path, index=False, compression="bz2")
    else:
        score_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
