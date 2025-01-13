import pandas as pd
import h5py as h5
from molfeat.trans.fp import FPVecTransformer
from tqdm import tqdm
from torch.utils.data import Dataset
from einops import rearrange


MIN_SEQ_LEN = 50
MAX_SEQ_LEN = 2000


def get_fingerprints(ligand_list):
    MORGAN_FINGERPRINT_DIMS = 2048
    transformer = FPVecTransformer(
        kind="ecfp:4", length=MORGAN_FINGERPRINT_DIMS, verbose=True
    )

    valid_features, valid_ids = transformer(ligand_list, ignore_errors=True)
    return valid_features, valid_ids


def sanitize_string(s):
    return s.replace("/", "|")


def construct_ligand_h5(ligand_list, output_h5_file, n_jobs=16):
    """
    Takes in the list of SMILES strings
    Saves the Morgan fingerprints produced in the `output_h5_file`
    """
    # Remove duplicates and sort to maintain consistent order
    ligand_list = sorted(set(ligand_list))
    valid_features, valid_ids = get_fingerprints(ligand_list)

    with h5.File(output_h5_file, "w") as hdf_file:
        for feat, idx in tqdm(zip(valid_features, valid_ids), desc="Saving h5"):
            dataset_name = ligand_list[idx]
            dataset_name = sanitize_string(dataset_name)
            if dataset_name in hdf_file:
                print(f"Dataset {dataset_name} already exists. Skipping.")
                continue
            try:
                hdf_file.create_dataset(dataset_name, data=feat)
            except Exception as e:
                print(f"Error saving {dataset_name}: {e}")

    return [ligand_list[idx] for idx in valid_ids]


def construct_receptor_h5(receptor_seqs, output_h5_file, device=0):
    """
    Takes in a list of sequences and returns the Raygun embeddings
    Saves the data to the `output_h5_file`
    """
    import esm
    import torch
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    receptor_seqs = [
        seq
        for seq in receptor_seqs
        if len(seq) >= MIN_SEQ_LEN and len(seq) <= MAX_SEQ_LEN
    ]

    esmembeddings = []
    with torch.no_grad():
        for seq in tqdm(receptor_seqs, desc="Generating ESM representations"):
            _, _, tokens = bc([("seq", seq.upper())])
            embedding = esm_model(
                tokens.to(device), repr_layers=[33], return_contacts=False
            )["representations"][33]
            esmembeddings.append(embedding[:, 1:-1, :].cpu())

    del esm_model

    raymodel, esmtotokdecoder, hypparams = torch.hub.load(
        "rohitsinghlab/raygun", "pretrained_uniref50_95000_750M"
    )
    raymodel = raymodel.to(device)

    with h5.File(output_h5_file, "w") as hdf_file:
        with torch.no_grad():
            for emb, seq in tqdm(
                zip(esmembeddings, receptor_seqs),
                desc="Constructing Raygun fixed-length encodings.",
            ):
                rayencode = raymodel.encoder(emb.to(device)).squeeze().cpu().numpy()
                hdf_file.create_dataset(seq, data=rayencode)
    del raymodel
    return


def get_concise_representations(concisemodel, ligand_list):
    import torch
    valid_features, valid_ids = get_fingerprints(ligand_list)
    representations = []
    with torch.no_grad():
        concisemodel.eval()
        for feature in valid_features:
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            rep = concisemodel.emb(feature_tensor).squeeze(0)
            rep = rearrange(rep, "n k -> (n k)")
            # rep = feature_tensor.squeeze(0)
            representations.append(rep)
    # Stack representations into a single tensor
    representations = torch.stack(representations).numpy()
    print(representations.shape)
    return representations


class DTIDataset(Dataset):
    """
    Takes in a TSV file with entries: `seq`, `lig`, (`0/1` if train else Nothing)

    for `seq` you input a h5 file that contains the raygun repr of dim: 1, 50, 1280
    for `lig` you have a h5 file that contains the morgan fingerprint of dim: 1, 2048

    So, three inputs in total
    """

    def __init__(self, dti_file, rec_embed_file, lig_embed_file, mode="train"):
        self.mode = mode
        self.dtidata = pd.read_csv(dti_file)
        entrycolumns = ["Target Sequence", "SMILES"]
        if mode == "train":
            entrycolumns += ["Label"]

        self.dtidata = self.dtidata[entrycolumns]

        self.ligand_embeddings = h5.File(lig_embed_file, "r")
        available_ligands = set(self.ligand_embeddings.keys())

        self.receptor_embeddings = h5.File(rec_embed_file, "r")
        available_receptors = set(self.receptor_embeddings.keys())

        self.dtidata["SMILES"] = self.dtidata["SMILES"].apply(sanitize_string)

        # remove the entries with no embeddings from dtidata
        self.dtidata = self.dtidata.loc[
            self.dtidata["SMILES"].apply(lambda x: x in available_ligands), :
        ]
        self.dtidata = self.dtidata.loc[
            self.dtidata["Target Sequence"].apply(lambda x: x in available_receptors), :
        ]
        self.dtidata = self.dtidata.reset_index(drop=True)
        return

    def __len__(self):
        return len(self.dtidata)

    def __getitem__(self, idx):
        import torch
        package = self.dtidata.loc[idx, :].values
        if self.mode == "train":
            receptor, ligand, label = package
        else:
            receptor, ligand = package

        rec_emb = torch.tensor(self.receptor_embeddings[receptor][:])
        lig_emb = torch.tensor(self.ligand_embeddings[ligand][:])

        if self.mode == "train":
            return rec_emb, lig_emb, torch.tensor(label, dtype=torch.long), idx
        else:
            return rec_emb, lig_emb


class LigandDataset(Dataset):
    def __init__(self, lig_file, lig_embed_file):
        self.ligand_embeddings = h5.File(lig_embed_file, "r")
        self.ligands = pd.read_csv(lig_file)["SMILES"].apply(sanitize_string)
        available_ligands = set(self.ligand_embeddings.keys())
        self.ligands = self.ligands.loc[
            self.ligands.apply(lambda x: x in available_ligands)
        ]
        self.ligands = self.ligands.drop_duplicates()
        self.ligands = self.ligands.reset_index(drop=True)
        return

    def __len__(self):
        return len(self.ligands)

    def __getitem__(self, idx):
        import torch
        ligand = self.ligands[idx]
        lig_emb = torch.tensor(self.ligand_embeddings[ligand][:])
        return lig_emb


class ScoreDataset(Dataset):
    """
    Takes in a TSV file with entries: `seq`, `lig`, (`0/1` if train else Nothing)

    for `seq` you input a h5 file that contains the raygun repr of dim: 1, 50, 1280
    for `lig` you have a h5 file that contains the morgan fingerprint of dim: 1, 2048

    So, three inputs in total
    """

    def __init__(self, recep, rec_embed_file):
        self.dtidata = pd.read_csv(recep)
        entrycolumns = ["Target Sequence"]

        self.dtidata = self.dtidata[entrycolumns]

        self.receptor_embeddings = h5.File(rec_embed_file, "r")
        available_receptors = set(self.receptor_embeddings.keys())

        self.dtidata = self.dtidata.loc[
            self.dtidata["Target Sequence"].apply(lambda x: x in available_receptors), :
        ]
        self.dtidata = self.dtidata.reset_index(drop=True)
        return

    def __len__(self):
        return len(self.dtidata)

    def __getitem__(self, idx):
        import torch
        receptor = self.dtidata["Target Sequence"][idx]

        rec_emb = torch.tensor(self.receptor_embeddings[receptor][:])

        return rec_emb


class ScoreFastaDataset(Dataset):
    """
    Takes in a TSV file with entries: `seq`, `lig`, (`0/1` if train else Nothing)

    for `seq` you input a h5 file that contains the raygun repr of dim: 1, 50, 1280
    for `lig` you have a h5 file that contains the morgan fingerprint of dim: 1, 2048

    So, three inputs in total
    """

    def __init__(self, fasta_file, rec_embed_file, max_entries=-1):
        from Bio import SeqIO

        self.seqdata = list(SeqIO.parse(fasta_file, "fasta"))
        if max_entries > 0:
            self.seqdata = self.seqdata[:max_entries]

        self.receptor_embeddings = h5.File(rec_embed_file, "r")
        available_receptors = set(self.receptor_embeddings.keys())
        self.seqdata = [x for x in self.seqdata if str(x.seq) in available_receptors]

        return

    def __len__(self):
        return len(self.seqdata)

    def __getitem__(self, idx):
        import torch
        receptor = str(self.seqdata[idx].seq)

        rec_emb = torch.tensor(self.receptor_embeddings[receptor][:])

        return rec_emb
