import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .drug_decoder import DrugEncoder
from esm.model.esm2 import TransformerLayer

activation_choices = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "gelu": nn.GELU,
}


class Concise(nn.Module):
    """
    Use this model for reproducibility with the paper results. 
    """

    def __init__(
        self,
        drug_layers,
        ligand_dim=2048,
        residue_dim=1280,
        drug_dim=128,
        proj_dim=256,
        nheads=32,
        activation="tanh",
        cosine_prediction=False,
    ):
        super(Concise, self).__init__()

        self.drug_dim = len(drug_layers)
        self.d_encoder = DrugEncoder(
            drug_layers, ligand_dim, drug_dim, activation=activation_choices[activation]
        )

        self.r_project = nn.Linear(residue_dim, proj_dim)
        self.d_project = nn.Linear(drug_dim, proj_dim)

        self.d_to_r_attention = nn.MultiheadAttention(
            proj_dim, nheads, batch_first=True
        )

        self.d_to_d_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.r_to_r_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.r_to_d_attention = nn.MultiheadAttention(
            proj_dim, nheads, batch_first=True
        )
        self.cosine_prediction = cosine_prediction

        if self.cosine_prediction:
            self.final = CosinePredictor(drug_dim * len(drug_layers), proj_dim)
        else:
            self.final = nn.Sequential(
                nn.Linear((len(drug_layers) + 1) * proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, 1),
                nn.Sigmoid(),
            )

    def emb(self, x):
        d_outs = self.d_encoder(x)
        d_emb = d_outs["emb"]
        d_emb = self.d_project(d_emb)
        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed
        return d_emb

    def prot_emb(self, x):
        assert len(x.shape) == 3, "Input must be of shape [batch, 50, 1280]"
        r_emb = self.r_project(x)
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed
        return r_emb.mean(dim=1)

    def codes(self, x):
        d_outs = self.d_encoder(x)
        d_codes = d_outs["codes"]
        return d_codes

    def score(self, r_emb, b_size=32):
        import tqdm

        device = next(self.parameters()).device
        levels = self.d_encoder.get_levels()  # [6,6]
        levels = torch.stack(levels)

        prod_levels = torch.prod(levels, dim=1)  # 36

        quals_cpl = prod_levels  # [36,36,36]
        all_codes = [torch.arange(n) for n in quals_cpl]

        combinations = torch.cartesian_prod(*all_codes)
        scores = []
        # make r_emb the same size as the batch size
        n_entries = combinations.shape[0]
        with torch.no_grad():
            for chunk in combinations.tensor_split(n_entries // b_size):
                chunk = chunk.to(device)
                shape_0 = chunk.shape[0]
                r_emb_p = r_emb.repeat(shape_0, 1, 1)

                res = self.forward(chunk, r_emb_p, is_morgan_fingerprint=False)
                # offload chunk to cpu
                chunk = chunk.cpu()
                scores.append(res["binding"])
        scores = torch.cat(scores).cpu()

        # zip the scores with the combinations and sort by score in descending order
        scores, idxs = torch.sort(scores, descending=True)
        combinations = combinations[idxs]
        return scores, combinations

    def forward(self, d_emb, r_emb, is_morgan_fingerprint=True):
        # encode drug
        if is_morgan_fingerprint:
            d_outs = self.d_encoder(d_emb)
            d_emb, d_codes = (d_outs["emb"], d_outs["codes"])
        else:
            d_codes = d_emb
            d_emb = self.d_encoder.embed(d_emb)

        d_emb = self.d_project(d_emb)
        r_emb = self.r_project(r_emb)

        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")

        d_r_mixed, _ = self.r_to_d_attention(d_emb, r_emb, r_emb)
        d_emb = d_emb + d_r_mixed

        r_d_mixed, _ = self.d_to_r_attention(r_emb, d_emb, d_emb)
        r_emb = r_emb + r_d_mixed

        # do a softmax reduction of the residues into a single column
        r_emb_wt = F.softmax(10 * r_emb, dim=1)
        r_emb = (r_emb * r_emb_wt).sum(dim=1)

        # stack all drug codes into a single column
        d_emb = rearrange(d_emb, "b n k -> b (n k)")

        if self.cosine_prediction:
            return {
                "binding": self.final(d_emb, r_emb),
                "codes": d_codes,
            }

        return {
            "binding": self.final(torch.cat([d_emb, r_emb], dim=-1)).squeeze(-1),
            "codes": d_codes,
        }


class ConciseSA(nn.Module):
    """
    Same Concise but with self attention added on the fixed-length receptor embeddings
    output from Raygun. Use the other model to reproduce the paper results.
    """
    def __init__(
        self,
        drug_layers,
        ligand_dim=2048,
        residue_dim=1280,
        drug_dim=128,
        proj_dim=256,
        nheads=32,
        activation="tanh",
        cosine_prediction=False,
    ):
        super(ConciseSA, self).__init__()

        self.drug_dim = len(drug_layers)
        self.d_encoder = DrugEncoder(
            drug_layers, ligand_dim, drug_dim, activation=activation_choices[activation]
        )

        self.r_project = nn.Linear(residue_dim, proj_dim)
        self.d_project = nn.Linear(drug_dim, proj_dim)

        self.d_to_d_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.r_to_r_attention = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.d_to_r_attention = nn.MultiheadAttention(
            proj_dim, nheads, batch_first=True
        )

        self.r_to_d_attention = nn.MultiheadAttention(
            proj_dim, nheads, batch_first=True
        )

        self.d_to_d_attention_2 = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )

        self.r_to_r_attention_2 = TransformerLayer(
            embed_dim=proj_dim,
            ffn_embed_dim=proj_dim,
            attention_heads=nheads,
            use_rotary_embeddings=True,
        )


        self.cosine_prediction = cosine_prediction

        if self.cosine_prediction:
            self.final = CosinePredictor(proj_dim * len(drug_layers), proj_dim)
        else:
            self.final = nn.Sequential(
                nn.Linear((len(drug_layers) + 1) * proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, 1),
                nn.Sigmoid(),
            )

    def emb(self, x):
        d_outs = self.d_encoder(x)
        d_emb = d_outs["emb"]
        d_emb = self.d_project(d_emb)
        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed
        return d_emb

    def prot_emb(self, x):
        assert len(x.shape) == 3, "Input must be of shape [batch, 50, 1280]"
        r_emb = self.r_project(x)
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed
        return r_emb.mean(dim=1)

    def codes(self, x):
        d_outs = self.d_encoder(x)
        d_codes = d_outs["codes"]
        return d_codes

    def score(self, r_emb, b_size=32, cutoff=0.5):
        device = next(self.parameters()).device

        # Pre-process receptor embeddings once
        with torch.no_grad():
            r_emb = r_emb.to(device)
            r_emb_proj = self.r_project(r_emb)
            r_r_mixed, _ = self.r_to_r_attention(
                rearrange(r_emb_proj, "b n k -> n b k")
            )
            r_emb_processed = rearrange(r_r_mixed, "n b k -> b n k") + r_emb_proj

        # Setup drug combinations
        prod_levels = torch.tensor(self.d_encoder.get_levels()).to(device)
        all_codes = [torch.arange(int(n), device=device) for n in prod_levels]
        combinations = torch.cartesian_prod(*all_codes)

        scores = []
        with torch.no_grad():
            for chunk in combinations.split(b_size):
                # Expand preprocessed r_emb for batch
                r_emb_proj = r_emb_processed.expand(chunk.size(0), -1, -1)

                # Use cached forward
                res = self.forward_cached(
                    chunk, r_emb_proj, None, is_morgan_fingerprint=False
                )
                scores.append(res["binding"])

        scores = torch.cat(scores)

        # Filter and sort
        mask = scores >= cutoff
        scores = scores[mask]
        combinations = combinations[mask]

        scores, idxs = torch.sort(scores, descending=True)
        return scores.cpu(), combinations[idxs].cpu()

    def forward(self, d_emb, r_emb, is_morgan_fingerprint=True):
        # encode drug
        if is_morgan_fingerprint:
            d_outs = self.d_encoder(d_emb)
            d_emb, d_codes = (d_outs["emb"], d_outs["codes"])
        else:
            d_codes = d_emb
            d_emb = self.d_encoder.embed(d_emb)

        d_emb = self.d_project(d_emb)
        r_emb = self.r_project(r_emb)

        d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed
        r_r_mixed, _ = self.r_to_r_attention(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed

        d_r_mixed, _ = self.r_to_d_attention(d_emb, r_emb, r_emb)
        d_emb = d_emb + d_r_mixed

        r_d_mixed, _ = self.d_to_r_attention(r_emb, d_emb, d_emb)
        r_emb = r_emb + r_d_mixed

        d_d_mixed, _ = self.d_to_d_attention_2(rearrange(d_emb, "b n k -> n b k"))
        d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
        d_emb = d_emb + d_d_mixed

        r_r_mixed, _ = self.r_to_r_attention_2(rearrange(r_emb, "b n k -> n b k"))
        r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
        r_emb = r_emb + r_r_mixed

        # do a softmax reduction of the residues into a single column
        r_emb_wt = F.softmax(10 * r_emb, dim=1)
        r_emb = (r_emb * r_emb_wt).sum(dim=1)

        if self.cosine_prediction:

            # reshape d_emb to be B x (3 * proj_dim)
            d_emb = rearrange(d_emb, "b n k -> b (n k)")

            return {
                "binding": self.final(d_emb, r_emb).squeeze(-1),
                "codes": d_codes,
            }

        return {
            "binding": self.final(torch.cat([d_emb, r_emb], dim=-1)).squeeze(-1),
            "codes": d_codes,
        }

    def forward_cached(self, d_emb, r_emb, r_emb_processed, is_morgan_fingerprint=True):
        """
        Args:
            d_emb: Drug embeddings
            r_emb_processed: Already processed receptor embeddings (after projection and self-attention)
            is_morgan_fingerprint: Whether input is Morgan fingerprint
        """
        with torch.no_grad():
            # encode drug
            if is_morgan_fingerprint:
                d_outs = self.d_encoder(d_emb)
                d_emb, d_codes = (d_outs["emb"], d_outs["codes"])
            else:
                d_codes = d_emb
                d_emb = self.d_encoder.embed(d_emb)

            d_emb = self.d_project(d_emb)

            # Drug self-attention
            d_d_mixed, _ = self.d_to_d_attention(rearrange(d_emb, "b n k -> n b k"))
            d_emb = d_emb + rearrange(d_d_mixed, "n b k -> b n k")

            # Cross attention with pre-processed receptor embeddings
            d_r_mixed, _ = self.r_to_d_attention(d_emb, r_emb, r_emb)
            d_emb = d_emb + d_r_mixed

            r_d_mixed, _ = self.d_to_r_attention(r_emb, d_emb, d_emb)
            r_emb = r_emb + r_d_mixed

            d_d_mixed, _ = self.d_to_d_attention_2(rearrange(d_emb, "b n k -> n b k"))
            d_d_mixed = rearrange(d_d_mixed, "n b k -> b n k")
            d_emb = d_emb + d_d_mixed

            r_r_mixed, _ = self.r_to_r_attention_2(rearrange(r_emb, "b n k -> n b k"))
            r_r_mixed = rearrange(r_r_mixed, "n b k -> b n k")
            r_emb = r_emb + r_r_mixed

            # Softmax reduction
            r_emb_wt = F.softmax(10 * r_emb, dim=1)
            r_emb = (r_emb * r_emb_wt).sum(dim=1)

            # mean over the middle dimension of d_emb
            d_emb = rearrange(d_emb, "b n k -> b (n k)")
            if self.cosine_prediction:
                return {
                    "binding": self.final(d_emb, r_emb),
                    "codes": d_codes,
                }
            return {
                "binding": self.final(torch.cat([d_emb, r_emb], dim=-1)).squeeze(-1),
                "codes": d_codes,
            }


class CosinePredictor(nn.Module):
    def __init__(self, drug_dim, proj_dim):
        super().__init__()
        self.ligand = nn.Sequential(nn.Linear(drug_dim, proj_dim), nn.ReLU())
        self.rec = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU())
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, ligand, rec):
        ligand = self.ligand(ligand)
        rec = self.rec(rec)
        return self.cosine(ligand, rec)


def load_concise(model, path):
    model.load_state_dict(torch.load(path)["model_state_dict"])
    return model
