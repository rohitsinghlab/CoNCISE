import torch
import torch.nn as nn
from einops import rearrange, pack, unpack
from typing import Type, List, Dict, Optional
from torch.nn import Module
from torch import Tensor, int32



def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class ParamfreeFSQ(Module):
    """
    A quantization model for embeddings:

    `levels`: a list of integers. The L=len(`levels`) represents the dimension of the intermediate space used for quantization.
    `dim`: Input dim that is accepted by the forward layer of FSQ
    `num_codebooks`: set to 1 by default; DONOT change for the Concise project)
    `keep_num_codebooks_dim`, `scale`: set to default; DONOT change for the Concise project

    ----------------
    Model parameters
    ----------------
    Has no parameters: All the parameters specified in the original FSQ code is removed

    -------------
    Forward block
    -------------

    Input : tensor[`batch`, `seq_length`, `dim`]
    Output: tensor[`batch`, `seq_length`, `levels`], tensor[`batch`, choice = Quantized(Product(`levels`))]
    """

    def __init__(
        self,
        levels: List[int],
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = len(_levels) * num_codebooks

        has_projections = self.dim != effective_codebook_dim

        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = rearrange(codes, "b n c d -> b n (c d)")

        # reconstitute image or video dimensions

        if is_img_or_video:
            codes = unpack_one(codes, ps, "b * d")
            codes = rearrange(codes, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return codes, indices


class ResidualFSQ(nn.Module):
    """

    Architecture:
        1. Input normalization (LayerNorm)
        2. Projection to quantization space (in_proj)
        3. Parameter-free quantization (FSQ)
        4. Projection back to embedding space (out_proj)
        5. Output normalization and activation
        6. Residual connection

    Args:
        fsq_levels (List[int]): Quantization levels for the FSQ layer
        i (int): Layer index (used for identification)
        dim (int, optional): Dimension of input/output embeddings. Defaults to 2048.
        activation (Type[nn.Module], optional): Activation function class. Defaults to nn.Tanh.

    Attributes:
        fsq (ParamfreeFSQ): The parameter-free quantization module
        activation (nn.Module): Activation function instance
        in_proj (nn.Sequential): Input projection network
        out_proj (nn.Sequential): Output projection network
        ln1 (nn.LayerNorm): Input normalization layer
        ln2 (nn.LayerNorm): Output normalization layer
        scale (int): Scaling factor for hidden dimensions (2 for Tanh, 1 for others)

    Shape:
        - Input: (batch_size, 1, dim)
        - Output: Dict containing:
            - quantized: (batch_size, 1, dim)
            - residual: (batch_size, 1, dim)
            - points: (batch_size, 1, fsq_dim)
            - indices: (batch_size, 1, 1) if return_code=True
    """

    def __init__(
        self,
        fsq_levels: List[int],
        i: int,
        dim: int = 2048,
        activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super(ResidualFSQ, self).__init__()

        # Initialize FSQ layer and activation
        self.fsq = ParamfreeFSQ(levels=fsq_levels)
        self.activation = activation()

        # Get effective dimension after quantization
        self.fsq_dim: int = self.fsq.effective_codebook_dim

        # Set scaling factor based on activation type
        self.scale: int = 2 if isinstance(self.activation, nn.Tanh) else 1

        # Build projection networks
        self.in_proj = self._build_input_projection(dim)
        self.out_proj = self._build_output_projection(dim)

        # Initialize normalization layers
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def _build_input_projection(self, dim: int) -> nn.Sequential:
        """Builds the input projection network."""
        return nn.Sequential(
            nn.Linear(dim, self.scale * dim),
            nn.GELU(),
            nn.Linear(self.scale * dim, self.fsq_dim),
        )

    def _build_output_projection(self, dim: int) -> nn.Sequential:
        """Builds the output projection network."""
        return nn.Sequential(
            nn.Linear(self.fsq_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def set_levels(self, levels: List[int], device: torch.device) -> None:
        """
        Updates the quantization levels of the FSQ layer.

        Args:
            levels (List[int]): New quantization levels to use
            device (torch.device): Device to place the new FSQ layer on
        """
        self.fsq = ParamfreeFSQ(levels=levels).to(device)

    def forward(self, x: Tensor, return_code: bool = False) -> Dict[str, Tensor]:
        """
        Forward pass of the ResidualFSQ module. (Algorithm 1 in Supplementary Information)

        Args:
            x (Tensor): Input tensor of shape (batch_size, 1, dim)
            return_code (bool, optional): Whether to return quantization indices.
                Defaults to False.

        Returns:
            Dict[str, Tensor]: Dictionary containing:
                - quantized: Quantized representation after activation
                - residual: Residual signal (input - quantized)
                - points: Raw quantization points before projection
                - indices: (Optional) Quantization indices if return_code=True
        """
        # Input normalization
        x = self.ln1(x)

        # Project to quantization space and quantize
        projected = self.in_proj(x)
        quantized_points, indices = self.fsq(projected)

        # Project back to embedding space
        output = self.out_proj(quantized_points)
        output = self.ln2(output)

        # Apply activation and compute residual
        activated_output = self.activation(output)
        residual = x - activated_output

        # Prepare return dictionary
        result = {
            "quantized": activated_output,
            "residual": residual,
            "points": quantized_points,
        }

        # Optionally include quantization indices
        if return_code:
            result["indices"] = indices.unsqueeze(1)

        return result
