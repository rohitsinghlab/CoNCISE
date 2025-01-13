import torch
from einops.layers.torch import Rearrange
import torch.nn as nn
from typing import List, Dict, Type
from .fsq import ResidualFSQ


class DrugEncoder(torch.nn.Module):
    """

    Architecture:
        1. Pre-transformation: Input -> Linear -> Activation -> Linear -> Activation
        2. Series of ResidualFSQ layers that each:
           - Quantize the input
           - Compute residual
           - Pass residual to next layer

    Args:
        layers (List[List[int]]): Configuration for FSQ layers. Each inner list specifies the
            quantization levels for a single ResidualFSQ layer.
        dim (int, optional): Input dimension of the drug representation. Defaults to 2048.
        latent_dim (int, optional): Dimension of the latent space after pre-transformation.
            Defaults to 128.
        activation (Type[torch.nn.Module], optional): Activation function to use throughout
            the network. Defaults to nn.Tanh.

    Attributes:
        residualfsqs (torch.nn.ModuleList): List of ResidualFSQ layers.
        pre_transform (torch.nn.Sequential): Pre-transformation network that processes the
            input before quantization.

    Shape:
        - Input: (batch_size, dim)
        - Output: Dict containing:
            - codes: (batch_size, num_layers)
            - emb: (batch_size, num_layers, latent_dim)
            - points: (batch_size, num_layers, latent_dim)
    """

    residualfsqs: torch.nn.ModuleList
    pre_transform: torch.nn.Sequential

    def __init__(
        self,
        layers: List[List[int]],
        dim: int = 2048,
        latent_dim: int = 128,
        activation: Type[torch.nn.Module] = nn.Tanh,
    ) -> None:
        super(DrugEncoder, self).__init__()

        # Initialize pre-transformation network
        self.pre_transform = self._build_pre_transform(dim, latent_dim, activation)

        # Initialize ResidualFSQ layers
        self.residualfsqs = self._build_residual_fsq_layers(
            layers, latent_dim, activation
        )

    def _build_pre_transform(
        self, dim: int, latent_dim: int, activation: Type[torch.nn.Module]
    ) -> torch.nn.Sequential:
        """Builds the pre-transformation network."""
        return torch.nn.Sequential(
            Rearrange("b d -> b 1 d"),  # Add sequence dimension
            torch.nn.Linear(dim, dim // 2),
            activation(),
            torch.nn.Linear(dim // 2, latent_dim),
            activation(),
        )

    def _build_residual_fsq_layers(
        self,
        layers: List[List[int]],
        latent_dim: int,
        activation: Type[torch.nn.Module],
    ) -> torch.nn.ModuleList:
        """Builds the sequence of ResidualFSQ layers."""
        return torch.nn.ModuleList(
            [
                ResidualFSQ(
                    layer_config,
                    idx,
                    dim=latent_dim,
                    activation=activation,
                )
                for idx, layer_config in enumerate(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the DrugEncoder. (Algorithm 2 in Supplementary Information)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - codes: Quantization indices for each layer (batch_size, num_layers)
                - emb: Quantized embeddings (batch_size, num_layers, latent_dim)
                - points: Quantization points (batch_size, num_layers, latent_dim)
        """
        # Pre-transform input
        x = self.pre_transform(x)

        # Process first layer separately to initialize lists
        res = self.residualfsqs[0](x, return_code=True)

        # Initialize collection lists
        codebookids: List[torch.Tensor] = [res["indices"]]
        quantizeds: List[torch.Tensor] = [res["quantized"]]
        points: List[torch.Tensor] = [res["points"]]
        residual: torch.Tensor = res["residual"]

        # Process remaining layers
        for layer in self.residualfsqs[1:]:
            res = layer(residual, return_code=True)
            codebookids.append(res["indices"])
            quantizeds.append(res["quantized"])
            points.append(res["points"])
            residual = res["residual"]

        return {
            "codes": torch.cat(codebookids, dim=1).squeeze(-1),
            "emb": torch.cat(quantizeds, dim=1),
            "points": torch.cat(points, dim=1),
        }

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert quantization codes back to embeddings.

        This method reconstructs embeddings from the quantization codes produced by the forward pass.
        It's useful for tasks like generation or reconstruction where you start with codes
        rather than raw inputs.

        Args:
            codes (torch.Tensor): Tensor of quantization indices of shape (batch_size, num_layers)

        Returns:
            torch.Tensor: Reconstructed embeddings of shape (batch_size, num_layers, latent_dim)
        """
        # Split codes into individual codebook IDs along dim=1
        cids: List[torch.Tensor] = codes.split(1, dim=1)
        embeddings: List[torch.Tensor] = []

        # Process each layer's codes
        for rfsq, cid in zip(self.residualfsqs, cids):
            # Convert indices to embeddings
            emb = rfsq.fsq.indices_to_codes(cid.squeeze(1)).unsqueeze(
                1
            )  # Shape: [B, 1, K]

            # Project and activate
            projected = rfsq.activation(
                rfsq.ln2(rfsq.out_proj(emb))
            )  # Shape: [B, 1, K]
            embeddings.append(projected)

        # Concatenate all layer embeddings
        return torch.hstack(embeddings)  # Shape: [B, N, K]

    def set_levels(self, levels: List[int], device: torch.device) -> None:
        """
        Set quantization levels for all ResidualFSQ layers.

        Args:
            levels (List[int]): List of quantization levels to set
            device (torch.device): Device to place the new levels on
        """
        for rfsq in self.residualfsqs:
            rfsq.set_levels(levels, device)

    def get_levels(self) -> List[List[int]]:
        """
        Get current quantization levels from all ResidualFSQ layers.

        Returns:
            List[List[int]]: List of quantization levels for each layer
        """
        return [rfsq.fsq._levels for rfsq in self.residualfsqs]
