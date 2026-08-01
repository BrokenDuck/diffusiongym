"""Data codec: encode/decode between user-facing samples and model latent states."""

from collections.abc import Mapping
from typing import Any, Protocol

type Conditioning = Mapping[str, Any]


class DataCodec[RawT, StateT](Protocol):
    """Convert between user-facing samples and model latent states.

    Examples:
      - Toy data: identity codec (RawT = StateT = DDTensor)
      - SD3.5: VAE encoder/decoder (RawT = PIL image, StateT = latent DDTensor)
      - FlowMol: molecule graph to latent representation
    """

    def encode(self, raw: RawT, *, conditioning: Conditioning) -> StateT:
        """Encode a raw sample into the model's latent state."""
        ...

    def decode(self, latent: StateT, *, conditioning: Conditioning) -> RawT:
        """Decode a latent state into a user-facing sample."""
        ...


class IdentityCodec[StateT]:
    """No-op codec for models where the latent state is the raw sample."""

    def encode(self, raw: StateT, *, conditioning: Conditioning) -> StateT:
        return raw

    def decode(self, latent: StateT, *, conditioning: Conditioning) -> StateT:
        return latent
