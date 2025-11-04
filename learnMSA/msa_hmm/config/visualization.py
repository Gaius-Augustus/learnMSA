"""Visualization configuration parameters."""

from pydantic import BaseModel


class VisualizationConfig(BaseModel):
    """Visualization parameters."""

    logo: str = ""
    """Produces a pdf of the learned sequence logo."""

    logo_gif: str = ""
    """Produces a gif that animates the learned sequence logo over
    training time. Slows down training significantly."""
