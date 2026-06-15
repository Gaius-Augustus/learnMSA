from pydantic import BaseModel


class VisualizationConfig(BaseModel):
    """Visualization parameters."""

    plot: str = ""
    """Produces a pdf of the learned HMM."""

    plot_head: int = -1
    """The HMM head to plot. If not set, the best model based on the model
    selection criterion will be plotted."""

    logo_gif: str = ""
    """Produces a gif that animates the learned sequence logo over
    training time. Slows down training significantly."""
