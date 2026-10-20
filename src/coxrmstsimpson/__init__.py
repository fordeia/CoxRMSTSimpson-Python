"""RMST from Cox models using composite Simpson's rule."""
from .core import cox_rmst, simpson_rmst
from .plotting import plot_cox_rmst

__version__ = "0.2.2"
__all__ = ["cox_rmst", "simpson_rmst", "plot_cox_rmst"]
