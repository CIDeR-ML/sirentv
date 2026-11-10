from .mse import WeightedL2Loss
from .cosine import WeightedCosineDissimilarity
from .poisson import WeightedPoissonNLLLoss
from .quantile_recon import WeightedReconstructedQuantileLoss
from .grad_frob import WeightedGradFrobLoss
from .js import JSDivergenceLoss
from .smooth_l1 import WeightedSmoothL1Loss, SmoothL1Loss
from .builder import build_loss
