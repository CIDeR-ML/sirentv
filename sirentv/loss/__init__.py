from .mse import WeightedL2Loss, L2Loss, UncertainMSE, VisibilityGradient_L2Loss
from .cosine import WeightedCosineDissimilarity
from .poisson import WeightedPoissonNLLLoss
from .reg import L2Regularization, L1Regularization, LNRegularization
from .js import JSDivergenceLoss
from .smooth_l1 import WeightedSmoothL1Loss, SmoothL1Loss, VisibilityGradient_SmoothL1Loss
from .builder import build_loss, build_regularizer