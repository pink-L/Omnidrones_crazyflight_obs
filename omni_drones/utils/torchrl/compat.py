# Compatibility shim for torchrl API changes (0.3.x -> 0.11.x)
#
# torchrl >= 0.6 renamed the spec classes:
#   CompositeSpec          -> Composite
#   BoundedTensorSpec      -> Bounded
#   UnboundedContinuousTensorSpec -> UnboundedContinuous
#   DiscreteTensorSpec     -> Categorical
#   MultiDiscreteTensorSpec -> MultiCategorical
#
# This module re-exports under the old names so existing OmniDrones code
# continues to work without mass-renaming every file.

from torchrl.data import (
    Composite as CompositeSpec,
    Bounded as BoundedTensorSpec,
    UnboundedContinuous as UnboundedContinuousTensorSpec,
    Categorical as DiscreteTensorSpec,
    MultiCategorical as MultiDiscreteTensorSpec,
    TensorSpec,
)
