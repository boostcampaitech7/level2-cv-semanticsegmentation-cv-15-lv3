from mmseg.registry import MODELS
from mmseg.models.decode_heads import ASPPHead, FCNHead, SegformerHead
from .mixins import LossByFeatMixIn

@MODELS.register_module()
class ASPPHeadWithoutAccuracy(LossByFeatMixIn, ASPPHead):
    pass

@MODELS.register_module()
class FCNHeadWithoutAccuracy(LossByFeatMixIn, FCNHead):
    pass

@MODELS.register_module()
class SegformerHeadWithoutAccuracy(LossByFeatMixIn, SegformerHead):
    pass