from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from .mixins import PostProcessResultMixin

@MODELS.register_module()
class EncoderDecoderWithoutArgmax(PostProcessResultMixin, EncoderDecoder):
    pass