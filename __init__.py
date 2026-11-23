from .nodes import MossImageCpp
from .sampler import MossImageCppSampler
from comfy_api.latest import ComfyExtension


class MossImageCppExtension(ComfyExtension):
    async def get_node_list(self):
        return [MossImageCppSampler, MossImageCpp]


async def comfy_entrypoint():
    return MossImageCppExtension()
