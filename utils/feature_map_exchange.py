import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class FeatureMapExchange(object):
    def __init__(self):
        super(FeatureMapExchange, self).__init__()

    @staticmethod
    def channel_swap(t1, t2, threshold=0.4, prob=1/2) -> tuple:
        """measure the similarity on the channel dimension to determine which channels to swap"""
        n, c, h, w = t1.shape
        sim = cosine_similarity(t1, t2, dim=1) # n, h, w
        sim = sim.unsqueeze(0) # c, n, h, w
        expand_tensor = sim.clone()

        # use concatenation to build the same dimensions
        for _ in range(c-1): # c, n, h, w
            sim = torch.cat([sim, expand_tensor], dim=0)

        # create a logical mask, and set the mask to True
        # for the channels that are less than the threshold
        step = int(1/prob)
        exchange_map = torch.arange(c) % step == 0
        step_mask = exchange_map.unsqueeze(0).unsqueeze(2)\
                                .unsqueeze(3).expand((n, c, h, w))

        mask = sim < threshold

        device = mask.device
        step_mask = step_mask.to(device)
        mask = mask.permute(1, 0, 2, 3)
        
        # swap
        t1[~step_mask&mask,...], t2[~step_mask&mask,...] =\
            t2[~step_mask&mask,...], t1[~step_mask&mask,...]

        return t1, t2
        
    @staticmethod
    def spatial_swap(t1, t2, threshold=0.5, prob=1/2) -> tuple:
        """measure the similarity on the channel dimension to determine which hw to swap"""
        n, c, h, w = t1.shape
        sim = cosine_similarity(t1, t2, dim=1) # n, h, w
        sim = sim.unsqueeze(0) # c, n, h, w
        expand_tensor = sim.clone()

        # use concatenation to build the same dimensions
        for _ in range(c-1): # c, n, h, w
            sim = torch.cat([sim, expand_tensor], dim=0)

        sim = sim.permute(1, 0, 2, 3) # n, c, h, w

        # create a logical mask, and set the mask to True
        # for the channels that are less than the threshold
        step = int(1/prob)
        step_mask = torch.arange(w) % step == 0
        # align the dimensions of the mask for mask computation
        step_mask = step_mask.unsqueeze(0).unsqueeze(0)\
                            .unsqueeze(0).expand((n, c, h, w)) 
        
        mask = sim < threshold

        device = mask.device
        step_mask = step_mask.to(device)

        # swap
        t1[..., ~step_mask&mask], t2[..., ~step_mask&mask] =\
            t2[..., ~step_mask&mask], t1[..., ~step_mask&mask]

        return t1, t2

if __name__ == "__main__":
    tensor1 = torch.rand((4, 32, 256, 256))
    tensor2 = torch.rand((4, 32, 256, 256))

    tensor1, tensor2 = FeatureMapExchange.channel_swap(tensor1, tensor2)
    tensor1, tensor2 = FeatureMapExchange.channel_swap(tensor1, tensor2)