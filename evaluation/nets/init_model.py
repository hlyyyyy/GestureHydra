import torch

from nets.embedding_net import TrainWrapper


def init_model(args, pretrained=False, model_path=None):
    generator = TrainWrapper(args)
    if pretrained:
        model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        generator.load_state_dict(model_ckpt['generator'])
    return generator
