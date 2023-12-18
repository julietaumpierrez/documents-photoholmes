import torch


def load_network_weight(net, weight_path: str, device: str | torch.device):
    net_state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(net_state_dict)
