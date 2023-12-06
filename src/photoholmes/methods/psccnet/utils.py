import torch


def load_network_weight(net, weight_path, device):
    net_state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(net_state_dict)
