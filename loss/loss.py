import torch


def calc_loss(out, y, args):

    # font_idxが一致している組は1, 一致していない組は-1
    sign = (y[:, 0] == y[:, 1])
    sign = torch.tensor([1 if s else -1 for s in sign], dtype=torch.float32)
    sign = sign.to(args.device)

    loss = torch.mean(sign.unsqueeze(1) * out)
    return - loss
