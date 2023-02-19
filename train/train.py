from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from loss.loss import calc_loss


def train_style_discriminator(
        data_loader,
        networks,
        opts,
        epoch,
        args):
    # set nets
    D = networks['D']

    # set opts
    d_opt = opts['D']

    # switch to train mode
    D.train()

    it = iter(data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)
    for i in t_train:
        try:
            img, font_idx, cnt_idx = next(it)
        except BaseException:
            it = iter(data_loader)
            img, font_idx, cnt_idx = next(it)

        # img.shape is [batch_size, 2, img_size, img_size]
        # font_idx.shape is [batch_size, 2]
        # cnt_idx.shape is [batch_size, 2]

        random_idx = torch.randperm(img.size(0))
        random_img = img[random_idx]
        random_img[:, 1] = img[:, 1]
        random_font_idx = font_idx[random_idx]
        random_font_idx[:, 1] = font_idx[:, 1]
        random_cnt_idx = cnt_idx[random_idx]
        random_cnt_idx[:, 1] = cnt_idx[:, 1]

        # - x: images of shape (batch, 2, image_size, image_size).
        # - y: domain indices of shape (batch).

        # out = D(img)
        # random_out = D(random_img)
        # loss = calc_loss(out, font_idx) + \
        #     calc_loss(random_out, random_font_idx)

        total_img = torch.cat([img, random_img], dim=0)
        total_font_idx = torch.cat([font_idx, random_font_idx], dim=0)
        total_img = total_img.to(args.device)
        total_font_idx = total_font_idx.to(args.device)
        out = D(total_img)
        loss = calc_loss(out, total_font_idx, args)

        d_opt.zero_grad()
        loss.backward()
        d_opt.step()
        if (i + 1) % args.log_step == 0:
            print(f'epoch: {epoch + 1}, iter: {i}, loss: {loss.item()}')
