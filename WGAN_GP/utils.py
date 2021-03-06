import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device='cpu'):
    # BATCH_SIZE, C, H, W = real.shape
    # epsilon = torch.randn((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)  # one epsilon value for each example
    # interpolated_image = real * epsilon + fake * (1 - epsilon)

    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_image = real * alpha + fake * (1 - alpha)

    # calculate critic scores
    mixed_scores = critic(interpolated_image)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_image,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
