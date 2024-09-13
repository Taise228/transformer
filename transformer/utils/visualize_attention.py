from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


def visualize_attn(src_morpheme, tgt_morpheme, attention, output_dir, img_name):
    """ Visualize attention weights
    
    Args:
        src_morpheme (List[str]): list of source morphemes
        tgt_morpheme (List[str]): list of target morphemes
        attention (List[torch.Tensor]): attention weights
            length: Number of layers
            each tensor shape: (num_heads, tgt_len, src_len)
        output_dir (str): output directory path
        img_name (str): image name
    """

    attn_mat = torch.stack(attention, dim=0).cpu()   # shape: (num_layers, num_heads, tgt_len, src_len)

    attn_mat = torch.mean(attn_mat, dim=1)   # shape: (num_layers, tgt_len, src_len)

    # add residual attention (re-normalize attention weights)
    residual_att = torch.eye(attn_mat.size(1))
    aug_att_mat = attn_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    attn_mat_np = joint_attentions[-1].detach().numpy()
    fig, ax = plt.subplots()
    cax = ax.matshow(attn_mat_np, cmap='bone')
    for (p, q), val in np.ndenumerate(attn_mat_np):
        ax.text(q, p, f'{val:.2f}', ha='center', va='center', color='black')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + src_morpheme, rotation=90)
    ax.set_yticklabels([''] + tgt_morpheme)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(f'{output_dir}/{img_name}.png')
    plt.close()
