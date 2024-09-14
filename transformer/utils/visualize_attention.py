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
    if src_morpheme == tgt_morpheme:
        residual_att = torch.eye(len(src_morpheme))
        aug_att_mat = attn_mat + residual_att
    else:
        aug_att_mat = attn_mat
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, attn_mat in enumerate(aug_att_mat):
        attn_mat_np = attn_mat.detach().numpy()   # shape: (tgt_len, src_len)
        fig, ax = plt.subplots()
        cax = ax.matshow(attn_mat_np, cmap='bone')
        for (p, q), val in np.ndenumerate(attn_mat_np):
            ax.text(q, p, f'{val:.2f}', ha='center', va='center', color='black')
        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(src_morpheme)))
        ax.set_yticks(np.arange(len(tgt_morpheme)))
        ax.set_xticklabels(src_morpheme, rotation=90)
        ax.set_yticklabels(tgt_morpheme)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.savefig(f'{output_dir}/{img_name}_layer{i}.png')
        plt.close()
