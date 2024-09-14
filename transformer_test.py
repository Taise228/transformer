import torch
from transformers import AutoTokenizer
from transformer.models.transformer import Transformer
from transformer.utils import get_config
from transformer.utils import visualize_attn


if __name__ == '__main__':
    config_path = './transformer/config/train_transfomrer.yml'
    config = get_config(config_path)

    src_tokenizer = AutoTokenizer.from_pretrained(config['model']['src_tokenizer'])
    tgt_tokenizer = AutoTokenizer.from_pretrained(config['model']['tgt_tokenizer'])

    model = Transformer(src_tokenizer, tgt_tokenizer, d_model=config['model']['d_model'], num_heads=config['model']['num_heads'],
                        d_ff=config['model']['d_ff'], N=config['model']['N'], dropout=config['model']['dropout'],
                        device=config['model']['device'], max_len=config['model']['max_len'], eps=config['model']['eps'])

    ckpt_path = './weights/best.pth'
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])

    src = 'Hello, world!.'
    src = src_tokenizer(src, return_tensors='pt')['input_ids'][0]
    output = model.inference(src)
    inf_morpheme = tgt_tokenizer.convert_ids_to_tokens(output['predictions'])[1:]
    print(inf_morpheme)

    # visualize
    src_morpheme = src_tokenizer.convert_ids_to_tokens(src)
    print(src_morpheme)

    visualize_attn(src_morpheme, src_morpheme, output['encoder_attention'], './results/encoder_attention', 'src_src')
    visualize_attn(src_morpheme, inf_morpheme, output['decoder_cross_attention'], './results/decoder_cross_attention', 'inf_src')
    visualize_attn(inf_morpheme, inf_morpheme, output['decoder_self_attention'], './results/decoder_self_attention', 'inf_inf')
