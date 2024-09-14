# transformer

This is educational, my own implementation of transformer.   
Original paper: [Attention Is All You Need (Vaswani et al, 2017)](https://arxiv.org/abs/1706.03762)   
I ([@Taise228](https://github.com/Taise228)) read the paper, learned, and wrote all the codes from scratch, using GitHub Copilot.   

# Pretrain

I used [JSEC Dataset](https://nlp.stanford.edu/projects/jesc/) to pretrain the model. Below is a citation for JSEC Dataset:

```
@ARTICLE{pryzant_jesc_2018,
   author = {{Pryzant}, R. and {Chung}, Y. and {Jurafsky}, D. and {Britz}, D.},
    title = "{JESC: Japanese-English Subtitle Corpus}",
  journal = {Language Resources and Evaluation Conference (LREC)},
 keywords = {Computer Science - Computation and Language},
     year = 2018
}  
```

# Usage

## Inference

```python
from transformers import AutoTokenizer
from transformer.models.transformer import Transformer
from transformer.utils import get_config

config = get_config(config_path)   # you have to set

src_tokenizer = AutoTokenizer.from_pretrained(config['model']['src_tokenizer'])
tgt_tokenizer = AutoTokenizer.from_pretrained(config['model']['tgt_tokenizer'])

model = Transformer(src_tokenizer, tgt_tokenizer, d_model=config['model']['d_model'], num_heads=config['model']['num_heads'],
                    d_ff=config['model']['d_ff'], N=config['model']['N'], dropout=config['model']['dropout'],
                    device=config['model']['device'], max_len=config['model']['max_len'], eps=config['model']['eps'])

checkpoint = torch.load(ckpt_path)   # you have to set
model.load_state_dict(checkpoint['model'])

src = 'Hello, world!.'
src = src_tokenizer(src, return_tensors='pt')['input_ids']
output = model.inference(src)
inf_morpheme = tgt_tokenizer.convert_ids_to_tokens(output['predictions'][0])
print(inf_morpheme)

# visualize
src_attn = []
for attn in output['encoder_attention']:
    src_attn.append(attn[0])   # first batch
visualize_attn(src_morpheme, src_morpheme, src_attn, './results/encoder_attention', 'sample')
```

## Training

### training for machine translation task

This is mainly for pretraining.

```bash
python train_transformer.py --config ./transformer/config/train_transfomrer.yml 
```

### fine-tuning for tasks you want

1. Build your own model using [TransformerEncoder](/transformer/models/transformer_encoder.py) and [TransformerDecoder](/transformer/models/transformer_decoder.py).
2. Prepare your own script to run training, like [train_transformer.py](/train_transformer.py). In that, you can load pretrained weights.

# TODO

- publish pretrained weights
- distributed learning
