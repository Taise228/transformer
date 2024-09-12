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
from
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

- half precision
- distributed learning
