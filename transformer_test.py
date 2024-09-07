from transformer.models.transformer import Transformer
from transformer.utils.visualize_attention import visualize_attn
from transformers import AutoTokenizer


if __name__ == '__main__':
    src_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tgt_tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    print(f'target vocab size: {len(tgt_tokenizer)}')

    model = Transformer(src_tokenizer, tgt_tokenizer, d_model=512, num_heads=8, d_ff=2048, device='cpu')

    src_sentence = 'Hello, how are you doing?'
    tgt_sentence = 'こんにちは、元気ですか？'

    src = src_tokenizer(src_sentence, return_tensors='pt')['input_ids']
    tgt = tgt_tokenizer(tgt_sentence, return_tensors='pt')['input_ids']

    print(src)
    print(tgt)

    src_morpheme = src_tokenizer.convert_ids_to_tokens(src[0])
    tgt_morpheme = tgt_tokenizer.convert_ids_to_tokens(tgt[0])
    print(tgt_morpheme)

    output = model(src, tgt)
    print(output['logits'].shape)   # shape: (batch_size, tgt_len, tgt_vocab_size)
    output_ids = output['logits'].argmax(-1)
    output_morpheme = tgt_tokenizer.convert_ids_to_tokens(output_ids[0])
    print(output_morpheme)

    # visualize
    src_attn = []
    for attn in output['encoder_attention']:
        src_attn.append(attn[0])   # first batch
    visualize_attn(src_morpheme, src_morpheme, src_attn, './results/encoder_attention')
