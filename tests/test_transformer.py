import torch
import tests.transformer as transformer 
decoder_layer = transformer.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = transformer.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
print(out.shape)