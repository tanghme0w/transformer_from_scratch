from torch import nn
from component import Encoder, EncoderLayer, Generator


class EncoderDecoder(nn.Module):

    def __init__(self, src_vocab, tgt_vocab, N, emb_size, d_ff, h, dropout):
        super().__init__()
        self.encoder = Encoder(EncoderLayer(emb_size=emb_size, d_ff=d_ff, dropout=dropout, h=h), N)
        # TODO decoder setup
        # TODO positional embedding
        # generator setup
        self.generator = Generator(out_dim=emb_size, vocab_size=tgt_vocab)  # generate distribution when inference
        # initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.src_embed(src)
        tgt_emb = self.tgt_embed(tgt)
        return self.decoder(
            tgt_emb,
            self.encoder(src_emb, src_mask),
            src_mask,
            tgt_mask
        )

