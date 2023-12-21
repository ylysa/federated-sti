import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
from models.layers import chunk_module, GenoEmbeddings

class SplitTransformer(keras.Model):
  def __init__(
      self,
      embed_dim,
      num_heads,
      # latent_dim = 512,
      chunk_size,
      attention_range,
      activation=tf.nn.gelu,
      dropout_rate=0.25,
      attn_block_repeats=1,
                ):
    super(SplitTransformer, self).__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.chunk_size = chunk_size
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.attn_block_repeats = attn_block_repeats
    self.attention_range = attention_range

  def build(self, input_shape):
    self.chunk_starts = list(range(0, input_shape[1], self.chunk_size))
    self.chunk_ends = []
    for cs in self.chunk_starts:
      self.chunk_ends.append(min(cs+self.chunk_size, input_shape[1]))
    self.mask_starts = [max(0, cs-self.attention_range) for cs in self.chunk_starts]
    self.mask_ends = [min(ce+self.attention_range, input_shape[1]) for ce in self.chunk_ends]
    self.chunkers = [chunk_module(self.embed_dim, self.num_heads,
                                  self.mask_ends[i] - self.mask_starts[i],
                                  3,
                                  self.attention_range,
                                  start_offset=cs - self.mask_starts[i],
                                  end_offset=self.mask_ends[i]-self.chunk_ends[i],
                                  #inChannel=3,
                                  attn_block_repeats=1, include_embedding=True) for i,cs in enumerate(self.chunk_starts)]

    self.concat_layer = layers.Concatenate(axis=-2)
    self.embedding = GenoEmbeddings(self.embed_dim)
    super(SplitTransformer, self).build(input_shape)


  def call(self, inputs):
    x = self.embedding(inputs)
    chunks = [self.chunkers[i](x[:, self.mask_starts[i]:self.mask_ends[i]]) for i, chunker in enumerate(self.chunkers)]
    y = self.concat_layer(chunks)
    return y