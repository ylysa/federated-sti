import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, constraints, models
from tensorflow import keras
# custom layers
num_heads = 8

class CrossAttentionLayer(layers.Layer):
  def __init__(self, local_dim, global_dim, n_heads,
               start_offset=0, end_offset=0,
               activation=tf.nn.gelu, dropout_rate=0.1,
               ):
    super(CrossAttentionLayer, self).__init__()
    self.local_dim = local_dim
    self.global_dim = global_dim
    self.dropout_rate = dropout_rate
    self.activation = activation
    self.start_offset = start_offset
    self.end_offset = end_offset
    self.num_heads = n_heads
    self.layer_norm00 = layers.LayerNormalization()
    self.layer_norm01 = layers.LayerNormalization()
    self.layer_norm1 = layers.LayerNormalization()
    self.ffn = tf.keras.Sequential(
          [
            layers.Dense(self.local_dim//2, activation=self.activation,
                        ),
            layers.Dense(self.local_dim,
                        activation=self.activation,
                        ), ]
      )
    self.add0 = layers.Add()
    self.add1 = layers.Add()
    self.attention = layers.MultiHeadAttention(num_heads=self.num_heads,
                                               key_dim=self.local_dim,)

  def call(self, inputs, training):
    local_repr = self.layer_norm00(inputs[0])
    global_repr = self.layer_norm01(inputs[1])
    query = local_repr[:, self.start_offset:local_repr.shape[1]-self.end_offset, :]
    key = global_repr
    value = global_repr

    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention_output = self.attention(
        query, key, value
    )
    # Skip connection 1.
    attention_output = self.add0([attention_output, query])

    # Apply layer norm.
    attention_output = self.layer_norm1(attention_output)
    # Apply Feedforward network.
    outputs = self.ffn(attention_output)
    # Skip connection 2.
    outputs = self.add1([outputs, attention_output])
    return outputs

class MaskedTransformerBlock(layers.Layer):
  def __init__(self, embed_dim, num_heads, ff_dim, attention_range, start_offset=0, end_offset=0, attn_block_repeats=1, activation=tf.nn.gelu, dropout_rate=0.1, use_ffn=True):
    super(MaskedTransformerBlock, self).__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.ff_dim = ff_dim
    self.start_offset = start_offset
    self.end_offset = end_offset
    self.attention_range = attention_range
    self.attn_block_repeats = attn_block_repeats
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.use_ffn = use_ffn
    self.att0 = [layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim) for _ in range(attn_block_repeats)]
    if self.use_ffn:
      self.ffn = [tf.keras.Sequential(
          [
            layers.Dense(self.ff_dim, activation=self.activation,
                        ),
            layers.Dense(self.embed_dim,
                        activation=self.activation,
                        ), ]
      ) for _ in range(attn_block_repeats)]
    self.layer_norm0 = [layers.LayerNormalization() for _ in range(attn_block_repeats)]
    self.layer_norm1 = [layers.LayerNormalization() for _ in range(attn_block_repeats)]

  def build(self, input_shape):
    assert(self.end_offset >= 0)
    self.feature_size = input_shape[1]
    attention_mask = np.zeros((self.feature_size,
                               self.feature_size), dtype=bool)
    for i in range(self.start_offset, self.feature_size - self.end_offset):
      attention_indices = np.arange(max(0, i-self.attention_range), min(self.feature_size, i+self.attention_range))
      attention_mask[i, attention_indices] = True
    self.attention_mask = tf.constant(attention_mask[self.start_offset:self.feature_size-self.end_offset])

  def call(self, inputs, training):

    x = inputs
    for i in range(self.attn_block_repeats-1):
      x = self.layer_norm0[i](x)
      attn_output = self.att0[i](x, x)
      out1 = x + attn_output
      out1 = self.layer_norm1[i](out1)
      if self.use_ffn:
        ffn_output = self.ffn[i](out1)
        x = out1 + ffn_output
      else:
        x = out1

    x = self.layer_norm0[-1](inputs)
    attn_output = self.att0[-1](x[:, self.start_offset:x.shape[1]-self.end_offset, :], x,
                            )
    out1 = x[:, self.start_offset:x.shape[1]-self.end_offset, :] + attn_output
    out1 = self.layer_norm1[-1](out1)
    if self.use_ffn:
      ffn_output = self.ffn[-1](out1)
      x = out1 + ffn_output
    else:
      x = out1
    return x

class GenoEmbeddings(layers.Layer):
  def __init__(self, embedding_dim,
               embeddings_initializer='glorot_uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None):
    super(GenoEmbeddings, self).__init__()
    self.embedding_dim = embedding_dim
    self.embeddings_initializer = initializers.get(embeddings_initializer)
    self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.embeddings_constraint = constraints.get(embeddings_constraint)

  def build(self, input_shape):
    # print(input_shape)

    self.num_of_allels = input_shape[-1]
    self.n_snps = input_shape[-2]
    self.position_embedding = layers.Embedding(
            input_dim=self.n_snps, output_dim=self.embedding_dim
        )
    self.embedding = self.add_weight(
            shape=(self.num_of_allels, self.embedding_dim),
            initializer=self.embeddings_initializer,
            trainable=True, name='geno_embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            experimental_autocast=False
        )
    self.positions = tf.range(start=0, limit=self.n_snps, delta=1)
  def call(self, inputs):
    self.immediate_result = tf.einsum('ijk,kl->ijl', inputs, self.embedding)
    return self.immediate_result + self.position_embedding(self.positions)


class Chunker(layers.Layer):
  def __init__(self, embed_dim, num_heads, ff_dim, chk_size, attention_range,
               activation=tf.nn.gelu, dropout_rate=0.25, attn_block_repeats=1,
               include_embedding_layer=False):
    super(Chunker, self).__init__()
    self.concat = layers.Concatenate(axis=-2)
    self.chunk_size = chk_size
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.ff_dim = ff_dim
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.attention_range = attention_range
    self.attn_block_repeats = attn_block_repeats
    self.include_embedding_layer = include_embedding_layer

  def build(self, input_shape):
    self.chunk_starts = list(range(0, input_shape[1], self.chunk_size))
    self.chunk_ends = []
    for cs in self.chunk_starts:
      self.chunk_ends.append(min(cs+self.chunk_size, input_shape[1]))
    self.mask_starts = [max(0, cs-self.attention_range) for cs in self.chunk_starts]
    self.mask_ends = [min(ce+self.attention_range, input_shape[1]) for ce in self.chunk_ends]
    self.chunkers = [SelfAttnChunk(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim,
                           attention_range=attention_range,
                           include_embedding_layer=self.include_embedding_layer,
                           start_offset=cs - self.mask_starts[i],
                            end_offset=self.mask_ends[i]-self.chunk_ends[i],
                           attn_block_repeats=self.attn_block_repeats) for i, cs in enumerate(self.chunk_starts)]

  def call(self, inputs, training):
    x = inputs
    chunks = [chunker(x[:, self.mask_starts[i]:self.mask_ends[i]]) for i, chunker in enumerate(self.chunkers)]
    y = self.concat(chunks)
    return y


class SelfAttnChunk(layers.Layer):
  def __init__(self, embed_dim, num_heads, ff_dim, attention_range,
               start_offset=0, end_offset=0,
               attn_block_repeats=1,
               include_embedding_layer=False):
    super(SelfAttnChunk, self).__init__()
    self.attention_range = attention_range
    self.ff_dim = ff_dim
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.attn_block_repeats = attn_block_repeats
    self.include_embedding_layer = include_embedding_layer

    self.attention_block = MaskedTransformerBlock(embed_dim=self.embed_dim,
                                                   num_heads=self.num_heads, ff_dim=self.ff_dim,
                                                   attention_range=attention_range, start_offset=start_offset,
                                                   end_offset=end_offset, attn_block_repeats=1)
    if include_embedding_layer:
      self.embedding = GenoEmbeddings(embed_dim)


  def build(self, input_shape):
    pass

  def call(self, inputs, training):
    if self.include_embedding_layer:
      x = self.embedding(inputs)
    else:
      x = inputs
    x = self.attention_block(x)
    return x

class CrossAttnChunk(layers.Layer):
  def __init__(self, n_heads, start_offset=0, end_offset=0):
    super(CrossAttnChunk, self).__init__()
    self.attention_range = 0
    self.start_offset = start_offset
    self.end_offset = end_offset
    self.n_heads = n_heads


  def build(self, input_shape):
    self.local_dim = input_shape[0][-1]
    self.global_dim = input_shape[1][-1]
    self.attention_block = CrossAttentionLayer(local_dim=self.local_dim, global_dim=self.global_dim,
                                              start_offset=self.start_offset, end_offset=self.end_offset,
                                              n_heads=self.n_heads
                                              )
    pass

  def call(self, inputs, training):
    x = inputs
    x = self.attention_block(x)
    return x

# Modules

class ConvBlock(layers.Layer):
  def __init__(self, embed_dim):
    super(ConvBlock, self).__init__()
    self.embed_dim = embed_dim
    self.const = None
    self.conv000 = layers.Conv1D(embed_dim, 3, padding='same', activation=tf.nn.gelu,
                                 kernel_constraint=self.const,
                    )
    self.conv010 = layers.Conv1D(embed_dim, 5, padding='same', activation=tf.nn.gelu,
                                 kernel_constraint=self.const,
                    )
    self.conv011 = layers.Conv1D(embed_dim, 7, padding='same', activation=tf.nn.gelu,
                                 kernel_constraint=self.const,
                    )

    self.conv020 = layers.Conv1D(embed_dim, 7, padding='same', activation=tf.nn.gelu,
                                 kernel_constraint=self.const,
                    )
    self.conv021 = layers.Conv1D(embed_dim, 15, padding='same', activation=tf.nn.gelu,
                                 kernel_constraint=self.const,
                    )
    self.add = layers.Add()

    self.conv100 = layers.Conv1D(embed_dim, 3, padding='same',
                                 activation=tf.nn.gelu,
                                 kernel_constraint=self.const,)
    self.bn0 = layers.BatchNormalization()
    self.bn1 = layers.BatchNormalization()
    self.dw_conv = layers.DepthwiseConv1D(embed_dim, 1, padding='same')
    self.activation = layers.Activation(tf.nn.gelu)

  def call(self, inputs, training):
    # Could add skip connection here?
    xa = self.conv000(inputs)

    xb = self.conv010(xa)
    xb = self.conv011(xb)

    xc = self.conv020(xa)
    xc = self.conv021(xc)

    xa = self.add([xb, xc])
    xa = self.conv100(xa)
    xa = self.bn0(xa)
    xa = self.dw_conv(xa)
    xa = self.bn1(xa)
    xa = self.activation(xa)
    return xa

def chunk_module(embed_dim, num_heads, input_len, input_channels, attention_range,
               start_offset=0, end_offset=0,
               attn_block_repeats=1, include_embedding=False):
  projection_dim = embed_dim
  inputs = layers.Input(shape=(input_len, embed_dim))
  xa = inputs
  xa0 = SelfAttnChunk(embed_dim=projection_dim, num_heads=num_heads, ff_dim=projection_dim//2, attention_range=attention_range,
            start_offset=start_offset, end_offset=end_offset, attn_block_repeats=1, include_embedding_layer=False)(xa)

  xa = ConvBlock(projection_dim)(xa0)
  xa_skip = ConvBlock(projection_dim)(xa)

  xa = layers.Dense(projection_dim, activation=tf.nn.gelu)(xa)
  xa = ConvBlock(projection_dim)(xa)
  xa = CrossAttnChunk(0, 0)([xa, xa0])
  xa = layers.Dropout(0.25)(xa)
  xa = ConvBlock(projection_dim)(xa)

  xa = layers.Concatenate(axis=-1)([xa_skip, xa])

  xa = layers.Conv1D(projection_dim//2, 5, padding='same', activation=tf.nn.gelu)(xa)

  xa_out1 = layers.Conv1D(2, 5, padding='same', activation=tf.nn.softmax)(xa)

  model = keras.Model(inputs=inputs, outputs=xa_out1)
  return model
