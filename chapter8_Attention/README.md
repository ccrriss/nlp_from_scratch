# Chapter 8: Attention

This chapter introduces the Attention mechanism for Seq2Seq models.

Attention allows the Decoder to directly access all Encoder hidden states, instead of relying only on the final Encoder hidden state.

## Main Ideas

* Basic Seq2Seq compresses the entire input sequence into one hidden state.
* This can make it difficult to handle long input sequences.
* Attention lets the Decoder look at all Encoder hidden states.
* At each Decoder time step, Attention calculates which Encoder positions are important.
* The context vector is computed as a weighted sum of Encoder hidden states.
* Attention improves the connection between input and output sequences.
* Attention weights can also be visualized to understand what the model is focusing on.

## Problem with Basic Seq2Seq

In a basic Seq2Seq model:

```text
input sequence
    ↓
Encoder
    ↓
final hidden state h
    ↓
Decoder
    ↓
output sequence
```

The Encoder only passes the final hidden state to the Decoder.

This means all input information must be compressed into one vector.

For long sequences, this can cause information loss.

## Attention Mechanism

Attention uses all Encoder hidden states:

```text
hs_enc.shape = (N, T_enc, H)
```

Where:

```text
N     = batch size
T_enc = Encoder time length
H     = hidden size
```

At each Decoder time step, the Decoder hidden state is compared with all Encoder hidden states.

The result is an attention weight:

```text
a.shape = (N, T_enc)
```

The attention weight represents how much the Decoder focuses on each Encoder time step.

## WeightSum

`WeightSum` calculates the context vector.

```text
hs_enc.shape = (N, T_enc, H)
a.shape      = (N, T_enc)
c.shape      = (N, H)
```

The context vector is a weighted sum of Encoder hidden states.

```text
c = sum(a * hs_enc)
```

## AttentionWeight

`AttentionWeight` calculates the attention weights.

Process:

```text
Decoder hidden state h
    ↓
compare with all Encoder hidden states
    ↓
score
    ↓
softmax
    ↓
attention weight a
```

Shape:

```text
hs_enc.shape = (N, T_enc, H)
h.shape      = (N, H)
a.shape      = (N, T_enc)
```

The softmax makes the attention weights sum to 1.

## Attention Layer

The `Attention` layer combines:

```text
AttentionWeight
WeightSum
```

Forward process:

```text
hs_enc, h_dec
    ↓
AttentionWeight
    ↓
attention weight a
    ↓
WeightSum
    ↓
context vector c
```

Backward process:

```text
dc
    ↓
WeightSum.backward()
    ↓
dhs_enc, da
    ↓
AttentionWeight.backward()
    ↓
dhs_enc, dh_dec
```

The gradients for Encoder hidden states are added together.

## TimeAttention

`TimeAttention` applies Attention to every Decoder time step.

```text
hs_enc.shape = (N, T_enc, H)
hs_dec.shape = (N, T_dec, H)
cs.shape     = (N, T_dec, H)
```

Where:

```text
T_enc = Encoder time length
T_dec = Decoder time length
```

Important point:

```text
attention weight length follows T_enc
context sequence length follows T_dec
```

For each Decoder time step:

```text
hs_dec[:, t, :] attends to all hs_enc
```

## Attention Encoder

The Attention Encoder returns all hidden states, not only the final hidden state.

Basic Seq2Seq Encoder:

```text
return hs[:, -1, :]
```

Attention Encoder:

```text
return hs
```

This is necessary because Attention needs all Encoder hidden states.

## Attention Decoder

The Attention Decoder uses:

```text
1. Decoder LSTM hidden states
2. Encoder hidden states
3. Attention context vectors
```

Forward process:

```text
decoder input
    ↓
Embedding
    ↓
LSTM
    ↓
decoder hidden states
    ↓
Attention with encoder hidden states
    ↓
context vectors
    ↓
concat(context, decoder hidden states)
    ↓
Affine
    ↓
score
```

Shape:

```text
context.shape = (N, T_dec, H)
hs_dec.shape  = (N, T_dec, H)

concat.shape  = (N, T_dec, 2H)
score.shape   = (N, T_dec, V)
```

Where:

```text
V = vocabulary size
```

## Training and Inference

During training:

```text
Encoder.forward()
Decoder.forward()
SoftmaxWithLoss.forward()
backward()
update parameters
```

During inference:

```text
Encoder.forward()
Decoder.generate()
```

In inference, the Encoder still processes the full input sequence:

```text
xs.shape     = (1, T_enc)
hs_enc.shape = (1, T_enc, H)
```

The Decoder generates one token at a time:

```text
current decoder output.shape = (1, 1, H)
```

At each generation step, the Decoder attends to the full Encoder hidden states.

## Important Shape Notes

```text
hs_enc.shape = (N, T_enc, H)
hs_dec.shape = (N, T_dec, H)
a.shape      = (N, T_enc)
c.shape      = (N, H)
cs.shape     = (N, T_dec, H)
score.shape  = (N, T_dec, V)
```

In the book's implementation:

```text
N and H should match between Encoder and Decoder.
T_enc and T_dec can be different.
```

## Key Takeaways

* Attention solves the limitation of compressing the input into only one vector.
* The Decoder can access all Encoder hidden states.
* Attention weights show which input positions are important.
* The context vector is a weighted sum of Encoder hidden states.
* `TimeAttention` applies Attention to every Decoder time step.
* In Attention Seq2Seq, the Encoder returns all hidden states.
* During inference, the model still uses the full Encoder sequence.
* Attention is an important foundation for modern NLP models.