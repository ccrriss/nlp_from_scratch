# Chapter 6: LSTM and Better RNN Language Model

This chapter improves the basic RNN language model by introducing LSTM, dropout, weight tying, and gradient clipping.

## Main Ideas

* A basic RNN has difficulty learning long-term dependencies.
* LSTM uses gates and a memory cell to keep important information for a longer time.
* `TimeLSTM` applies the same LSTM layer across multiple time steps.
* Dropout helps reduce overfitting during training.
* Weight tying shares the embedding weight with the output affine layer.
* Gradient clipping helps prevent exploding gradients.
* Perplexity is used to evaluate language models.

## LSTM

LSTM keeps two states:

```text
h: hidden state
c: cell state
```

The hidden state `h` is used as the output for each time step.

The cell state `c` stores longer-term information.

LSTM uses gates to control information flow:

```text
forget gate: decides what to forget
input gate:  decides what to add
output gate: decides what to output
```

Key idea:

```text
previous state + current input -> next hidden state and cell state
```

## TimeLSTM

`TimeLSTM` applies an LSTM layer over multiple time steps.

Input:

```text
xs.shape = (N, T, D)
```

Output:

```text
hs.shape = (N, T, H)
```

Where:

```text
N = batch size
T = number of time steps
D = input size / word vector size
H = hidden size
```

The forward process is:

```text
x_0 -> h_0
x_1 -> h_1
x_2 -> h_2
...
x_T -> h_T
```

Each time step passes both the hidden state and cell state to the next time step.

## Stateful LSTM

If `stateful=False`, the hidden state and cell state are reset at each forward call.

If `stateful=True`, the final states from the previous forward call are reused in the next forward call.

This is useful for language modeling because text is continuous.

## BetterRnnlm

The improved language model uses the following structure:

```text
TimeEmbedding
TimeDropout
TimeLSTM
TimeDropout
TimeLSTM
TimeDropout
TimeAffine
TimeSoftmaxWithLoss
```

Compared with the basic RNNLM, this model uses:

```text
1. LSTM instead of simple RNN
2. two LSTM layers
3. dropout
4. weight tying
5. gradient clipping
```

## Dropout

Dropout randomly removes some activations during training.

During training:

```text
train_flg = True
```

During evaluation or prediction:

```text
train_flg = False
```

Dropout is used only during training.

## Weight Tying

Weight tying shares the input embedding weight and the output affine weight.

```text
Embedding weight: embed_W
Affine weight:    embed_W.T
```

If:

```text
embed_W.shape = (V, D)
```

Then:

```text
embed_W.T.shape = (D, V)
```

Because of this, weight tying requires:

```text
wordvec_size == hidden_size
```

Where:

```text
V = vocabulary size
D = word vector size
H = hidden size
```

## remove_duplicate

When weight tying is used, the same weight appears in the parameter list more than once.

`remove_duplicate()` merges shared weights and their gradients before updating parameters.

This prevents the same weight from being updated incorrectly.

## Gradient Clipping

RNN and LSTM can suffer from exploding gradients.

Gradient clipping limits the size of gradients during training.

```text
if gradient norm is too large:
    scale gradients down
```

This makes training more stable.

## Perplexity

Perplexity measures how well a language model predicts the next word.

```text
perplexity = exp(loss)
```

Lower perplexity means better prediction.

## Key Takeaways

* LSTM improves RNN by using gates and a memory cell.
* `TimeLSTM` processes sequence data over multiple time steps.
* `stateful=True` allows the model to carry states across batches.
* Dropout helps reduce overfitting.
* Weight tying reduces parameters by sharing embedding and output weights.
* `remove_duplicate()` is needed when shared weights are used.
* Gradient clipping helps stabilize RNN/LSTM training.
* Perplexity is the main evaluation metric for language models.

## Bug Note: Perplexity Issue

While training the RNN/LSTM language model, perplexity became abnormal.

Cause:
A variable name was mistakenly written as a gradient variable name.

Lesson:
Deep learning code may run without errors even when the computation graph is wrong.
When perplexity is abnormal, check:

* variable names
* forward/backward consistency
* params and grads order
* whether optimizer updates the correct parameters