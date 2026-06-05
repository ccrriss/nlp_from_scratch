# Chapter 5: RNN

This chapter introduces Recurrent Neural Networks (RNNs) and language models.

## Main Ideas

* A language model predicts the next word based on previous words.
* CBOW uses surrounding words but does not strongly preserve word order.
* RNN processes sequence data step by step.
* RNN passes the hidden state from one time step to the next.
* The hidden state stores information from previous inputs.

## RNN Forward

A single RNN step can be written as:

```text
h_next = tanh(x @ Wx + h_prev @ Wh + b)
```

Where:

```text
x:      current input
h_prev: previous hidden state
Wx:     input-to-hidden weight
Wh:     hidden-to-hidden weight
b:      bias
h_next: next hidden state
```

Key idea:

```text
current input + previous hidden state -> next hidden state
```

## Shape Notes

```text
x.shape      = (N, D)
h_prev.shape = (N, H)

Wx.shape     = (D, H)
Wh.shape     = (H, H)
b.shape      = (H,)

h_next.shape = (N, H)
```

Where:

```text
N = batch size
D = input size / word vector size
H = hidden size
```

## TimeRNN

`TimeRNN` applies the same RNN layer across multiple time steps.

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
T = number of time steps
```

The forward process is:

```text
x_0 -> h_0
x_1 -> h_1
x_2 -> h_2
...
x_T -> h_T
```

Each time step uses the previous hidden state.

## Stateful RNN

If `stateful=False`, the hidden state is reset at each forward call.

If `stateful=True`, the final hidden state from the previous forward call is reused as the initial hidden state for the next forward call.

This is useful for language modeling because text is continuous.

## Backpropagation Through Time

RNN backward propagation goes through time in reverse order.

Forward:

```text
t = 0 -> 1 -> 2 -> ... -> T-1
```

Backward:

```text
t = T-1 -> T-2 -> ... -> 0
```

The gradient at each time step comes from:

```text
1. the current output
2. the next time step
```

## Truncated BPTT

For long sequences, backpropagating through the entire sequence is expensive.

Truncated BPTT splits a long sequence into shorter parts and performs backpropagation within each part.

The hidden state can still be carried forward, but the backward graph is truncated.

## Key Takeaways

* RNN is designed for sequence data.
* The hidden state connects different time steps.
* TimeRNN is an RNN layer unfolded over time.
* `stateful=True` allows the model to remember the previous batch's hidden state.
* RNN can be trained with Backpropagation Through Time.
* Truncated BPTT makes training long sequences more practical.
