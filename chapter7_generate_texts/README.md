# Chapter 7: Seq2Seq

This chapter introduces Seq2Seq models for sequence-to-sequence tasks.

Seq2Seq can be used for tasks such as addition, machine translation, dialogue systems, and text generation.

## Main Ideas

* Seq2Seq converts one sequence into another sequence.
* The Encoder reads the input sequence and converts it into a hidden state.
* The Decoder uses the hidden state to generate the output sequence.
* The Encoder and Decoder are connected through the hidden state `h`.
* During training, the Decoder uses the correct previous character as input.
* During inference, the Decoder uses its own previous prediction as input.
* Peeky Seq2Seq allows the Decoder to access the Encoder's hidden state at every time step.

## Seq2Seq Structure

The basic Seq2Seq model contains:

```text
Encoder
Decoder
SoftmaxWithLoss
```

The forward process is:

```text
input sequence
    ↓
Encoder
    ↓
hidden state h
    ↓
Decoder
    ↓
score
    ↓
SoftmaxWithLoss
```

## Encoder

The Encoder reads the input sequence and outputs the final hidden state.

Input:

```text
xs.shape = (N, T)
```

Output:

```text
h.shape = (N, H)
```

Where:

```text
N = batch size
T = input sequence length
H = hidden size
```

The hidden state `h` represents the information extracted from the input sequence.

## Decoder

The Decoder receives the Encoder's hidden state and generates the output sequence.

During training, the target sequence is split into:

```text
decoder input  = target[:, :-1]
decoder target = target[:, 1:]
```

Example:

```text
target:         _ 6 2
decoder input:  _ 6
decoder target: 6 2
```

The Decoder learns to predict the next character based on the previous correct character and the Encoder hidden state.

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

Inference does not use loss or backward propagation.

The Decoder starts generation from a start token such as:

```text
_
```

Then it generates one character at a time.

## Teacher Forcing

During training, the Decoder receives the correct previous character as input.

```text
input:  _
target: 6

input:  6
target: 2
```

This helps the model learn more efficiently.

During inference, the correct answer is not available, so the Decoder uses its own previous prediction.

## Reverse Input

The input sequence can be reversed before training:

```python
x_train = x_train[:, ::-1]
x_test = x_test[:, ::-1]
```

This can improve accuracy because important information may become closer to the Encoder's final hidden state.

## Peeky Seq2Seq

Peeky Seq2Seq allows the Decoder to directly access the Encoder's hidden state at each time step.

Basic Decoder:

```text
Decoder uses h only as the initial hidden state.
```

Peeky Decoder:

```text
Decoder uses h as:
1. the initial hidden state
2. part of the LSTM input
3. part of the Affine input
```

This gives the Decoder more direct access to the encoded input information.

## Data Preprocessing

The dataset is processed before training.

Main steps:

```text
1. read raw text data
2. split question and answer
3. build character vocabulary
4. convert characters to IDs
5. pad sequences to fixed length
6. split train and test data
```

The model does not directly process raw strings.  
It trains on numerical ID sequences.

## Shape Notes

For the addition task:

```text
x.shape = (N, input_length)
t.shape = (N, target_length)
```

The Decoder uses:

```text
decoder_xs.shape = (N, target_length - 1)
decoder_ts.shape = (N, target_length - 1)
```

In Peeky Decoder:

```text
h.shape       = (N, H)
hs.shape      = (N, T, H)
embed.shape   = (N, T, D)
concat.shape  = (N, T, H + D)
score.shape   = (N, T, V)
```

Where:

```text
D = word vector size
V = vocabulary size
```

## Key Takeaways

* Seq2Seq maps an input sequence to an output sequence.
* The Encoder compresses the input into a hidden state.
* The Decoder generates the output based on the hidden state.
* Teacher forcing is used during training.
* Inference generates one token at a time.
* Reversing the input sequence can improve performance.
* Peeky Seq2Seq gives the Decoder more direct access to Encoder information.
* Data preprocessing determines the input and target sequences used by the model.