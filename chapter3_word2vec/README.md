# Chapter 3: Word2Vec

This chapter introduces inference-based word representations using Word2Vec.

## Main Ideas

- Word2Vec learns word vectors through a prediction task.
- CBOW predicts the center word from surrounding context words.
- Skip-gram does the opposite: it predicts context words from the center word.
- One-hot vectors are used as input in the simple implementation.
- `W_in` is the input-side weight matrix, and each row becomes a word vector after training.

## CBOW Flow

```text
context words → W_in → hidden vector → W_out → score → SoftmaxWithLoss
```

## Key Notes

- `create_contexts_target()` creates context-target pairs.
- `convert_one_hot()` converts word IDs into one-hot vectors.
- `SoftmaxWithLoss` computes cross entropy loss.
- In this chapter, cross entropy is equivalent to negative log likelihood.
- Training updates `W_in`, so the final `W_in` contains learned word vectors.
