# Chapter 4: Word2Vec Speedup

This chapter improves the simple Word2Vec implementation from Chapter 3.

## Main Ideas

- One-hot vectors are inefficient for large vocabularies.
- `one_hot @ W_in` is equivalent to selecting one row from `W_in`.
- The Embedding layer directly retrieves word vectors by word IDs.
- Full softmax over the entire vocabulary is expensive.
- Negative sampling replaces multi-class classification with several binary classification tasks.
- The model learns to distinguish the correct target word from sampled negative words.

## Key Components

- `Embedding`: retrieves word vectors directly from `W_in`.
- `EmbeddingDot`: computes the dot product between hidden vectors and target word vectors.
- `UnigramSampler`: samples negative words based on word frequency.
- `NegativeSamplingLoss`: computes loss for one positive sample and several negative samples.
- Improved CBOW: uses Embedding and Negative Sampling for faster training.

## Key Notes

- `np.add.at()` is used in `Embedding.backward()` to correctly accumulate gradients for repeated word IDs.
- `W_in` is still used as the final word vector matrix after training.
- Negative sampling avoids computing scores for all vocabulary words.
  