# Chapter 02: PPMI and SVD

## Goal

Implement basic distributional word representations from scratch.

## What this chapter covers

- Build a co-occurrence matrix from a corpus
- Convert co-occurrence counts into PPMI values
- Apply SVD to reduce high-dimensional word vectors
- Search for similar words using cosine similarity

## Key Ideas

PMI measures how strongly two words are associated:

PMI(x, y) = log2(P(x, y) / (P(x)P(y)))

PPMI removes negative PMI values:

PPMI(x, y) = max(0, PMI(x, y))

SVD compresses the high-dimensional PPMI matrix into lower-dimensional word vectors.
