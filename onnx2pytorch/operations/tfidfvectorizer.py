import torch
from torch import nn


class TfIdfVectorizer(nn.Module):
    """ONNX TfIdfVectorizer: n-gram counts of an integer sequence, optionally weighted."""

    def __init__(
        self,
        mode,
        ngram_counts,
        ngram_indexes,
        max_gram_length=1,
        max_skip_count=0,
        min_gram_length=1,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
    ):
        super().__init__()
        if pool_strings:
            raise NotImplementedError(
                "TfIdfVectorizer with pool_strings not implemented, "
                "string tensors have no PyTorch equivalent."
            )
        if mode not in ("TF", "IDF", "TFIDF"):
            raise NotImplementedError(
                "TfIdfVectorizer with mode={} not implemented.".format(mode)
            )
        self.mode = mode
        self.max_gram_length = max_gram_length
        self.max_skip_count = max_skip_count
        self.min_gram_length = min_gram_length
        self.ngram_indexes = ngram_indexes
        self.output_size = max(ngram_indexes) + 1
        self.weights = weights
        self.ngram_ids, self.prefixes = self.build_pool(
            [int(v) for v in pool_int64s or ()], ngram_counts
        )

    def build_pool(self, pool, ngram_counts):
        """Map each pooled n-gram onto its id and collect all of their prefixes."""
        ngram_ids = {}
        prefixes = set()
        ngram_id = 1
        for size, start in enumerate(ngram_counts, start=1):
            end = ngram_counts[size] if size < len(ngram_counts) else len(pool)
            n_grams = (end - start) // size
            if self.min_gram_length <= size <= self.max_gram_length:
                for i in range(n_grams):
                    gram = tuple(pool[start + i * size : start + (i + 1) * size])
                    ngram_ids[gram] = ngram_id
                    prefixes.update(gram[:j] for j in range(1, size + 1))
                    ngram_id += 1
            else:
                ngram_id += n_grams
        return ngram_ids, prefixes

    def count_row(self, row, frequencies, offset):
        start_gram_size = self.min_gram_length
        for skip_distance in range(1, self.max_skip_count + 2):
            for start in range(len(row)):
                if start + skip_distance * (start_gram_size - 1) >= len(row):
                    break

                gram = ()
                item = start
                while len(gram) < self.max_gram_length and item < len(row):
                    gram = gram + (row[item],)
                    if gram not in self.prefixes:
                        break
                    ngram_id = self.ngram_ids.get(gram)
                    if len(gram) >= start_gram_size and ngram_id is not None:
                        index = self.ngram_indexes[ngram_id - 1]
                        frequencies[offset + index] += 1
                    item += skip_distance

            # Unigrams are not affected by the skip distance, count them only once
            if start_gram_size == 1:
                start_gram_size += 1
                if start_gram_size > self.max_gram_length:
                    break

    def forward(self, input: torch.Tensor):
        rows = (
            input.reshape(-1, input.shape[-1])
            if input.ndim == 2
            else input.reshape(1, -1)
        )
        frequencies = [0] * (rows.shape[0] * self.output_size)
        if self.ngram_ids:
            for row_num, row in enumerate(rows.tolist()):
                self.count_row(row, frequencies, row_num * self.output_size)

        output = torch.tensor(frequencies, dtype=torch.float32)
        if self.mode == "IDF":
            output = (output > 0).to(torch.float32)
        if self.weights and self.mode in ("IDF", "TFIDF"):
            output = output * torch.tensor(self.weights, dtype=torch.float32).repeat(
                rows.shape[0]
            )

        if input.ndim == 2:
            return output.reshape(rows.shape[0], self.output_size)
        return output
