import numpy as np
from torch import nn


class StringNormalizer(nn.Module):
    """ONNX StringNormalizer: remove stopwords from and change the case of a string tensor."""

    def __init__(
        self,
        case_change_action="NONE",
        is_case_sensitive=0,
        locale=None,
        stopwords=None,
    ):
        super().__init__()
        if case_change_action not in ("NONE", "LOWER", "UPPER"):
            raise NotImplementedError(
                "StringNormalizer with case_change_action={} not implemented.".format(
                    case_change_action
                )
            )
        self.case_change_action = case_change_action
        self.is_case_sensitive = bool(is_case_sensitive)
        self.locale = locale
        self.stopwords = tuple(stopwords or ())

    def forward(self, X: np.ndarray):
        x = np.asarray(X).astype(np.str_)
        if x.ndim == 1:
            return np.array(self._normalize(x.tolist()), dtype=object)
        elif x.ndim == 2 and x.shape[0] == 1:
            return np.array([self._normalize(x[0].tolist())], dtype=object)
        else:
            raise ValueError(
                "StringNormalizer expects a 1-D tensor or a 2-D tensor of shape "
                "[1, C], got shape {}.".format(tuple(x.shape))
            )

    def _normalize(self, words):
        if self.is_case_sensitive:
            stops = set(self.stopwords)
            words = [w for w in words if w not in stops]
        else:
            stops = set(w.lower() for w in self.stopwords)
            words = [w for w in words if w.lower() not in stops]

        if self.case_change_action == "LOWER":
            words = [w.lower() for w in words]
        elif self.case_change_action == "UPPER":
            words = [w.upper() for w in words]

        # All inputs filtered out, the output holds a single empty string
        return words or [""]

    def extra_repr(self) -> str:
        return (
            "case_change_action={}, is_case_sensitive={}, locale={}, "
            "stopwords={}".format(
                self.case_change_action,
                self.is_case_sensitive,
                self.locale,
                self.stopwords,
            )
        )
