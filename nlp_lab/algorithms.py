"""R-Drop and FGM components, independent of data and experiment UI."""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def symmetric_kl(first, second):
    """Mean per-example symmetric KL; sum classes, average the two directions."""
    log_p = F.log_softmax(first.float(), dim=-1)
    log_q = F.log_softmax(second.float(), dim=-1)
    return .5 * ((log_p.exp() * (log_p - log_q)).sum(-1) +
                 (log_q.exp() * (log_q - log_p)).sum(-1)).mean()


def clean_objective(model, batch, regularization="ce", rdrop_alpha=.5):
    first = model(**batch)
    ce = first.loss.float()
    kl = ce.new_zeros(())
    if regularization in ("rdrop", "rdrop_fgm"):
        second = model(**batch)  # independent dropout, same labeled examples
        ce = .5 * (ce + second.loss.float())
        kl = symmetric_kl(first.logits, second.logits)
    return ce + rdrop_alpha * kl, {"clean_ce": float(ce.detach()), "symmetric_kl": float(kl.detach())}


def embedding_direction(model, clean_loss):
    """Current microbatch's gradient, independent of accumulated .grad buffers."""
    weights = model.get_input_embeddings().weight
    direction = torch.autograd.grad(clean_loss, weights, retain_graph=True, allow_unused=False)[0].detach()
    if not torch.isfinite(direction).all():
        raise ValueError("FGM 梯度含非有限值")
    return direction


@contextmanager
def perturb_embeddings(model, direction, epsilon):
    """One global L2-bounded embedding-matrix step; always restore, even on failure."""
    weights = model.get_input_embeddings().weight
    norm = torch.linalg.vector_norm(direction.float())
    if float(norm) == 0:
        yield False
        return
    original = weights.detach().clone()
    try:
        with torch.no_grad():
            weights.add_(epsilon * direction / norm)
        yield True
    finally:
        with torch.no_grad():
            weights.copy_(original)
