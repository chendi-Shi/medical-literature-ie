import pytest

torch = pytest.importorskip("torch")
from nlp_lab.algorithms import embedding_direction, perturb_embeddings, symmetric_kl


def test_symmetric_kl_matches_distribution_reference_and_backpropagates():
    p = torch.tensor([[1., 2., -1.], [2., 0., 3.]], requires_grad=True)
    q = torch.tensor([[.5, 1., -2.], [1., 2., 3.]], requires_grad=True)
    first = torch.distributions.Categorical(logits=p)
    second = torch.distributions.Categorical(logits=q)
    reference = .5 * (torch.distributions.kl_divergence(first, second) + torch.distributions.kl_divergence(second, first)).mean()
    loss = symmetric_kl(p, q)
    assert loss.detach() == pytest.approx(float(reference.detach()), abs=1e-6)
    assert float(symmetric_kl(p, p).detach()) == pytest.approx(0, abs=1e-7)
    loss.backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
    assert p.grad.abs().sum() > 0 and q.grad.abs().sum() > 0


class Toy:
    def __init__(self):
        self.embeddings = torch.nn.Embedding(3, 2)
        with torch.no_grad():
            self.embeddings.weight.fill_(1.)

    def get_input_embeddings(self):
        return self.embeddings


def test_fgm_uses_current_gradient_and_restores_after_exception():
    model = Toy()
    weight = model.embeddings.weight
    weight.grad = torch.full_like(weight, 999.)  # previous accumulation window
    loss = model.embeddings(torch.tensor([0])).square().sum()
    direction = embedding_direction(model, loss)
    assert direction[1:].abs().sum() == 0
    assert weight.grad[1, 0] == 999
    loss.backward()
    original = weight.detach().clone()
    with pytest.raises(RuntimeError, match="forced"):
        with perturb_embeddings(model, direction, .3):
            assert float(torch.linalg.vector_norm(weight - original)) == pytest.approx(.3, abs=1e-6)
            assert torch.equal(weight[1:], original[1:])
            raise RuntimeError("forced")
    assert torch.equal(weight, original)


def test_zero_gradient_attack_is_noop():
    model = Toy()
    original = model.embeddings.weight.detach().clone()
    with perturb_embeddings(model, torch.zeros_like(original), .5) as applied:
        assert not applied
    assert torch.equal(model.embeddings.weight, original)
