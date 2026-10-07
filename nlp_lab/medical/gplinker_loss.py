"""Match upstream GPLinker's mean-over-examples, sum-over-heads loss."""
from ..ie.model import pointer_loss


def sum_heads_objective(outputs,targets,use_tail=True):
    names=('entity','head','tail') if use_tail else ('entity','head')
    components={name:pointer_loss(outputs[name],targets[name])*outputs[name].shape[1] for name in names}
    return sum(components.values()),components
