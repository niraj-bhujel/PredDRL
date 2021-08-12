import torch


def huber_loss(x, delta=1.):
    """

    Args:
        x: np.ndarray or tf.Tensor
            Values to compute the huber loss.
        delta: float
            Positive floating point value. Represents the
            maximum possible gradient magnitude.

    Returns: tf.Tensor
        The huber loss.
    """
    # delta = torch.ones_like(x) * delta
    
    less_than_max = 0.5 * torch.square(x)
    greater_than_max = delta * (torch.abs(x) - 0.5 * delta)

    return torch.where(torch.abs(x) <= delta, less_than_max, greater_than_max)
