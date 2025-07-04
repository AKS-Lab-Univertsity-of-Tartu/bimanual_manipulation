"""Damping task implementation."""

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .posture_task import PostureTask


class DampingTask(PostureTask):
    r"""L2-regularization on joint velocities (a.k.a. *velocity damping*).

    This task, typically used with a low priority in the task stack, adds a
    Levenberg-Marquardt term to the quadratic program, favoring **minimum-norm
    joint velocities** in redundant or near-singular situations. Formally, it
    contributes the following term to the quadratic program:

    .. math::
        \frac{1}{2}\,\Delta \mathbf{q}^\top \Lambda\,\Delta \mathbf{q},

    where :math:`\Delta \mathbf{q}\in\mathbb{R}^{n_v}` is the vector of joint
    displacements and :math:`\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_{n_v})`
    is a diagonal matrix of per-DoF damping weights specified via ``cost``. A larger
    :math:`\lambda_i` reduces motion in DoF :math:`i`. With no other active
    tasks, the robot remains at rest. Unlike the `damping` parameter in
    :func:`~.solve_ik`, which is uniformly applied to all DoFs, this task does
    not affect the floating-base coordinates.

    .. note::

        This task does not favor a particular posture, only small instantaneous motion.
        If you need a posture bias, use :class:`~.PostureTask` instead.

    Example:

    .. code-block:: python

        # Uniform damping across all degrees of freedom.
        damping_task = DampingTask(model, cost=1.0)

        # Custom damping.
        cost = np.zeros(model.nv)
        cost[:3] = 1.0  # High damping for the first 3 joints.
        cost[3:] = 0.1  # Lower damping for the remaining joints.
        damping_task = DampingTask(model, cost)
    """

    def __init__(self, model: mujoco.MjModel, cost: npt.ArrayLike):
        super().__init__(model, cost, gain=0.0, lm_damping=0.0)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the damping task error.

        The damping task does not chase a reference; its desired joint velocity
        is identically **zero**, so the task error is always:

        .. math:: e = \mathbf 0 \in \mathbb R^{n_v}.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Zero vector of length :math:`n_v`.
        """
        return np.zeros(configuration.nv)
