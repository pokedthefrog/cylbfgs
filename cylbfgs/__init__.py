"""
L-BFGS and OWL-QN optimisation methods.
"""

import warnings
from typing import Optional, Callable

import numpy as np
import numpy.typing as npt

from ._lowlevel import LBFGS, LBFGSError, LINE_SEARCH_ALGO


def fmin_lbfgs(
        fun: Callable,
        x0: npt.NDArray[np.float64],
        args: tuple = (),
        callback: Optional[Callable] = None,
        m: int = 6,
        epsilon: float = 1e-5,
        past: int = 0,
        delta: float = 1e-5,
        max_iterations: int = 0,
        linesearch: str = 'default',
        max_linesearch: int = 40,
        min_step: float = 1e-20,
        max_step: float = 1e+20,
        ftol: float = 1e-4,
        wolfe: float = 0.9,
        gtol: float = 0.9,
        cbfgs_epsilon: float = 1e-6,
        cbfgs_alpha: float = 1.,
        orthantwise_c: float = 0.,
        orthantwise_w: Optional[npt.NDArray[np.float64]] = None,
        orthantwise_start: int = 0,
        orthantwise_end: int = -1,
) -> dict:
    """Minimise a function using the L-BFGS or OWL-QN methods.

    Parameters
    ----------
    fun
        Computes the objective function to be minimised and its gradient.

            ``fun(x, g, *args) -> float``,

        where ``x`` and ``g`` are 1-D arrays of shape (n,) that hold the current
        variable values and the gradient, respectively, and ``args`` is a tuple
        of additional, fixed arguments needed to completely specify the function.
        It must return the current (i.e., evaluated at ``x``) objective function
        value and store its current gradient in ``g``.

    x0
        Initial guess; a real 1-D array of shape (n,), where ``n`` is the number
        of independent variables. A copy is made prior to optimisation.

    args
        A tuple of additional arguments to be passed to ``fun`` and ``callback``
        in order to completely specify the functions.

    callback
        A callable to be called after each iteration.

            ``callback(x, g, fx, xnorm, gnorm, step, k, ls) -> int``,

        where ``x`` and ``g`` are 1-D arrays of shape (n,) that hold the current
        variable and gradient values, respectively, ``fx`` is the current
        objective function value, ``xnorm`` and ``gnorm`` are the Euclidean (L2)
        norms of ``x`` and ``g``, ``step`` is the line search step used for this
        iteration, ``k`` is the iteration count, and ``ls`` is the number of
        evaluations called during the iteration. Returning any value that is not
        `None` or zero will terminate the optimisation process.

    m
        The number of corrections to approximate the inverse hessian matrix. The
        L-BFGS routine stores the computation results of the previous ``m``
        iterations to approximate the inverse hessian matrix of the current one.
        Values less than `3` are not recommended. Large values will result in
        excessive computing time.

    epsilon
        Tolerance for gradient-based convergence test. This parameter determines
        the accuracy with which the solution is to be found. A minimisation
        terminates when:

            ``∥g∥ < epsilon * max(1, ∥x∥)``,

        where ``∥.∥`` denotes the Euclidean (L2) norm, and ``x``, ``g`` are the
        current variable and gradient values.

    past
        Distance for delta-based convergence test. It determines the distance,
        in iterations, to compute the rate of decrease of the objective function.
        If the value of this parameter is zero, the library does not perform the
        delta-based convergence test.

    delta
        Tolerance for delta-based convergence test. It determines the minimum
        rate of decrease of the objective function. The library stops iterations
        when the following condition is met:

            ``(f' - f) / f < delta``,

        where ``f'`` is the objective value of ``past`` iterations ago, and ``f``
        is the objective value of the current iteration.

    max_iterations
        The maximum number of iterations. The optimisation process terminates
        with the maximum iterations status code when the iteration count exceeds
        this parameter. Setting this parameter to zero continues an optimisation
        process until a convergence or an error.

    linesearch
        The line search algorithm to be used by the L-BFGS routine. Valid values
        are the following:

        - `default`: Same as `more_thuente`.
        - `more_thuente`: Method proposed by Moré and Thuente.
        - `backtracking_armijo`: Backtracking with the Armijo condition.
        - `backtracking`: Same as `backtracking_wolfe`.
        - `backtracking_wolfe`: Backtracking with the (regular) Wolfe conditions.
        - `backtracking_strong_wolfe`: Backtracking with the strong Wolfe
          conditions.

        If the OWL-QN method is invoked (i.e., ``orthantwise_c != 0``), only the
        backtracking line search with the Armijo condition is available. This is
        specified as `backtracking` or `backtracking_armijo`.

    max_linesearch
        The maximum number of trials for the line search. This parameter
        controls the number of function and gradient evaluations, per iteration,
        for the line search routine.

    min_step
        The minimum step of the line search routine. This value need not be
        modified unless the exponent is too small for the machine being used, or
        the problem is extremely badly scaled (in which case the exponent should
        be decreased).

    max_step
        The maximum step of the line search routine. This value need not be
        modified unless the exponent is too large for the machine being used, or
        the problem is extremely badly scaled (in which case the exponent should
        be increased).

    ftol
        A parameter to control the accuracy of the line search routine. It must
        be greater than zero and less than `0.5`.

    wolfe
        A coefficient for the Wolfe conditions. This parameter is only valid
        when the backtracking line search algorithm is used with the (regular or
        strong) Wolfe conditions. It must be greater than ``ftol`` and less than
        `1.0`.

    gtol
        A parameter to control the accuracy of the line search routine.
        If the function and gradient evaluations are inexpensive with respect to
        the cost of the iteration (which is sometimes the case when solving very
        large problems) it may be advantageous to set this to a small value. A
        typical small value is `0.1`. This parameter must be greater than ``ftol``
        and less than `1.0`.

    cbfgs_epsilon
        Coefficient for the cautious BFGS (C-BFGS) update rule. This parameter
        must be a small positive number. At iteration ``k``, the limited BFGS
        memory update is skipped when the following condition is not satisfied:

            ``y_k^T s_k >= cbfgs_epsilon * ∥g_k∥^cbfgs_alpha * ∥s_k∥^2``,

        where ``∥.∥`` denotes the Euclidean (L2) norm, ``s_k = x_{k+1} - x_k``,
        and ``y_k = g_{k+1} - g_k``.

    cbfgs_alpha
        Exponent for the cautious BFGS (C-BFGS) update rule. This parameter must
        be a positive number. At iteration ``k``, the limited BFGS memory update
        is skipped when the following condition is not satisfied:

            ``y_k^T s_k >= cbfgs_epsilon * ∥g_k∥^cbfgs_alpha * ∥s_k∥^2``,

        where ``∥.∥`` denotes the Euclidean (L2) norm, ``s_k = x_{k+1} - x_k``,
        and ``y_k = g_{k+1} - g_k``.

    orthantwise_c
        Coefficient for the L1 norm of variables. This should be set to zero for
        standard minimisation problems. Setting this to a positive value invokes
        the Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, minimising:

            ``f(x) + c * |x|``,

        where ``f(x)`` is the objective function, and ``|x|`` the L1 norm of the
        variables. This parameter is the coefficient ``c``. As ``|x|`` is not
        differentiable at zero, the library modifies the user-provided objective
        function and gradient evaluations suitably; as such, the user merely has
        to return the (unregularised!) objective function value ``f(x)`` and its
        gradient ``g(x)``, as usual.

    orthantwise_w
        Weights for L1-regularisation with differential shrinkage. This is valid
        only for the OWL-QN method (i.e., ``orthantwise_c != 0``). Unless set by
        the user, OWL-QN minimises the objective function ``f(x)`` combined with
        the L1 norm ``|x|`` of the variables, ``f(x) + c * |x|``. If set,
        each variable is weighted by its corresponding weight, and the objective
        function now becomes:

            ``f(x) + c * |x|_w, |x|_w := |diag(w) x|``.

        The length of this vector must match the number of regularised variables,
        which may be less than the total, as indicated by ``orthantwise_start``
        and ``orthantwise_end``.

    orthantwise_start
        Start index for computing the L1 norm of the variables. It is only valid
        for the OWL-QN method (i.e., ``orthantwise_c != 0``). This parameter,
        denoted as ``b (0 <= b < N)``, specifies the index number from which the
        library computes the L1 norm of the variables ``x``:

            ``|x| := |x_b| + |x_{b+1}| + ... + |x_N|``.

        In other words, when it is set, variables ``x_0, ..., x_{b-1}`` (e.g., a
        bias term in logistic regression) are skipped when computing the L1 norm
        and thus stay protected from being regularised.

    orthantwise_end
        End index for computing the L1 norm of the variables. Only valid for the
        OWL-QN method (i.e., ``orthantwise_c != 0``). This parameter, denoted as
        ``e (0 < e <= N)``, specifies the index number at which the library
        stops computing the L1 norm of the variables ``x``.

    Returns
    -------
    dict
        A dictionary that holds the optimisation results; ``x`` is the solution,
        ``fun`` is the objective function value at the solution, ``success`` is
        a boolean flag indicating whether the optimiser exited successfully, and
        ``status`` and ``message`` describe the cause of termination.

    """
    # Check validity of the `linesearch` argument.
    if linesearch not in LINE_SEARCH_ALGO:
        raise LBFGSError(f"Invalid parameter `linesearch` specified. "
                         f"Valid options are: {', '.join(LINE_SEARCH_ALGO)}.")

    if orthantwise_c != 0:
        if linesearch not in ('backtracking', 'backtracking_armijo',):
            warnings.warn("Invalid parameter `linesearch` specified. "
                          "OWL-QN can only use the backtracking method with the Armijo condition, "
                          "specified as: backtracking or backtracking_armijo.")

        linesearch = 'backtracking'

    opt = LBFGS()
    opt.xtol = np.finfo(np.float64).eps  # set machine epsilon

    opt.m = m
    opt.epsilon = epsilon
    opt.past = past
    opt.delta = delta
    opt.max_iterations = max_iterations
    opt.linesearch = linesearch
    opt.max_linesearch = max_linesearch
    opt.min_step = min_step
    opt.max_step = max_step
    opt.ftol = ftol
    opt.wolfe = wolfe
    opt.gtol = gtol
    opt.cbfgs_epsilon = cbfgs_epsilon
    opt.cbfgs_alpha = cbfgs_alpha
    opt.orthantwise_c = orthantwise_c
    opt.orthantwise_w = orthantwise_w
    opt.orthantwise_start = orthantwise_start
    opt.orthantwise_end = orthantwise_end

    return opt.minimize(fun, x0, args=args, callback=callback)


__all__ = ['fmin_lbfgs', ]
