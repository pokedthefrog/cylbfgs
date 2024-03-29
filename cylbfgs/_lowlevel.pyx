"""
Cython wrapper around liblbfgs.
"""

import warnings
from typing import Optional, Callable

# noinspection PyUnresolvedReferences
cimport numpy as np
import numpy as np
import numpy.typing as npt

np.import_array()


ctypedef enum LineSearchAlgo:
    LBFGS_LINESEARCH_DEFAULT                   = 0,
    LBFGS_LINESEARCH_MORETHUENTE               = 0,
    LBFGS_LINESEARCH_BACKTRACKING_ARMIJO       = 1,
    LBFGS_LINESEARCH_BACKTRACKING              = 2,
    LBFGS_LINESEARCH_BACKTRACKING_WOLFE        = 2,
    LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3,


ctypedef enum ReturnCode:
    LBFGS_SUCCESS         = 0,
    LBFGS_CONVERGENCE     = 0,
    LBFGS_STOP,
    LBFGS_ALREADY_MINIMIZED,

    LBFGSERR_UNKNOWNERROR = -1024,
    LBFGSERR_LOGICERROR,
    LBFGSERR_OUTOFMEMORY,
    LBFGSERR_CANCELED,
    LBFGSERR_INVALID_N,
    LBFGSERR_INVALID_N_SSE,
    LBFGSERR_INVALID_X_SSE,
    LBFGSERR_INVALID_EPSILON,
    LBFGSERR_INVALID_TESTPERIOD,
    LBFGSERR_INVALID_DELTA,
    LBFGSERR_INVALID_LINESEARCH,
    LBFGSERR_INVALID_MINSTEP,
    LBFGSERR_INVALID_MAXSTEP,
    LBFGSERR_INVALID_FTOL,
    LBFGSERR_INVALID_WOLFE,
    LBFGSERR_INVALID_GTOL,
    LBFGSERR_INVALID_XTOL,
    LBFGSERR_INVALID_MAXLINESEARCH,
    LBFGSERR_INVALID_CBFGS_EPSILON
    LBFGSERR_INVALID_CBFGS_ALPHA,
    LBFGSERR_INVALID_ORTHANTWISE_C,
    LBFGSERR_INVALID_ORTHANTWISE_W,
    LBFGSERR_INVALID_ORTHANTWISE_START,
    LBFGSERR_INVALID_ORTHANTWISE_END,
    LBFGSERR_OUTOFINTERVAL,
    LBFGSERR_INCORRECT_TMINMAX,
    LBFGSERR_ROUNDING_ERROR,
    LBFGSERR_MINIMUMSTEP,
    LBFGSERR_MAXIMUMSTEP,
    LBFGSERR_MAXIMUMLINESEARCH,
    LBFGSERR_MAXIMUMITERATION,
    LBFGSERR_WIDTHTOOSMALL,
    LBFGSERR_INVALIDPARAMETERS,
    LBFGSERR_INCREASEGRADIENT,


cdef extern from "lbfgs.h":
    ctypedef double lbfgsfloatval_t

    # Callback interface to provide objective function and grad. evaluations.
    ctypedef lbfgsfloatval_t (*lbfgs_evaluate_t)(
            void *,                   # instance
            const lbfgsfloatval_t *,  # x
            lbfgsfloatval_t *,        # g
            int,                      # n
            lbfgsfloatval_t,          # step
    ) except -1

    # Callback interface to receive the progress of the optimisation process.
    ctypedef int (*lbfgs_progress_t)(
            void *,                   # instance
            const lbfgsfloatval_t *,  # x
            const lbfgsfloatval_t *,  # g
            lbfgsfloatval_t,          # fx
            lbfgsfloatval_t,          # xnorm
            lbfgsfloatval_t,          # gnorm
            lbfgsfloatval_t,          # step
            int,                      # n
            int,                      # k
            int,                      # ls
    ) except -1

    ctypedef struct lbfgs_parameter_t:
        int              m
        lbfgsfloatval_t  epsilon
        int              past
        lbfgsfloatval_t  delta
        int              max_iterations
        LineSearchAlgo   linesearch
        int              max_linesearch
        lbfgsfloatval_t  min_step
        lbfgsfloatval_t  max_step
        lbfgsfloatval_t  ftol
        lbfgsfloatval_t  wolfe
        lbfgsfloatval_t  gtol
        lbfgsfloatval_t  xtol
        lbfgsfloatval_t  cbfgs_epsilon
        lbfgsfloatval_t  cbfgs_alpha
        lbfgsfloatval_t  orthantwise_c
        lbfgsfloatval_t *orthantwise_w
        int              orthantwise_start
        int              orthantwise_end

    ReturnCode lbfgs(
            int,                  # n
            lbfgsfloatval_t *,    # x
            lbfgsfloatval_t *,    # ptr_fx
            lbfgs_evaluate_t,     # proc_evaluate
            lbfgs_progress_t,     # proc_progress
            void *,               # instance
            lbfgs_parameter_t *,  # param
    )

    void lbfgs_parameter_init(lbfgs_parameter_t *)
    lbfgsfloatval_t *lbfgs_malloc(int)
    void lbfgs_free(lbfgsfloatval_t *)


# Callback into the Python objective and grad. evaluation callable.
cdef lbfgsfloatval_t call_eval(
        void *cb_data,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        int n,
        lbfgsfloatval_t step,
) except -1:
    cdef object callback_data
    cdef np.npy_intp n_dim[1]

    callback_data = <object>cb_data
    (f, progress_fn, args) = callback_data

    n_dim[0] = <np.npy_intp>n
    x_array = np.PyArray_SimpleNewFromData(1, n_dim, np.NPY_DOUBLE, <void *>x)
    g_array = np.PyArray_SimpleNewFromData(1, n_dim, np.NPY_DOUBLE, <void *>g)

    return f(x_array, g_array, *args)


# Callback into the Python optimisation progress reporting callable.
cdef int call_progress(
        void *cb_data,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        lbfgsfloatval_t fx,
        lbfgsfloatval_t xnorm,
        lbfgsfloatval_t gnorm,
        lbfgsfloatval_t step,
        int n,
        int k,
        int ls,
) except -1:
    cdef object callback_data
    cdef np.npy_intp n_dim[1]

    callback_data = <object>cb_data
    (f, progress_fn, args) = callback_data

    if not progress_fn:
        return 0

    n_dim[0] = <np.npy_intp>n
    x_array = np.PyArray_SimpleNewFromData(1, n_dim, np.NPY_DOUBLE, <void *>x)
    g_array = np.PyArray_SimpleNewFromData(1, n_dim, np.NPY_DOUBLE, <void *>g)

    r = progress_fn(x_array, g_array, fx, xnorm, gnorm, step, k, ls, *args)
    return 0 if r is None \
        else r


# Copy an ndarray to a buffer allocated with lbfgs_malloc; needed to get the
# alignment right for SSE instructions.
cdef lbfgsfloatval_t *aligned_copy(x) except NULL:
    cdef int n
    cdef lbfgsfloatval_t *x_copy
    cdef Py_ssize_t i

    n = x.shape[0]
    x_copy = lbfgs_malloc(n)
    if x_copy is NULL:
        raise MemoryError

    for i in range(n):
        x_copy[i] = x[i]
    return x_copy


LINE_SEARCH_ALGO = {
    'default':                    LBFGS_LINESEARCH_DEFAULT,
    'more_thuente':               LBFGS_LINESEARCH_MORETHUENTE,
    'backtracking_armijo':        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
    'backtracking':               LBFGS_LINESEARCH_BACKTRACKING,
    'backtracking_wolfe':         LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
    'backtracking_strong_wolfe' : LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE,
}


_RETURN_MESSAGE = {
    'SUCCESS': {
        # Also handles `LBFGS_CONVERGENCE`.
        LBFGS_SUCCESS:              "Success: reached convergence (gtol).",
        LBFGS_STOP:                 "Success: met stopping criterion (ftol).",
        LBFGS_ALREADY_MINIMIZED:    "The initial variables already minimise the objective function.",
    },
    # Warn user but return results.
    'WARNING': {
        LBFGSERR_OUTOFINTERVAL:     "The line search step went out of the interval of uncertainty.",
        LBFGSERR_INCORRECT_TMINMAX: "A logic error occurred; "
                                    "alternatively, the interval of uncertainty became too small.",
        LBFGSERR_ROUNDING_ERROR:    "A rounding error occurred; "
                                    "alternatively, no line search step satisfies the sufficient "
                                    "decrease and curvature conditions.",
        LBFGSERR_MINIMUMSTEP:       "The line search step became smaller than `min_step`.",
        LBFGSERR_MAXIMUMSTEP:       "The line search step became larger than `max_step`.",
        LBFGSERR_MAXIMUMLINESEARCH: "The line search routine reached the maximum number of evaluations.",
        LBFGSERR_MAXIMUMITERATION:  "The algorithm routine reached the maximum number of iterations.",
        LBFGSERR_WIDTHTOOSMALL:     "The relative width of the interval of uncertainty became smaller "
                                    "than machine epsilon.",
        LBFGSERR_INVALIDPARAMETERS: "A logic error (negative line search step) occurred.",
        LBFGSERR_INCREASEGRADIENT:  "The current search direction increases the objective function value.",
    },
    # Unrecoverable; raise an error.
    'ERROR': {
        LBFGSERR_UNKNOWNERROR:              "Unknown error.",
        LBFGSERR_LOGICERROR:                "Logic error.",
        LBFGSERR_OUTOFMEMORY:               "Insufficient memory.",
        LBFGSERR_CANCELED:                  "The minimisation process has been cancelled.",
        LBFGSERR_INVALID_N:                 "Invalid number of variables specified.",
        LBFGSERR_INVALID_N_SSE:             "Invalid number of variables specified (for SSE).",
        LBFGSERR_INVALID_X_SSE:             "Array `x` must be aligned to 16 (for SSE).",
        LBFGSERR_INVALID_EPSILON:           "Invalid parameter `epsilon` specified.",
        LBFGSERR_INVALID_TESTPERIOD:        "Invalid parameter `past` specified.",
        LBFGSERR_INVALID_DELTA:             "Invalid parameter `delta` specified.",
        LBFGSERR_INVALID_LINESEARCH:        "Invalid parameter `linesearch` specified.",
        LBFGSERR_INVALID_MINSTEP:           "Invalid parameter `min_step` specified.",
        LBFGSERR_INVALID_MAXSTEP:           "Invalid parameter `max_step` specified.",
        LBFGSERR_INVALID_FTOL:              "Invalid parameter `ftol` specified.",
        LBFGSERR_INVALID_WOLFE:             "Invalid parameter `wolfe` specified.",
        LBFGSERR_INVALID_GTOL:              "Invalid parameter `gtol` specified.",
        LBFGSERR_INVALID_XTOL:              "Invalid parameter `xtol` specified.",
        LBFGSERR_INVALID_MAXLINESEARCH:     "Invalid parameter `max_linesearch` specified.",
        LBFGSERR_INVALID_CBFGS_EPSILON:     "Invalid parameter `cbfgs_epsilon` specified.",
        LBFGSERR_INVALID_CBFGS_ALPHA:       "Invalid parameter `cbfgs_alpha` specified.",
        LBFGSERR_INVALID_ORTHANTWISE_C:     "Invalid parameter `orthantwise_c` specified.",
        LBFGSERR_INVALID_ORTHANTWISE_W:     "Invalid parameter `orthantwise_w` specified.",
        LBFGSERR_INVALID_ORTHANTWISE_START: "Invalid parameter `orthantwise_start` specified.",
        LBFGSERR_INVALID_ORTHANTWISE_END:   "Invalid parameter `orthantwise_end` specified.",
    },
}


class LBFGSError(Exception):
    pass


cdef class LBFGS(object):
    """
    liblbfgs, wrapped in a class to permit parameter setting.
    """
    cdef lbfgs_parameter_t params

    def __init__(self):
        lbfgs_parameter_init(&self.params)

    @property
    def m(self):
        return self.params.m

    @m.setter
    def m(self, int value):
        self.params.m = value

    @property
    def epsilon(self):
        return self.params.epsilon

    @epsilon.setter
    def epsilon(self, double value):
        self.params.epsilon = value

    @property
    def past(self):
        return self.params.past

    @past.setter
    def past(self, int value):
        self.params.past = value

    @property
    def delta(self):
        return self.params.delta

    @delta.setter
    def delta(self, double value):
        self.params.delta = value

    @property
    def max_iterations(self):
        return self.params.max_iterations

    @max_iterations.setter
    def max_iterations(self, int value):
        self.params.max_iterations = value

    @property
    def linesearch(self):
        return LINE_SEARCH_ALGO[self.params.linesearch]

    @linesearch.setter
    def linesearch(self, str value):
        # Will raise `KeyError` for invalid values. This is intentional,
        # as the validity of the `linesearch` argument is supposed to be
        # checked before assignment.
        self.params.linesearch = LINE_SEARCH_ALGO[value]

    @property
    def max_linesearch(self):
        return self.params.max_linesearch

    @max_linesearch.setter
    def max_linesearch(self, int value):
        self.params.max_linesearch = value

    @property
    def min_step(self):
        return self.params.min_step

    @min_step.setter
    def min_step(self, double value):
        self.params.min_step = value

    @property
    def max_step(self):
        return self.params.max_step

    @max_step.setter
    def max_step(self, double value):
        self.params.max_step = value

    @property
    def ftol(self):
        return self.params.ftol

    @ftol.setter
    def ftol(self, double value):
        self.params.ftol = value

    @property
    def wolfe(self):
        return self.params.wolfe

    @wolfe.setter
    def wolfe(self, double value):
        self.params.wolfe = value

    @property
    def gtol(self):
        return self.params.gtol

    @gtol.setter
    def gtol(self, double value):
        self.params.gtol = value

    @property
    def xtol(self):
        return self.params.xtol

    @xtol.setter
    def xtol(self, double value):
        self.params.xtol = value

    @property
    def cbfgs_epsilon(self):
        return self.params.cbfgs_epsilon

    @cbfgs_epsilon.setter
    def cbfgs_epsilon(self, double value):
        self.params.cbfgs_epsilon = value

    @property
    def cbfgs_alpha(self):
        return self.params.cbfgs_alpha

    @cbfgs_alpha.setter
    def cbfgs_alpha(self, double value):
        self.params.cbfgs_alpha = value

    @property
    def orthantwise_c(self):
        return self.params.orthantwise_c

    @orthantwise_c.setter
    def orthantwise_c(self, double value):
        self.params.orthantwise_c = value

    @property
    def orthantwise_w(self):
        if self.params.orthantwise_w == NULL:
            return None

        cdef np.npy_intp n_dim[1]
        n_dim[0] = <np.npy_intp>(self.orthantwise_end - self.orthantwise_start)

        return np.PyArray_SimpleNewFromData(1, n_dim, np.NPY_DOUBLE,
                                            <void *> self.params.orthantwise_w)

    @orthantwise_w.setter
    def orthantwise_w(self, np.ndarray[np.float64_t, ndim=1] vector):
        cdef lbfgsfloatval_t *w

        if vector is None:
            w = NULL
        else:
            w = &vector[0]

        self.params.orthantwise_w = w

    @property
    def orthantwise_start(self):
        return self.params.orthantwise_start

    @orthantwise_start.setter
    def orthantwise_start(self, int value):
        self.params.orthantwise_start = value

    @property
    def orthantwise_end(self):
        return self.params.orthantwise_end

    @orthantwise_end.setter
    def orthantwise_end(self, int value):
        self.params.orthantwise_end = value


    def minimize(
            self,
            fun: Callable,
            x0: npt.NDArray[np.float64],
            args: tuple = (),
            callback: Optional[Callable] = None,
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

        Returns
        -------
        dict
            A dictionary that holds the optimisation results; ``x`` is the solution,
            ``fun`` is the objective function value at the solution, ``success`` is
            a boolean flag indicating whether the optimiser exited successfully, and
            ``status`` and ``message`` describe the cause of termination.

        """
        cdef int res
        cdef int n
        cdef np.npy_intp n_dim[1]

        cdef lbfgsfloatval_t *x
        cdef lbfgsfloatval_t fx

        # Check validity of input arguments.
        if not callable(fun):
            raise TypeError(f"`fun` must be callable, got {type(fun)}.")

        if (callback is not None
                and not callable(callback)):
            raise TypeError(f"`callback` must be callable, got {type(callback)}.")

        x0 = np.atleast_1d(np.asarray(x0))
        if x0.ndim != 1:
            raise ValueError("`x0` must only have one dimension.")

        n = x0.size
        n_dim[0] = <np.npy_intp>n
        if n != n_dim[0]:
            raise LBFGSError(f"Array of {n} elements (`x0`) too large to handle.")

        x = aligned_copy(x0)

        try:
            cb_data = (fun, callback, args)
            ret = lbfgs(n, x, &fx, call_eval, call_progress, <void *>cb_data, &self.params)

            # Raise an error?
            if ret in _RETURN_MESSAGE['ERROR']:
                if ret == LBFGSERR_OUTOFMEMORY:
                    raise MemoryError
                else:
                    raise LBFGSError(_RETURN_MESSAGE['ERROR'][ret])

            # Unknown error.
            elif ret not in _RETURN_MESSAGE['WARNING'] \
                    and ret not in _RETURN_MESSAGE['SUCCESS']:
                raise LBFGSError(ret)

            # Either return successfully or with warning.
            message = None

            success = True
            if ret in _RETURN_MESSAGE['WARNING']:
                success = False

                message = _RETURN_MESSAGE['WARNING'][ret]
                warnings.warn(message)
            else:
                message = _RETURN_MESSAGE['SUCCESS'][ret]

            x_res = np.PyArray_SimpleNewFromData(1, n_dim, np.NPY_DOUBLE, <void *>x).copy()
            return {
                'x':       x_res,
                'success': success,
                'status':  ret,
                'message': message,
                'fun':     fx,
            }

        finally:
            lbfgs_free(x)
