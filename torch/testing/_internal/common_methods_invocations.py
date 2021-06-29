from functools import wraps, partial
from itertools import product, chain
import itertools
import collections
import copy
import operator
import random
import numbers

import torch
import numpy as np
from torch._six import inf
import collections.abc

from typing import List, Sequence, Tuple, Union

from torch.testing import \
    (make_non_contiguous, floating_types, floating_types_and, complex_types,
     floating_and_complex_types, floating_and_complex_types_and,
     all_types_and_complex_and, all_types_and, all_types_and_complex,
     integral_types_and, all_types)
from .._core import _dispatch_dtypes
from torch.testing._internal.common_device_type import \
    (skipIf, skipCUDAIfNoMagma, skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfNoCusolver,
     skipCPUIfNoLapack, skipCPUIfNoMkl, skipCUDAIfRocm, precisionOverride, toleranceOverride, tol)
from torch.testing._internal.common_cuda import CUDA11OrLater, SM53OrLater
from torch.testing._internal.common_utils import \
    (is_iterable_of_tensors,
     random_symmetric_matrix, random_symmetric_psd_matrix,
     make_fullrank_matrices_with_distinct_singular_values,
     random_symmetric_pd_matrix, make_symmetric_matrices,
     make_symmetric_pd_matrices,
     random_fullrank_matrix_distinct_singular_value,
     TEST_WITH_ROCM, IS_WINDOWS, IS_MACOS, make_tensor, TEST_SCIPY,
     torch_to_numpy_dtype_dict, slowTest, TEST_WITH_ASAN,
     GRADCHECK_NONDET_TOL,)
import torch.testing._internal.opinfo_helper as opinfo_helper

from setuptools import distutils

if TEST_SCIPY:
    import scipy.special


class DecorateInfo(object):
    """Describes which test, or type of tests, should be wrapped in the given
       decorators when testing an operator. Any test that matches all provided
       arguments will be decorated. The decorators will only be applied if the
       active_if argument is True."""

    __slots__ = ['decorators', 'cls_name', 'test_name', 'device_type', 'dtypes', 'active_if']

    def __init__(self, decorators, cls_name=None, test_name=None, *,
                 device_type=None, dtypes=None, active_if=True):
        self.decorators = list(decorators) if isinstance(decorators, collections.abc.Sequence) else [decorators]
        self.cls_name = cls_name
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if

    def is_active(self, cls_name, test_name, device_type, dtype):
        return (
            self.active_if and
            (self.cls_name is None or self.cls_name == cls_name) and
            (self.test_name is None or self.test_name == test_name) and
            (self.device_type is None or self.device_type == device_type) and
            (self.dtypes is None or dtype in self.dtypes)
        )


class SkipInfo(DecorateInfo):
    """Describes which test, or type of tests, should be skipped when testing
       an operator. Any test that matches all provided arguments will be skipped.
       The skip will only be checked if the active_if argument is True."""

    def __init__(self, cls_name=None, test_name=None, *,
                 device_type=None, dtypes=None, active_if=True):
        super().__init__(decorators=skipIf(True, "Skipped!"), cls_name=cls_name,
                         test_name=test_name, device_type=device_type, dtypes=dtypes,
                         active_if=active_if)

class SampleInput(object):
    """Represents sample inputs to a function."""

    __slots__ = ['input', 'args', 'kwargs', 'output_process_fn_grad', 'broadcasts_input', 'name']

    def __init__(self, input, *, args=tuple(), kwargs=None, output_process_fn_grad=lambda x: x, broadcasts_input=False, name=""):
        # input is the first input to the op and must be either a Tensor or TensorList (Sequence[Tensor]).
        # This follows the typical pattern where for Tensor inputs op(t, ...) = t.op(...).
        # op with TensorList inputs do not support method or inplace variants.
        assert isinstance(input, torch.Tensor) or is_iterable_of_tensors(input)
        self.input: Union[torch.Tensor, Sequence[torch.Tensor]] = input
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.output_process_fn_grad = output_process_fn_grad
        self.name = name

        # Specifies if `self.input` is broadcasted or not,
        # given that the operator supports broadcasting.
        # This field is used to verify the behavior for inplace variant.
        #
        # If a SampleInput is marked with `broadcasts_input=True`,
        # it is verified that we get a `RuntimerError` with this sample,
        # and inplace variant. Also inplace grad{grad} tests are skipped,
        # for such inputs (as they will error out otherwise).
        self.broadcasts_input = broadcasts_input

    def _repr_helper(self, formatter):
        # Helper function to return the details of the SampleInput as `str`
        # It consolidates all the fields of SampleInput and allows,
        # formatting the fields like `input`, `args`, etc with `formatter`
        # callable to customize the representation.
        # Look at `summary` method for example.
        arguments = [
            f'input={formatter(self.input)}',
            f'args={formatter(self.args)}',
            f'kwargs={formatter(self.kwargs)}',
            f'output_process_fn_grad={self.output_process_fn_grad}',
            f'broadcasts_input={self.broadcasts_input}',
            f'name={repr(self.name)}']

        return f'SampleInput({", ".join(a for a in arguments if a is not None)})'

    def __repr__(self):
        return self._repr_helper(lambda x: x)

    def summary(self):
        # Returns the SampleInput details in a more
        # friendly format.
        # It formats `Tensor` and `TensorList`
        # in a more condensed representation.
        def formatter(arg):
            # Format any instance of `Tensor` (standalone, in list, or in dict)
            # by Tensor[TensorShape]
            # Eg. Tensor with shape (3, 4) is formatted as Tensor[3, 4]
            if isinstance(arg, torch.Tensor):
                shape = str(tuple(arg.shape)).replace('(', '').replace(')', '')
                return f"Tensor[{shape}]"
            elif isinstance(arg, dict):
                return {k: formatter(v) for k, v in arg.items()}
            elif is_iterable_of_tensors(arg):
                return "TensorList[" + ", ".join(map(formatter, arg)) + "]"
            elif isinstance(arg, (list, tuple)):  # Handle list, tuple
                return "(" + ",".join(map(formatter, arg)) + ")"

            return repr(arg)

        return self._repr_helper(formatter)

    # Returns the NumPy version of the sample input object in the form of a tuple: (input, args, kwargs)
    def numpy(self):
        # Converts tensors to ndarrays by calling .detach().cpu().numpy() on them
        # Numbers, strings, and bool are preserved as is
        # Lists, tuples and dicts are handled by calling this function recursively
        def to_numpy(x):
            def _np(t):
                return t.detach().cpu().numpy()

            if isinstance(x, torch.Tensor):
                return _np(x)
            elif isinstance(x, list):
                return list(map(to_numpy, x))
            elif isinstance(x, tuple):
                return tuple(map(to_numpy, x))
            elif isinstance(x, dict):
                return {k: to_numpy(v) for k, v in x.items()}
            elif isinstance(x, (numbers.Number, bool, str)):
                return x

            raise ValueError("Unknown type {0}!".format(type(x)))

        sample_np_input, np_args, np_kwargs = to_numpy(self.input), to_numpy(self.args), to_numpy(self.kwargs)
        return (sample_np_input, np_args, np_kwargs)

class AliasInfo(object):
    """Class holds alias information. For example, torch.abs ->
    torch.absolute, torch.Tensor.absolute, torch.Tensor.absolute_
    """

    def __init__(self, alias_name):
        self.name = alias_name
        self.op = _getattr_qual(torch, alias_name)
        self.method_variant = getattr(torch.Tensor, alias_name, None)
        self.inplace_variant = getattr(torch.Tensor, alias_name + "_", None)

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)


_NOTHING = object()  # Unique value to distinguish default from anything else


# Extension of getattr to support qualified names
# e.g. _getattr_qual(torch, 'linalg.norm') -> torch.linalg.norm
def _getattr_qual(obj, name, default=_NOTHING):
    try:
        for path in name.split('.'):
            obj = getattr(obj, path)
        return obj
    except AttributeError:
        if default is not _NOTHING:
            return default
        else:
            raise

# Note [OpInfos]
# ~~~~~~~~~~~~~~
#
# This note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# See also: the OpInfo tracker (https://github.com/pytorch/pytorch/issues/54261)
# See also: "Writing Test Templates" in common_device_type.py to learn how to
#   parametrize a test template using OpInfos.
#
# An OpInfo is a collection of metadata related to a PyTorch operator. This
#   metadata is used to generate tests that validate properties of the operator,
#   like if it implements the correct gradient formula.
#
# WHY OPINFOS?
# ~~~~~~~~~~~~
#
# OpInfos are principally intended to do two things:
#
#   1) to simplify testing an operator
#   2) to allow systems (like autograd, torchscript, fx, nnc...) to test
#        against every PyTorch operator
#
# Both these goals are still a work in progress. Not every operator has an
#   OpInfo, and some operator tests still have to be written manually.
#
# The utility of OpInfos can also be motivated from a different perspective.
#   PyTorch is a complicated framework with many interrelated systems, too
#   many for any one person to keep track of. An OpInfo can be thought of as the
#   interface between an operator implementer and those other systems. Instead of
#   requiring the implementer of torch.foo understand how to test its forward
#   mode AD or NNC support that's typically handled automatically just by
#   defining an OpInfo. This is a helpful perspective to have, because it's often
#   surprising to OpInfo writers that just implementing an OpInfo typically can't
#   verify an operator is actually implemented correctly. "If an OpInfo doesn't
#   validate my op works as expected, what's the point of it?" But the point of
#   it is that it lets engineers focus on testing their operator logic instead
#   of having to write tests for how the operator interacts with each of
#   PyTorch's many systems. And, OK, sometimes it validates your op works
#   the way you want and all you have to do is write an OpInfo and you're done
#   testing... more on that below.
#
# WHAT'S AN OPINFO?
# ~~~~~~~~~~~~~~~~~
#
# So what is an OpInfo? It's a Python class that describes an operator's properties,
#   like which dtypes it supports on the CPU and whether it has any aliases.
#   These properties can be divided into three categories:
#
#   1) Metadata describing the operator, like the operator's name and if it
#     "supports" the out kwarg.
#   2) Test directives, like "skips" that tell the test suite to skip some
#     tests.
#   3) A "sample inputs" function that generates valid inputs for the operator.
#
# OpInfo attributes are described in more detail below.
#
# THE SAMPLE INPUTS FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The "sample inputs" function merits special elaboration. This function is
#   crucial to testing with OpInfos. A typical OpInfo test has to treat the operator
#   as a black box. There's no structure for the test to understand or exploit.
#   Without "sample inputs" it wouldn't even know how to call the OpInfo's
#   operator. The sample input function saves the day by providing different
#   "SampleInputs" that can be used to call the operator. A sample input
#   function should have the following signature:
#
#   def sample_inputs_foo(op_info, device, dtype, requires_grad, **kwargs):
#
#   And should return a list of SampleInputs (see the class description above).
#   Each SampleInput defines an "input", "args", "kwargs",
#   an "output_process_fn_grad" function, the "broadcasts_input" bool and
#   a "name".
#
# The "input" is the first argument to the operator, or the tensor that
#   the method or inplace variants of the operator should be called on, and
#   should be on the requested device, of the requested dtype, and its
#   requires_grad attribute should be set to the requires_grad argument.
#
# "args" should contain positional arguments, and "kwargs" keyword arguments.
#
# "output_process_fn_grad" has an interesting name. It's a function that maps
#   the operator's output (when given the input, args, and kwargs) to the
#   portion of the output to gradcheck. For example, consider an operator
#   like torch.linalg.slogdet
#   (https://pytorch.org/docs/master/generated/torch.linalg.slogdet.html).
#   This operator returns a tuple of two tensors, but the first tensor
#   cannot be backwarded through. Its "output_process_fn_grad" filters
#   this output tuple to just the second argument, which we can call backward
#   on. Functions that produce a single tensor can ignore this argument.
#
# "broadcasts_input" is a bool indicated if the SampleInput causes the operator
#   to broadcast the "input" argument. This is important for tests to understand
#   because inplace variants of operations throw a runtime error if they
#   would broadcast their input arguments, so tests that work with inplace
#   variants filter SampleInputs that broadcast their input.
#
# "name" is a string that's just used for debugging. It appears when printing
#   the SampleInput.
#
# OPINFO FILE ORGANIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# All OpInfos are currently defined in this file. Most OpInfo tests are defined
#   in test_ops.py, but some system-specific tests are defined in those
#   systems' test files, and subclass-specific tests are defined in the test
#   file that corresponds to that subclass (see the below).
#   Expect a reorganization in the future.
#
# WHAT'S TESTED?
# ~~~~~~~~~~~~~~
#
# Every OpInfo in the op_db sequence has the following properties validated in
# test_ops.py:
#
#   - that its supported dtypes are specified correctly
#   - that it supports the out= argument properly (if it allows out=),
#       see https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
#   - that it works with the conjugate view bit properly
#   - that its function, method, and inplace variants perform the same operation
#       (that is, that torch.add, torch.Tensor.add, and torch.Tensor.add_ all
#       do the same thing).
#   - that its inplace variant preserves the input's storage
#   - that its gradient formula is implemented correctly, and that it supports
#       gradgrad and complex grad and gradgrad and forward mode AD properly for
#       the op's function and inplace variants (method variants are skipped
#       to reduce test time).
#   - that the operation performs the same operation when traced or scripted
#       using the jit
#   - that the operation is autodifferentiated by the jit as expected
#   - that the operator's aliases, if any, perform the same operation and that
#       the jit understands the alias
#
# Additional OpInfo tests are in test_jit_fuser_te.py, test_fx_experimental.py,
#   and test_fx.py. These tests validate that operators work with NNC and FX
#   as expected.
#
# For performance, some of the above tests may only run on the first
#   SampleInput returned by an OpInfo's sample input function.
#
# In addition to these tests, some subclasses (discussed in the next section)
#   define additional tests.
#
# Critically, as mentioned above, what's not tested is that the operator
#   works as expected. When implementing an OpInfo an engineer must still
#   typically write one or more tests validating the operator's behavior.
#
# OPINFO (SUB)CLASSES
# ~~~~~~~~~~~~~~~~~~~
#
# In addition to the OpInfo base class there are several specialized OpInfo
#   subclasses. For example, the UnaryUfuncInfo subclass is used for
#   unary elementwise operations. These operations have a common structure
#   that test_unary_ufuncs.py exploits with additional automated testing.
#   The automated testing in test_unary_ufuncs.py is so thorough, comparing
#   the operator to a NumPy reference function on a plethora of values, that
#   just implementing an OpInfo for a unary elementwise operation is often
#   sufficient testing.
#
# The ForeachFuncInfo is another OpInfo subclass that is hyper-specialized to a
#   very unique class of operations. These OpInfos aren't included in the
#   op_db sequence and have their own tests.
#
# Other OpInfo subclasses, like SpectralFuncInfo, are just for convenience
# when writing OpInfos.
#
# TESTING A NEW OPERATOR
# ~~~~~~~~~~~~~~~~~~~~~~
#
# If you're adding a new operator to the torch, torch.fft, torch.linalg,
#   or torch.special namespaces then you should add an OpInfo for it. As
#   mentioned a couple times above, implementing an OpInfo is not usually
#   sufficient testing (unless the operator is a unary elementwise operator).
#   The OpInfo will only test the properties described in the "WHAT'S TESTED"
#   section. It DOES NOT verify that the operator is implemented correctly.
#
# We are currently reviewing if operators in the torch.nn.functional namespace
#   will be added as OpInfos, but you are encouraged to add an OpInfo for
#   such operators, too.
#
# TIPS FOR WRITING AN OPINFO AND OPINFO TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Writing an OpInfo can be a little daunting. Since the point of an OpInfo is to
#   be consumed by a variety of systems it can be hard to understand how to
#   deal with test failures or how to set the OpInfo metadata properly.
#
# Before adding an OpInfo it helps to look at other OpInfos. A sample inputs
#   function must be defined, and the operator's dtypes must be specified.
#   Once that's done you should run the operator's tests in test_ops.py
#   (these can be filtered using the "-k" argument in pytest). Tests that
#   fail should provide an error message that describes what to change about
#   your OpInfo. You don't need to worry about changing an OpInfo's default
#   values unless a test yells at you.
#
# Similarly, if you're writing a test that consumes OpInfos then it's critical
#   your test provides a clear error message describing what to do when it
#   fails. You should not assume the OpInfo implementer is familiar with your
#   system.
#
# If you see a confusing error message while developing an OpInfo then please
#   file an issue describing what happened.
#
# This trial-and-error approach can be frustrating to writing an OpInfo can
#   be frustrating, but it's probably necessary as long as OpInfos don't require
#   learning about all the systems that consume them. One thing that can help
#   is the get_supported_dtypes() function defined in opinfo_helper.py. This
#   function can be used to programmatically specify the dtypes an operator
#   supports, and is especially useful if writing an OpInfo on a machine
#   without a CUDA device. See its documentation for more details.
#
# THE FUTURE OF OPINFOS AND OPINFO TESTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the future we expect OpInfo coverage to improve, particularly for the
#   torch, torch.fft, torch.linalg, and torch.special namespaces, and possibly
#   for the torch.nn.functional namespace, too. In addition an analogous class,
#   ModuleInfo, will be developed to improve module testing.
#
# We also expect at least two new OpInfo subclasses: BinaryUfuncInfo and
#   ReductionInfo. Both will have new automated tests for correctness, too,
#   which might make testing binary elementwise operations and reductions as
#   simple as testing unary elementwise operations today.

# Classes and methods for the operator database
class OpInfo(object):
    """Operator information and helper functions for acquiring it."""

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref=None,  # An optional reference function that accepts ndarrays (AKA "NumPy arrays").
                            # If given, the op will be compared with its reference on each of its sample inputs.
                 # the following metadata describes the operator, its variants,
                 #   and its aliases, if any
                 aliases=None,  # iterable of aliases, e.g. ("absolute",) for torch.abs
                 variant_test_name='',  # additional string to include in the test name
                                        # this is useful when an op needs multiple OpInfos,
                                        # like divide does, often because it's really several
                                        # different ops behind the scenes
                 op=None,  # the function variant of the operation, populated as torch.<name> if None
                 method_variant=_NOTHING,  # explicitly specifies the method variant of the operator
                                           # if _NOTHING (default), the method variant will be autopopulated
                                           # if None, then the OpInfo specifies no method variant
                 inplace_variant=_NOTHING,  # explicitly specifies the inplace variant of the operator
                                            # if _NOTHING (default), the method variant will be autopopulated
                                            # if None, then the OpInfo specifies no method variant

                 # the following metadata are test directives for skipping or
                 # modifying tests and a pointer to the op's sample inputs function
                 # this function lets the OpInfo generate valid inputs
                 skips=tuple(),  # information about which tests to skip
                 decorators=None,  # decorators to apply to generated tests
                 sample_inputs_func=None,  # function to generate sample inputs

                 # the following metadata relates to dtype support and is tested for correctness in test_ops.py
                 dtypes=floating_types(),  # dtypes this function is expected to work with
                 # the following dtypesIf... options override the dtypes value
                 # on their respective device types
                 dtypesIfCPU=None,  # dtypes this function is expected to work with on CPU
                 dtypesIfCUDA=None,  # dtypes this function is expected to work with on CUDA
                 dtypesIfROCM=None,  # dtypes this function is expected to work with on ROCM
                 backward_dtypes=None,  # backward dtypes this function is expected to work with
                 backward_dtypesIfCPU=None,  # backward dtypes this function is expected to work with on CPU
                 backward_dtypesIfCUDA=None,  # backward dtypes this function is expected to work with on CUDA
                 backward_dtypesIfROCM=None,  # backward dtypes this function is expected to work with on ROCM
                 default_test_dtypes=None,  # dtypes to test with by default. Tests are instantiated with
                                            # these dtypes for the op unless otherwise specified.
                                            # This is helpful in reducing the test matrix.
                 # the following metadata describes the operators out= support
                 supports_out=True,  # whether the op supports the out kwarg
                                     # defaults to True, if the op does not allow the out kwarg or
                                     # supports it incorrectly then test_out in test_ops.py should fail
                 safe_casts_outputs=False,  # whether op allows safe casting when writing to out arguments

                 # the following metadata relates to autograd support
                 supports_autograd=True,  # whether the operation supports gradient computations
                                          # if true, gradient correctness is tested in test_ops.py
                                          # using the op's sample inputs
                 supports_gradgrad=True,  # whether the op supports second order gradients
                                          # if true, gradgrad correctness is tested in test_ops.py
                                          # (this value is ignored if supports_autograd=False)
                 supports_inplace_autograd=None,  # whether the operation supports inplace autograd
                                                  # if true, tested in test_ops.py
                                                  # defaults to supports_autograd's value
                 supports_forward_ad=False,  # Whether the operation support forward mode AD
                                             # If the value is True, we check that the gradients are correct
                                             # If the value is False, we test that forward grad is not implemented
                 gradcheck_wrapper=lambda op, *args, **kwargs: op(*args, **kwargs),  # wrapper function for gradcheck
                 check_batched_grad=True,  # whether to check batched grad when doing gradcheck
                 check_batched_gradgrad=True,  # whether to check batched grad grad when doing gradgradcheck
                 gradcheck_nondet_tol=0.0,  # tolerance for nondeterminism while performing gradcheck
                 gradcheck_fast_mode=None,  # Whether to use the fast implmentation for gradcheck/gradgradcheck.
                                            # When set to None, defers to the default value provided by the wrapper
                                            # function around gradcheck (testing._internal.common_utils.gradcheck)

                 # the following metadata relates to JIT support and is tested for correctness in test_ops.py
                 aten_name=None,  # name of the corresponding aten:: operator
                 assert_autodiffed=False,  # if a op's aten::node is expected to be symbolically autodiffed
                 autodiff_nonfusible_nodes=None,  # a list of strings with node names that are expected to be in a
                                                  # DifferentiableGraph when autodiffed. Ex: ['aten::add', 'aten::mm'],
                                                  # default is populated to be ['aten::(name of Python operator)']
                 autodiff_fusible_nodes=None,  # a list of strings with node names that are expected to be in FusionGroups
                                               # inside of DifferentiableGraphs when this operation is autodiffed.
                                               # Ex: ['aten::add', 'aten::mm'], defaults to an empty list
                                               # Note: currently no ops use fusible nodes

                 # the following metadata relates to sparse support and is used in test_sparse.py
                 supports_sparse=False,  # whether the op supports sparse inputs

                 # the following metadata relates to complex support and is checked in test_ops.py
                 test_conjugated_samples=True,
                 ):

        dtypes_args = (dtypes, dtypesIfCPU, dtypesIfCUDA, dtypesIfROCM)
        # Validates the dtypes are generated from the dispatch-related functions
        for dtype_list in dtypes_args:
            assert isinstance(dtype_list, (_dispatch_dtypes, type(None)))

        self.name = name
        self.ref = ref
        self.aten_name = aten_name if aten_name is not None else name
        self.variant_test_name = variant_test_name

        # Attribute to verify dynamic_dtypes are used.
        self.dynamic_dtypes = any(map(lambda dtypes: isinstance(
            dtypes, opinfo_helper._dynamic_dispatch_dtypes), dtypes_args))

        if self.dynamic_dtypes:
            # Make sure `dtyesIfCUDA` is dynamic, if dynamic dispatch is used for CPU
            # This is because, below we set dtypesIfCUDA to dtypes if they are None.
            assert isinstance(dtypesIfCUDA, opinfo_helper._dynamic_dispatch_dtypes), \
                (f"To use dynamic dypes for operator {name}, "
                 "acquire the dtypes dynamically for argument `dtypesIfCUDA`."
                 "This is to ensure that CUDA dtypes are acquired correctly as they"
                 "differ from CPU dtypes occasionally")

        self.dtypes = set(dtypes)

        # NOTE: backward dtypes must be acquired before forward dtypes
        #   since they fallback to explicit (not implicit!) specifications of
        #   forward dtypes
        self.backward_dtypes = set(backward_dtypes) if backward_dtypes is not None else self.dtypes
        self.backward_dtypesIfCPU = set(backward_dtypesIfCPU) if backward_dtypesIfCPU is not None else (
            backward_dtypes if backward_dtypes is not None
            else dtypesIfCPU if dtypesIfCPU is not None
            else dtypes)
        self.backward_dtypesIfCUDA = set(backward_dtypesIfCUDA) if backward_dtypesIfCUDA is not None else (
            backward_dtypes if backward_dtypes is not None
            else dtypesIfCUDA if dtypesIfCUDA is not None
            else dtypes)
        self.backward_dtypesIfROCM = set(backward_dtypesIfROCM) if backward_dtypesIfROCM is not None else (
            backward_dtypesIfCUDA if backward_dtypesIfCUDA is not None
            else backward_dtypes if backward_dtypes is not None
            else dtypesIfROCM if dtypesIfROCM is not None
            else dtypesIfCUDA if dtypesIfCUDA is not None
            else dtypes)

        self.dtypesIfCPU = set(dtypesIfCPU) if dtypesIfCPU is not None else self.dtypes
        self.dtypesIfCUDA = set(dtypesIfCUDA) if dtypesIfCUDA is not None else self.dtypes
        self.dtypesIfROCM = set(dtypesIfROCM) if dtypesIfROCM is not None else self.dtypesIfCUDA

        self._default_test_dtypes = set(default_test_dtypes) if default_test_dtypes is not None else None

        # NOTE: if the op is unspecified it is assumed to be under the torch namespace
        self.op = op if op else _getattr_qual(torch, self.name)
        method_variant = getattr(torch.Tensor, name, None) if method_variant is _NOTHING else method_variant
        # attributes like real, imag are not callable
        self.method_variant = method_variant if callable(method_variant) else None
        inplace_name = name + "_"
        self.inplace_variant = getattr(torch.Tensor, inplace_name, None) \
            if inplace_variant is _NOTHING else inplace_variant
        self.operator_variant = getattr(operator, name, None)

        self.supports_out = supports_out
        self.safe_casts_outputs = safe_casts_outputs

        self.skips = skips
        self.decorators = decorators
        self.sample_inputs_func = sample_inputs_func

        self.assert_autodiffed = assert_autodiffed
        self.autodiff_fusible_nodes = autodiff_fusible_nodes if autodiff_fusible_nodes else []
        if autodiff_nonfusible_nodes is None:
            self.autodiff_nonfusible_nodes = ['aten::' + self.name]
        else:
            self.autodiff_nonfusible_nodes = autodiff_nonfusible_nodes

        # autograd support
        self.supports_autograd = supports_autograd
        self.supports_inplace_autograd = supports_inplace_autograd
        if self.supports_inplace_autograd is None:
            self.supports_inplace_autograd = supports_autograd

        self.gradcheck_wrapper = gradcheck_wrapper
        self.supports_gradgrad = supports_gradgrad
        self.supports_forward_ad = supports_forward_ad
        self.check_batched_grad = check_batched_grad
        self.check_batched_gradgrad = check_batched_gradgrad
        self.gradcheck_nondet_tol = gradcheck_nondet_tol
        self.gradcheck_fast_mode = gradcheck_fast_mode

        self.supports_sparse = supports_sparse

        self.aliases = ()
        if aliases is not None:
            self.aliases = tuple(AliasInfo(a) for a in aliases)  # type: ignore[assignment]

        self.test_conjugated_samples = test_conjugated_samples

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    def get_op(self):
        """Returns the function variant of the operator, torch.<op_name>."""
        return self.op

    def get_method(self):
        """Returns the method variant of the operator, torch.Tensor.<op_name>.
        Returns None if the operator has no method variant.
        """
        return self.method_variant

    def get_inplace(self):
        """Returns the inplace variant of the operator, torch.Tensor.<op_name>_.
        Returns None if the operator has no inplace variant.
        """
        return self.inplace_variant

    def get_operator_variant(self):
        """Returns operator variant of the operator, e.g. operator.neg
        Returns None if the operator has no operator variant.
        """
        return self.operator_variant

    def conjugate_sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs but with the tensor input or first
        tensor in a sequence input conjugated.
        """

        # TODO: Remove the try/except once all operators have sample_inputs_func with
        #       **kwargs in their signature.
        try:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
        except TypeError:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad)

        conj_samples = list(samples)

        def conjugate(tensor):
            _requires_grad = tensor.requires_grad
            with torch.no_grad():
                tensor = tensor.conj()
            return tensor.requires_grad_(_requires_grad)

        for i in range(len(samples)):
            sample = conj_samples[i]
            # Note: it is assumed that the input here is either a tensor or tensorlist
            if isinstance(sample.input, torch.Tensor):
                sample.input = conjugate(sample.input)
            else:
                with torch.no_grad():
                    sample.input[0] = conjugate(sample.input[0])

        return tuple(conj_samples)

    def sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs.

        These samples should be sufficient to test the function works correctly
        with autograd, TorchScript, etc.
        """

        # TODO: Remove the try/except once all operators have sample_inputs_func with
        #       **kwargs in their signature.
        try:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
        except TypeError:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad)

        if 'include_conjugated_inputs' in kwargs and kwargs.get('include_conjugated_inputs'):
            conj_samples = self.conjugate_sample_inputs(device, dtype, requires_grad, **kwargs)
            samples_list = list(samples)
            samples_list.extend(conj_samples)
            samples = tuple(samples_list)

        return samples

    # Returns True if the test should be skipped and False otherwise
    def should_skip(self, cls_name, test_name, device_type, dtype):
        return any(si.is_active(cls_name, test_name, device_type, dtype)
                   for si in self.skips)

    def supported_dtypes(self, device_type):
        if device_type == 'cpu':
            return self.dtypesIfCPU
        if device_type == 'cuda':
            return self.dtypesIfROCM if TEST_WITH_ROCM else self.dtypesIfCUDA
        else:
            return self.dtypes

    def supported_backward_dtypes(self, device_type):
        if not self.supports_autograd:
            return set()

        backward_dtypes = None
        if device_type == 'cpu':
            backward_dtypes = self.backward_dtypesIfCPU
        elif device_type == 'cuda':
            backward_dtypes = self.backward_dtypesIfROCM if TEST_WITH_ROCM else self.backward_dtypesIfCUDA
        else:
            backward_dtypes = self.backward_dtypes

        allowed_backward_dtypes = floating_and_complex_types_and(torch.bfloat16, torch.float16)
        return set(allowed_backward_dtypes).intersection(backward_dtypes)

    def supports_complex_autograd(self, device_type):
        if device_type == 'cpu':
            return any(dtype.is_complex for dtype in self.backward_dtypesIfCPU)
        if device_type == 'cuda':
            if TEST_WITH_ROCM:
                return any(dtype.is_complex for dtype in self.backward_dtypesIfROCM)
            else:
                return any(dtype.is_complex for dtype in self.backward_dtypesIfCUDA)
        else:
            return any(dtype.is_complex for dtype in self.backward_dtypes)

    def supports_dtype(self, dtype, device_type):
        return dtype in self.supported_dtypes(device_type)

    def default_test_dtypes(self, device_type):
        """Returns the default dtypes used to test this operator on the device.

        Equal to the operator's default_test_dtypes filtered to remove dtypes
        not supported by the device.
        """
        supported = self.supported_dtypes(device_type)
        return (supported if self._default_test_dtypes is None
                else supported.intersection(self._default_test_dtypes))


L = 20
M = 10
S = 5


def sample_inputs_unary(op_info, device, dtype, requires_grad, **kwargs):
    low, high = op_info.domain
    low = low if low is None else low + op_info._domain_eps
    high = high if high is None else high - op_info._domain_eps

    return (SampleInput(make_tensor((L,), device=device, dtype=dtype,
                                    low=low, high=high,
                                    requires_grad=requires_grad)),
            SampleInput(make_tensor((), device=device, dtype=dtype,
                                    low=low, high=high,
                                    requires_grad=requires_grad)))

# Metadata class for unary "universal functions (ufuncs)" that accept a single
# tensor and have common properties like:
class UnaryUfuncInfo(OpInfo):
    """Operator information for 'universal unary functions (unary ufuncs).'
    These are functions of a single tensor with common properties like:
      - they are elementwise functions
      - the input shape is the output shape
      - they typically have method and inplace variants
      - they typically support the out kwarg
      - they typically have NumPy or SciPy references
    See NumPy's universal function documentation
    (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
    about the concept of ufuncs.
    """

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref,  # a reference function
                 dtypes=floating_types(),
                 dtypesIfCPU=None,
                 dtypesIfCUDA=None,
                 dtypesIfROCM=None,
                 default_test_dtypes=(
                     torch.uint8, torch.long, torch.half, torch.bfloat16,
                     torch.float32, torch.cfloat),  # dtypes which tests check by default
                 domain=(None, None),  # the [low, high) domain of the function
                 handles_large_floats=True,  # whether the op correctly handles large float values (like 1e20)
                 handles_extremals=True,  # whether the op correctly handles extremal values (like inf)
                 handles_complex_extremals=True,  # whether the op correct handles complex extremals (like inf -infj)
                 supports_complex_to_float=False,  # op supports casting from complex input to real output safely eg. angle
                 sample_inputs_func=sample_inputs_unary,
                 sample_kwargs=lambda device, dtype, input: ({}, {}),
                 supports_sparse=False,
                 **kwargs):
        super(UnaryUfuncInfo, self).__init__(name,
                                             dtypes=dtypes,
                                             dtypesIfCPU=dtypesIfCPU,
                                             dtypesIfCUDA=dtypesIfCUDA,
                                             dtypesIfROCM=dtypesIfROCM,
                                             default_test_dtypes=default_test_dtypes,
                                             sample_inputs_func=sample_inputs_func,
                                             supports_sparse=supports_sparse,
                                             **kwargs)
        self.ref = ref
        self.domain = domain
        self.handles_large_floats = handles_large_floats
        self.handles_extremals = handles_extremals
        self.handles_complex_extremals = handles_complex_extremals
        self.supports_complex_to_float = supports_complex_to_float

        # test_unary_ufuncs.py generates its own inputs to test the consistency
        # of the operator on sliced tensors, non-contig tensors, etc.
        # `sample_kwargs` is a utility function to provide kwargs
        # along with those inputs if required (eg. clamp).
        # It should return two dictionaries, first holding kwarg for
        # torch operator and second one for reference NumPy operator.
        self.sample_kwargs = sample_kwargs

        # Epsilon to ensure grad and gradgrad checks don't test values
        #   outside a function's domain.
        self._domain_eps = 1e-5

def sample_inputs_tensor_split(op_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype,
                         low=None, high=None, requires_grad=requires_grad)

    args_cases = (
        # Cases with tensor indices.
        (torch.tensor([1, 2, 3]),),
        (torch.tensor(1),),
        (torch.tensor([1, 2, 3]), 1),
        # Cases with list of indices.
        ((2, 4),),
        ((2, 4), 1),
        ((2, 4), -1),
        # Cases with integer section.
        (3,),
        (3, 1),
        (3, -1),
    )

    def generator():
        for args in args_cases:
            yield SampleInput(make_input((S, S, S)), args=args)

    return list(generator())


def sample_inputs_linalg_det(op_info, device, dtype, requires_grad):
    kw = dict(device=device, dtype=dtype)
    inputs = [
        make_tensor((S, S), **kw),
        make_tensor((1, 1), **kw),  # 1x1
        random_symmetric_matrix(S, **kw),  # symmetric
        random_symmetric_psd_matrix(S, **kw),  # symmetric_psd
        random_symmetric_pd_matrix(S, **kw),  # symmetric_pd

        # dim2_null, rank1 and rank2 are disabled because of
        # https://github.com/pytorch/pytorch/issues/53364
        # we should re-enable them once the issue is solved
        # random_square_matrix_of_rank(S, S - 2, **kw),  # dim2_null
        # random_square_matrix_of_rank(S, 1, **kw),  # rank1
        # random_square_matrix_of_rank(S, 2, **kw),  # rank2

        random_fullrank_matrix_distinct_singular_value(S, **kw),  # distinct_singular_value
        make_tensor((3, 3, S, S), **kw),  # batched
        make_tensor((3, 3, 1, 1), **kw),  # batched_1x1
        random_symmetric_matrix(S, 3, **kw),  # batched_symmetric
        random_symmetric_psd_matrix(S, 3, **kw),  # batched_symmetric_psd
        random_symmetric_pd_matrix(S, 3, **kw),  # batched_symmetric_pd
        random_fullrank_matrix_distinct_singular_value(S, 3, 3, **kw),  # batched_distinct_singular_values
        make_tensor((0, 0), **kw),
        make_tensor((0, S, S), **kw),
    ]
    for t in inputs:
        t.requires_grad = requires_grad
    return [SampleInput(t) for t in inputs]

def sample_inputs_linalg_matrix_power(op_info, device, dtype, requires_grad):
    # (<matrix_size>, (<batch_sizes, ...>))
    test_sizes = [
        (1, ()),
        (2, (0,)),
        (2, (2,)),
    ]

    inputs = []
    for matrix_size, batch_sizes in test_sizes:
        size = batch_sizes + (matrix_size, matrix_size)
        for n in (0, 3, 5):
            t = make_tensor(size, device, dtype, requires_grad=requires_grad)
            inputs.append(SampleInput(t, args=(n,)))
        for n in [-4, -2, -1]:
            t = random_fullrank_matrix_distinct_singular_value(matrix_size, *batch_sizes, device=device, dtype=dtype)
            t.requires_grad = requires_grad
            inputs.append(SampleInput(t, args=(n,)))

    return inputs

def sample_inputs_hsplit(op_info, device, dtype, requires_grad):
    return (SampleInput(make_tensor((6,), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),)

def sample_inputs_vsplit(op_info, device, dtype, requires_grad):
    return (SampleInput(make_tensor((6, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),)

def sample_inputs_dsplit(op_info, device, dtype, requires_grad):
    return (SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),
            SampleInput(make_tensor((S, S, 6), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),)

def sample_inputs_linalg_multi_dot(op_info, device, dtype, requires_grad):
    # Each test case consists of the sizes in the chain of multiplications
    # e.g. [2, 3, 4, 5] generates matrices (2, 3) @ (3, 4) @ (4, 5)
    test_cases = [
        [1, 2, 1],
        [2, 0, 2],
        [0, 2, 2],
        [2, 2, 2, 2],
        [2, 3, 4, 5],
        [5, 4, 0, 2],
        [2, 4, 3, 5, 3, 2]
    ]

    result = []
    for sizes in test_cases:
        tensors = []
        for size in zip(sizes[:-1], sizes[1:]):
            t = make_tensor(size, device, dtype, requires_grad=requires_grad)
            tensors.append(t)
        result.append(SampleInput(tensors))

    return result

def sample_inputs_linalg_matrix_norm(op_info, device, dtype, requires_grad, **kwargs):
    sizes = ((2, 2), (2, 3, 2))
    ords = ('fro', 'nuc', inf, -inf, 1, -1, 2, -2)
    dims = ((-2, -1), (-1, 0))

    inputs: List[SampleInput] = []
    for size, ord, dim, keepdim in product(sizes, ords, dims, [True, False]):
        t = make_tensor(size, device, dtype, requires_grad=requires_grad)
        inputs.append(SampleInput(t, args=(ord, dim, keepdim)))

    return inputs

def sample_inputs_linalg_norm(op_info, device, dtype, requires_grad):
    test_sizes = [
        (S,),
        (0,),
        (S, S),
        (0, 0),
        (S, 0),
        (0, S),
        (S, S, S),
        (0, S, S),
        (S, 0, S),
        (0, 0, 0),
    ]

    vector_ords = (None, 0, 0.5, 1, 2, 3.5, inf, -0.5, -1, -2, -3.5, -inf)
    matrix_ords = (None, 'fro', 'nuc', 1, 2, inf, -1, -2, -inf)

    inputs = []

    for test_size in test_sizes:
        is_vector_norm = len(test_size) == 1
        is_matrix_norm = len(test_size) == 2

        for keepdim in [False, True]:
            inputs.append(SampleInput(
                make_tensor(
                    test_size, device, dtype, low=None, high=None,
                    requires_grad=requires_grad),
                kwargs=dict(
                    keepdim=keepdim)))

            if not (is_vector_norm or is_matrix_norm):
                continue

            ords = vector_ords if is_vector_norm else matrix_ords

            for ord in ords:

                inputs.append(SampleInput(
                    make_tensor(
                        test_size, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad),
                    args=(ord,),
                    kwargs=dict(
                        keepdim=keepdim)))

                if ord in ['nuc', 'fro']:
                    inputs.append(SampleInput(
                        make_tensor(
                            test_size, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                        kwargs=dict(
                            ord=ord,
                            keepdim=keepdim,
                            dim=(0, 1))))
        return inputs


def sample_inputs_norm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), (2,), '2'),
        ((S, S), (0,), '0'),
        ((S, S), (0.5,), '0_5'),
        ((S, S), (1,), '1'),
        ((S, S), (3,), '3'),
        ((S, S), (-1,), 'neg_1'),
        ((S, S), (-2,), 'neg_2'),
        ((S, S), (-0.5,), 'neg_0_5'),
        ((S, S), (-1.5,), 'neg_1_5'),
    )

    cases_nonzero_input = (
        ((S, S, S), (1.5,), '1_5_default'),
        ((S, S, S), (1.5, 1), '1_5_dim'),
        ((S, S, S), (1.5, -1), '1_5_neg_dim'),
        ((S, S, S), (1.5, 1, True), 'keepdim_1_5_dim'),
        ((S, S, S), (1.5, -1, True), 'keepdim_1_5_neg_dim'),
    )

    cases_negdim_base = (
        ((S, S), (-2, 1,), 'neg_2_2_dim'),
        ((S, S), (-1, 1,), 'neg_1_2_dim'),
        ((S, S), (0, 1,), '0_2_dim'),
        ((S, S), (1, 1,), '1_2_dim'),
        ((S, S), (2, 1,), '2_2_dim'),
        ((S, S), (3, 1,), '3_2_dim'),
        ((S, S, S), (2, 1), '2_dim'),
        ((S, S, S), (3, 1), '3_dim'),
        ((S, S, S), (2, 1, True), 'keepdim_2_dim'),
        ((S, S, S), (3, 1, True), 'keepdim_3_dim'),
        ((), (2, 0), '2_dim_scalar'),
        ((), (3, 0), '3_dim_scalar'),
        ((), (2, 0, True), 'keepdim_2_dim_scalar'),
        ((), (3, 0, True), 'keepdim_3_dim_scalar'),
    )

    cases_negdim = []
    for case in cases_negdim_base:
        cases_negdim.append(case)
        shape, args, name = case
        new_args = copy.deepcopy(list(args))
        new_args[1] *= -1
        cases_negdim.append((shape, tuple(new_args), name.replace("_dim", "_neg_dim")))

    def generator():
        for shape, args, name in itertools.chain(cases, cases_negdim):
            yield SampleInput(make_arg(shape), args=args, name=name)

        for shape, args, name in cases_nonzero_input:
            yield SampleInput(make_arg(shape, exclude_zero=True), args=args, name=name)

    return list(generator())


def sample_inputs_norm_fro(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), (), 'default'),
        ((S, S), ('fro',), 'fro_default'),
        ((S, S), ('fro', [0, 1],), 'fro'),
    )

    def generator():
        for shape, args, name in cases:
            yield SampleInput(make_arg(shape), args=args, name=name)

    return list(generator())


def sample_inputs_norm_nuc(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), ('nuc',), 'nuc'),
        ((S, S, S), ('nuc', [1, 2]), 'nuc_batched'),
    )

    def generator():
        for shape, args, name in cases:
            yield SampleInput(make_arg(shape), args=args, name=name)

    return list(generator())


def sample_inputs_norm_inf(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), (-inf,), '-inf'),
        ((S, S), (inf,), 'inf'),
        ((S, S), (inf, 1,), 'inf_2_dim'),
        ((S, S), (inf, -1,), 'inf_2_neg_dim'),
    )

    def generator():
        for shape, args, name in cases:
            yield SampleInput(make_arg(shape), args=args, name=name)

    return list(generator())


def sample_inputs_linalg_vector_norm(op_info, device, dtype, requires_grad, **kwargs):
    size_1D = (S,)
    size_2D = (2, 2)

    test_cases = [
        # input size, ord, dim args
        (size_1D, 2, None),
        (size_1D, 2, (0,)),
        (size_1D, 0, None),
        (size_1D, 0, (0,)),
        (size_1D, 0.9, None),
        (size_1D, 0.9, (0,)),
        (size_1D, 1, None),
        (size_1D, 1, (0,)),
        (size_1D, -2.1, None),
        (size_1D, -2.1, (0,)),
        (size_1D, inf, None),
        (size_1D, inf, (0,)),
        (size_1D, -inf, None),
        (size_1D, -inf, (0,)),

        (size_2D, 2, None),
        (size_2D, 2, (0,)),
        (size_2D, 2, (-1, 0)),
        (size_2D, 0, None),
        (size_2D, 0, (0,)),
        (size_2D, 0, (-1, 0)),
        (size_2D, 0.9, None),
        (size_2D, 0.9, (0,)),
        (size_2D, 0.9, (-1, 0)),
        (size_2D, 1, None),
        (size_2D, 1, (0,)),
        (size_2D, 1, (-1, 0)),
        (size_2D, -2.1, None),
        (size_2D, -2.1, (0,)),
        (size_2D, -2.1, (-1, 0)),
        (size_2D, inf, None),
        (size_2D, inf, (0,)),
        (size_2D, inf, (-1, 0)),
        (size_2D, -inf, None),
        (size_2D, -inf, (0,)),
        (size_2D, -inf, (-1, 0)),
    ]
    inputs = []

    for test_size, ord, dim in test_cases:
        for keepdim in [False, True]:
            inputs.append(SampleInput(
                make_tensor(
                    test_size, device, dtype,
                    low=None, high=None,
                    requires_grad=requires_grad),
                args=(ord,),
                kwargs=dict(
                    keepdim=keepdim,
                    dim=dim)))

    return inputs

# In order to use the kwarg alpha, partials should be used in an OpInfo's sample_inputs_func
# eg. sample_inputs_func=partial(sample_inputs_binary_pwise, alpha=2)
# Then one sample input would also be generated corresponding to the value of alpha provided.
# In the future, kwargs 'alpha_floating', 'alpha_integral' & 'alpha_complex' can be used to
# specify scalars of floating, integral & complex types as values for "alpha".
# Keyword argument `rhs_exclude_zero` is used to exclude zero values from rhs tensor argument
# This is necessary for operations like `true_divide`, where divide by zero throws an exception.
def sample_inputs_binary_pwise(op_info, device, dtype, requires_grad, extra_kwargs=None, **kwargs):
    if extra_kwargs is None:
        extra_kwargs = {}

    scalar = 3.14 + 3.14j if dtype.is_complex else (3.14 if dtype.is_floating_point else 3)
    scalar = 1 if dtype is torch.bool else scalar
    tests_list = [
        ((S, S, S), (S, S, S), False),
        ((S, S, S), (S, S), False),
        ((), (), False),
        ((S, S, S), (), False),
        ((S, S, S), scalar, False),
        ((), scalar, False)
    ]
    tests_with_lhs_broadcasting = [
        ((S, S), (S, S, S), True),
        ((), (S, S, S), True),
        ((S, 1, S), (M, S), True),
    ]
    test_cases = tests_list + tests_with_lhs_broadcasting  # type: ignore[operator]
    samples = []
    for first_shape, shape_or_scalar, broadcasts_input in test_cases:
        arg = shape_or_scalar

        if isinstance(shape_or_scalar, tuple):
            exclude_zero = kwargs.get('rhs_exclude_zero', False)
            arg = make_tensor(shape_or_scalar, device=device, dtype=dtype,
                              requires_grad=requires_grad, exclude_zero=exclude_zero)
        samples.append(SampleInput(make_tensor(first_shape, device=device, dtype=dtype,
                                               requires_grad=requires_grad),
                                   args=(arg,), kwargs=extra_kwargs,
                                   broadcasts_input=broadcasts_input))
    # Adds an extra sample using "alpha" if it's passed in kwargs
    if 'alpha' in kwargs:
        a = make_tensor((S, S, S), device=device, dtype=dtype, requires_grad=requires_grad)
        b = make_tensor((S, S, S), device=device, dtype=dtype, requires_grad=requires_grad)
        extra_kwargs['alpha'] = kwargs['alpha']
        sample = SampleInput(a, args=(b,), kwargs=extra_kwargs)
        samples.append(sample)
    return tuple(samples)


def sample_inputs_t(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return (SampleInput(make_arg((1, 2))),
            SampleInput(make_arg((2,))),
            SampleInput(make_arg(())))


def sample_inputs_mm(op_info, device, dtype, requires_grad, **kwargs):
    args_list = (
        ((S, M), (M, S)),
    )
    inputs = tuple(SampleInput(make_tensor(first_shape, device, dtype,
                                           requires_grad=requires_grad),
                               args=(make_tensor(second_shape, device, dtype,
                                     requires_grad=requires_grad),))
                   for first_shape, second_shape in args_list)
    return inputs

def sample_inputs_addmm(op_info, device, dtype, requires_grad, **kwargs):
    alpha_val = kwargs.get('alpha', 2 + 3j if dtype.is_complex else 0.6)
    beta_val = kwargs.get('beta', 1 + 2j if dtype.is_complex else 0.2)
    tests_list = [
        ((2, 3), (2, 2), (2, 3), False)
    ]
    tests_with_lhs_broadcasting = [
        ((1,), (2, 2), (2, 3), True),
        ((), (2, 2), (2, 3), True)
    ]
    test_cases = tests_list + tests_with_lhs_broadcasting  # type: ignore[operator]
    inputs = tuple(SampleInput(make_tensor(shape_a, device, dtype, requires_grad=requires_grad),
                               args=(make_tensor(shape_b, device, dtype,
                                                 requires_grad=requires_grad),
                                     make_tensor(shape_c, device, dtype,
                                                 requires_grad=requires_grad)),
                               kwargs={'alpha': alpha_val, 'beta': beta_val},
                               broadcasts_input=broadcasts_input)
                   for shape_a, shape_b, shape_c, broadcasts_input in test_cases)
    return inputs

def sample_inputs_mv(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_bmm(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((M, S, M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((M, M, S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_dot_vdot(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_addmv(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (((S,), (S, M), (M,), 1, 1, False),
                  ((S,), (S, M), (M,), 0.2, 0.6, False),
                  )

    test_cases_with_broadcast = (((1,), (S, M), (M,), 1, 1, True),
                                 ((1,), (S, M), (M,), 0.2, 0.6, True),
                                 ((), (S, M), (M,), 1, 1, True),
                                 ((), (S, M), (M,), 0.2, 0.6, True),
                                 )

    cases = test_cases + test_cases_with_broadcast
    sample_inputs = []
    for input_args in cases:
        args = (make_tensor(input_args[0], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(input_args[1], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(input_args[2], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad))
        alpha, beta = input_args[3], input_args[4]
        broadcasts_input = input_args[5]
        sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]), kwargs=dict(beta=beta, alpha=alpha),
                                         broadcasts_input=broadcasts_input))
    return tuple(sample_inputs)

def sample_inputs_addbmm(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [((S, M), (S, S, S), (S, S, M), 1, 1),
                  ((1,), (S, S, S), (S, S, M), 1, 1),
                  ((S, M), (S, S, S), (S, S, M), 0.6, 0.2),
                  ((1,), (S, S, S), (S, S, M), 0.6, 0.2),
                  ((), (S, S, S), (S, S, M), 1, 1),
                  ((), (S, S, S), (S, S, M), 0.6, 0.2),
                  ]
    sample_inputs = []
    for input_args in test_cases:
        args = (make_tensor(input_args[0], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(input_args[1], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(input_args[2], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad))
        alpha, beta = input_args[3], input_args[4]
        sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]), kwargs=dict(beta=beta, alpha=alpha)))
        if dtype.is_complex:
            sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]),
                                             kwargs=dict(beta=beta * (1 + 2j), alpha=alpha * (2 + 3j))))

    return tuple(sample_inputs)

def sample_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [(((S, S), (S, S), (S, S)), False),
                  (((S, S), (S, 1), (1, S)), False),
                  (((1,), (S, S, 1), (1, S)), True),
                  (((), (), ()), False),
                  (((S, S), (), ()), True),
                  (((), (S, S, 1), (1, S)), True)
                  ]

    sample_inputs = []
    for input_args, broadcasts_input in test_cases:
        args = tuple(make_tensor(arg, device, dtype, requires_grad=requires_grad) if isinstance(arg, tuple) else arg
                     for arg in input_args)
        sample_inputs.append(SampleInput(args[0], args=args[1:], broadcasts_input=broadcasts_input))

        sample_inputs.append(SampleInput(args[0], args=args[1:], kwargs=dict(value=3.14), broadcasts_input=broadcasts_input))

    return tuple(sample_inputs)

def sample_inputs_baddbmm(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [((S, S, M), (S, S, S), (S, S, M), 1, 1, False),
                  ((1,), (S, S, S), (S, S, M), 1, 1, True),
                  ((S, S, M), (S, S, S), (S, S, M), 0.6, 0.2, False),
                  ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ((), (S, S, S), (S, S, M), 1, 1, True),
                  ((), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ]
    sample_inputs = []
    for (input_shape, batch1_shape, batch2_shape, alpha, beta, broadcasts_input) in test_cases:
        args = (make_tensor(input_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(batch1_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(batch2_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad))
        sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]),
                             kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=broadcasts_input))
        if dtype.is_complex:
            sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]),
                                             kwargs=dict(beta=beta * (1 + 2j), alpha=alpha * (2 + 3j)),
                                             broadcasts_input=broadcasts_input))
    return tuple(sample_inputs)

def sample_inputs_addr(op_info, device, dtype, requires_grad, **kwargs):
    input1 = SampleInput(
        make_tensor((S, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)))

    input2 = SampleInput(
        make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)),
        broadcasts_input=True)

    if dtype.is_complex:
        alpha, beta = 0.1 + 0.3j, 0.4 + 0.6j
    elif dtype.is_floating_point:
        alpha, beta = 0.2, 0.6
    else:
        alpha, beta = 2, 3

    input3 = SampleInput(
        make_tensor((S, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)),
        kwargs=dict(beta=beta, alpha=alpha))

    input4 = SampleInput(
        make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)),
        kwargs=dict(beta=beta, alpha=alpha),
        broadcasts_input=True)

    return (input1, input2, input3, input4)

def sample_inputs_xlogy(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((S, S), device, dtype, low=0, high=None, requires_grad=requires_grad),
            )
        ),
    )


def sample_inputs_xlog1py(self, device, dtype, requires_grad):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def generator():
        # same shape
        yield SampleInput(make_arg((S, S)), args=(make_arg((S, S), low=-1),))
        # rhs broadcast
        yield SampleInput(make_arg((S, S)), args=(make_arg((S,), low=-1),))
        # all zero `x`
        with torch.no_grad():
            x = make_arg((S, S))
            x.fill_(0)
        yield SampleInput(x, args=(make_arg((S, S), low=-1),))

        # randomly zero-masked `x`
        x = make_arg((S, S))
        y = make_arg((S, S), low=-1)
        with torch.no_grad():
            x[torch.rand(x.shape) > 0.5] = 0
        yield SampleInput(x, args=(y,))

        # Scalar x
        # `input` has to be a tensor
        # yield SampleInput(0, args=(make_arg((S, S), low=-1),))
        # yield SampleInput(2.1, args=(make_arg((S, S), low=-1),))

        # Scalar y
        yield SampleInput(make_arg((S, S)), args=(-0.5,))
        yield SampleInput(make_arg((S, S)), args=(1.2,))

    return list(generator())

def sample_inputs_zero_(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = ((), (S, S, S), (S,))

    def generator():
        for shape in cases:
            yield(SampleInput(make_arg(shape)))

    return list(generator())


def sample_inputs_logsumexp(self, device, dtype, requires_grad):
    inputs = (
        ((), (0,), True),
        ((S, S), (1,), True),
        ((S, S), (1,), False)
    )
    samples = []

    for shape, dim, keepdim in inputs:
        t = make_tensor(shape, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, args=(dim, keepdim)))

    return tuple(samples)

def sample_inputs_logcumsumexp(self, device, dtype, requires_grad):
    inputs = (
        ((S, S, S), 0),
        ((S, S, S), 1),
        ((), 0),
    )
    samples = []

    for shape, dim in inputs:
        t = make_tensor(shape, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, args=(dim,)))

    return tuple(samples)

def sample_inputs_trace(self, device, dtype, requires_grad, **kwargs):
    return (SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad))),)


def sample_inputs_renorm(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, S, S), (2, 1, 0.5)),
             ((S, S, S), (2, -1, 0.5)),
             ((S, S, S), (1, 2, 3)),
             ((S, S, S), (float('inf'), 2, 0.5)),
             )

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_transpose_swapdims(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((1, 2, 3), (-1, -2)),
             ((1, 2, 3), (-1, 2)),
             ((1, 2, 3), (1, -2)),
             ((1, 2, 3), (1, 2)),
             ((), (0, 0)),
             ((1, ), (0, 0)),
             ((M, M), (0, 1)),
             ((S, S, S), (2, 0)), )

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates always invertible input for linear algebra ops using
    random_fullrank_matrix_distinct_singular_value.
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 0]
    out = []
    for batch, n in product(batches, ns):
        a = random_fullrank_matrix_distinct_singular_value(n, *batch, dtype=dtype, device=device)
        a.requires_grad = requires_grad
        out.append(SampleInput(a))
    return out

def sample_inputs_linalg_cond(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    # autograd is not supported for inputs with zero number of elements
    shapes = ((S, S),
              (2, S, S),
              (2, 1, S, S), )

    def generator():
        for shape in shapes:
            yield SampleInput(make_arg(shape))

    return list(generator())

def np_sinc_with_fp16_as_fp32(x):
    # Wraps numpy's sinc function so that fp16 values are promoted to fp32
    # before sinc is invoked. Context: numpy's sinc returns NaN when evaluated
    # at 0 for fp16.
    if x.dtype == np.float16:
        return np.sinc(x.astype(np.float32))
    else:
        return np.sinc(x)

def sample_inputs_broadcast_to(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, 1, 1), (S, S, S)),
        ((S, 1, S), (S, S, S)),
        ((S, 1), (S, S, S)),
        ((1,), (S, S, S)),
        ((1, S), (1, 1, S)),
        ((), ()),
        ((), (1, 3, 2)),
    )

    return tuple(
        SampleInput(
            make_tensor(size, device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(shape,)) for size, shape in test_cases)

def sample_inputs_bitwise_shift(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        (S, S, S),
        (S,),
        (),
    )

    sample_inputs = []
    for size in test_cases:
        tensor1 = make_tensor(size, device, dtype, low=-32, high=32, requires_grad=requires_grad)
        tensor2 = make_tensor(size, device, dtype, low=0, high=5, requires_grad=requires_grad)
        sample_inputs.append(SampleInput(tensor1, args=(tensor2,)))
        sample_inputs.append(SampleInput(tensor1, args=(2,)))

    return tuple(sample_inputs)


def sample_inputs_cdist(op_info, device, dtype, requires_grad, **kwargs):
    small_S = 2
    test_cases = (
        ((S, S, 2), (S, S + 1, 2)),
        ((S, S), (S, S)),
        ((S, S, S), (S, S, S)),
        ((3, 5), (3, 5)),
        ((2, 3, 5), (2, 3, 5)),
        ((1, 2, 3), (1, 2, 3)),
        ((1, 1), (S, 1)),
        ((0, 5), (4, 5)),
        ((4, 5), (0, 5)),
        ((0, 4, 5), (3, 5)),
        ((4, 5), (0, 3, 5)),
        ((0, 4, 5), (1, 3, 5)),
        ((1, 4, 5), (0, 3, 5)),
        # Using S here would make this one test take 9s
        ((small_S, small_S, small_S + 1, 2), (small_S, small_S, small_S + 2, 2)),
        ((small_S, 1, 1, small_S), (1, small_S, small_S)),
        ((1, 1, small_S), (small_S, 1, small_S, small_S)),
    )

    samples = []
    for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
        # FIXME add an override for JIT and revert 0. back to 0
        # since it's accepted by eager
        for p in [0., 1., 2., 3., 0.5, 1.5, 2.5, float("inf")]:
            for t1_size, t2_size in test_cases:
                # The args should never be non-contiguous as this is not supported in the backward
                samples.append(SampleInput(
                    make_tensor(t1_size, device, dtype, requires_grad=requires_grad, noncontiguous=False),
                    args=(make_tensor(t2_size, device, dtype, requires_grad=requires_grad, noncontiguous=False), p, cm)))

    return samples


def sample_inputs_fill_(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype,
                       low=None, high=None, requires_grad=requires_grad)

    cases = (((S, S, S), (1,)),
             ((), (1,)),
             # For requires_grad=False below,
             # check https://github.com/pytorch/pytorch/issues/59137
             ((S, S, S), (make_arg((), requires_grad=False),)))

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_comparison_ops(self, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, S, S), (S, S, S), False),
        ((S, S, S), (), False),
        ((S, S, S), (1,), False),
        ((S,), (1,), False),
        ((), (), False),
    )
    test_cases_lhs_broadcasting = (
        ((S, 1, S), (S, S, S), True),
        ((1,), (S, S, S), True),
        ((1, S), (1, 1, S), True),
        ((), (0,), True),
        ((), (S, S, S), True),
    )
    cases = test_cases + test_cases_lhs_broadcasting
    sample_inputs = list(SampleInput(make_tensor(first_shape, device, dtype,
                                                 requires_grad=requires_grad),
                                     args=(make_tensor(second_shape, device, dtype,
                                                       requires_grad=requires_grad),),
                                     broadcasts_input=broadcasts_input)
                         for first_shape, second_shape, broadcasts_input in cases)
    equal_tensors_non_bool = (
        ([[[-8, 6], [9, 0]], [[0, 5], [5, 7]]]),
        ([[[6, 5]], [[1, -5]]]),
        ([[2], [-1]]),
        ([0, -6]),
        ([3],),
    )
    equal_tensors_bool = (
        ([[[1, 0], [0, 0]], [[0, 1], [1, 0]]]),
        ([[[1, 1]], [[1, 0]]]),
        ([[1], [0]]),
        ([0, 1]),
        ([1],),
    )
    more_cases = equal_tensors_bool if dtype is torch.bool else equal_tensors_non_bool
    more_inputs = list(SampleInput(torch.tensor(elements, device=device, dtype=dtype,
                                                requires_grad=requires_grad),
                                   args=(torch.tensor(elements, device=device, dtype=dtype,
                                                      requires_grad=requires_grad),))
                       for elements in more_cases)
    sample_inputs = [*sample_inputs, *more_inputs]
    return tuple(sample_inputs)


def sample_inputs_stack(op_info, device, dtype, requires_grad, **kwargs):
    tensors = [
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
    ]

    return (SampleInput(tensors, args=(0,)),)

def sample_inputs_hstack_dstack_vstack(op_info, device, dtype, requires_grad, **kwargs):
    tensors = [
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
    ]

    return (SampleInput(tensors),)

def sample_inputs_hypot(op_info, device, dtype, requires_grad):
    input = make_tensor((S, S), device, dtype, requires_grad=requires_grad)
    args = make_tensor((S, S), device, dtype, requires_grad=requires_grad)

    return (
        SampleInput(input, args=(args,)),
    )

def sample_inputs_gather(op_info, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, gather_variable((S, S), 1, M, True, device=device))),
        SampleInput(
            make_tensor((M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(1, gather_variable((M, S // 2), 0, S, True, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor([0], dtype=torch.int64, device=device))),
        SampleInput(
            make_tensor((S,), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor(0, dtype=torch.int64, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor(0, dtype=torch.int64, device=device))),
    )


def sample_inputs_take_along_dim(op_info, device, dtype, requires_grad, **kwargs):
    return (SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S), 1, S, True, device=device), 0)),

            # `indices` broadcast
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((1, S // 2), 0, S, True, device=device), 1)),

            # `self` broadcast
            SampleInput(make_tensor((1, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device), 1)),

            # without `dim` arg
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device), )),
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device),)),
            )

def sample_inputs_amax_amin(op_info, device, dtype, requires_grad, **kwargs):
    # Ordered as (shape, positional args, kwargs)
    test_cases: Tuple[tuple, tuple, dict] = (  # type: ignore[assignment]
        ((S, S, S), (), {}),
        ((S, S, S), (1,), {}),
        ((S, S, S), ((1, 2,),), {}),
        ((S, S, S), (1,), {'keepdim': True}),
        ((), (0,), {}),
        ((), (), {}),
        ((), (0,), {'keepdim': True}),
    )
    return tuple(SampleInput((make_tensor(size, device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad)),
                             args=args, kwargs=kwargs)
                 for size, args, kwargs in test_cases)

def sample_inputs_argmax_argmin(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((2, 2, 2), ()),
        ((2, 2, 2), (0,)),
        ((2, 2, 2), (1,)),
        ((2, 2, 2), (2,)),
        ((2, 2, 2), (2, True,)),
        ((2, 2, 2), (None,)),
        ((), (0,)),
        ((), ()),
        ((), (None, True,)),
        ((1,), ()),
        ((1,), (0,)),
        ((1,), (0, True)),
        ((2,), ()),
        ((2,), (0,)),
        ((2,), (0, True)),
        ((2, 2, 3), ()),
        ((2, 2, 3), (0,)),
        ((2, 2, 3), (1,)),
        ((2, 2, 3), (None, True)),
    )
    return tuple(SampleInput((make_tensor(size, device, dtype,
                                          requires_grad=requires_grad)),
                             args=args)
                 for size, args in test_cases)

def sample_inputs_diff(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((1,), 0, None, None),
        ((S,), 0, None, None),
        ((S, 1), 0, None, None),
        ((S, 1), 1, None, None),
        ((S, S), 0, None, None),
        ((S, S), 1, None, None),
        ((S, S), 0, (1, S), (2, S)),
        ((S, S), 0, None, (2, S)),
        ((S, S, S), 1, None, None),
        ((S, S, S), 1, (S, 1, S), (S, 1, S)),)

    sample_inputs = []
    for size, dim, size_prepend, size_append in test_cases:
        args = (make_tensor(size, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad), 1, dim,
                make_tensor(size_prepend, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad) if size_prepend else None,
                make_tensor(size_append, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad) if size_append else None)
        sample_inputs.append(SampleInput(args[0], args=args[1:]))

    return tuple(sample_inputs)

def sample_inputs_histogram(op_info, device, dtype, requires_grad):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))

    sample_inputs = []
    for size, bin_ct, weighted, density in product(sizes, range(1, 5), [False, True], [False, True]):
        input_tensor = make_arg(size)
        weight_tensor = make_arg(size) if weighted else None

        sample_inputs.append(SampleInput(input_tensor, args=(bin_ct,),
                                         kwargs=dict(weight=weight_tensor, density=density)))

        bins_tensor = make_arg((bin_ct + 1,))
        sample_inputs.append(SampleInput(input_tensor, args=(bins_tensor,),
                                         kwargs=dict(weight=weight_tensor, density=density)))

    return sample_inputs

def sample_inputs_gradient(op_info, device, dtype, requires_grad):
    sample_inputs = []
    test_cases_float = (
        ((S,), None, None, 1),
        ((S,), 2., None, 1),
        ((S, S), None, None, 2),
        ((S, S), [2.0, 2.1], None, 1),
        ((S, S), [2.0, 2.1], (0, 1), 1),
        ((4, 4, 4), [2., 1.], (0, 1), 2),
    )
    for size, spacing, dim, edge_order in test_cases_float:
        t = make_tensor(size, device, dtype, low=None, high=None, requires_grad=requires_grad)
        sample_inputs.append(SampleInput(t, kwargs=dict(dim=dim, spacing=spacing, edge_order=edge_order)))

    test_cases_tensor = (
        ((3, 3, 3), ((1.1, 2.0, 3.5), (4.0, 2, 6.0)), (0, -1), 1),
        ((3, 3, 3), ((1.0, 3.0, 2.0), (8.0, 6.0, 1.0)), (0, 1), 2),
    )
    for size, coordinates, dim, edge_order in test_cases_tensor:
        t = make_tensor(size, device, dtype, low=None, high=None, requires_grad=requires_grad)
        coordinates_tensor_list = []
        for coords in coordinates:
            a = torch.tensor(coords, dtype=dtype, device=device)
            coordinates_tensor_list.append(a)
        sample_inputs.append(SampleInput(t, kwargs=dict(dim=dim, spacing=coordinates_tensor_list, edge_order=edge_order)))

    return tuple(sample_inputs)

def sample_inputs_index_select(op_info, device, dtype, requires_grad):
    return (
        SampleInput(
            make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, index_variable(2, S, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor([0], dtype=torch.int64, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor(0, dtype=torch.int64, device=device))),
    )

def sample_inputs_getitem(op_info, device, dtype, requires_grad, **kwargs):
    test_args = [
        ([1, 2],),
        (slice(0, 3),),
        ([slice(0, 3), 1],),
        ([[0, 2, 3], [1, 3, 3], [0, 0, 2]],),
        ([[0, 0, 3], [1, 1, 3], [0, 0, 2]],),
        ([slice(None), slice(None), [0, 3]],),
        ([slice(None), [0, 3], slice(None)],),
        ([[0, 3], slice(None), slice(None)],),
        ([[0, 3], [1, 2], slice(None)],),
        ([[0, 3], ],),
        ([[0, 3], slice(None)],),
        ([[0, 3], Ellipsis],),
        ([[0, 2, 3], [1, 3, 3], torch.LongTensor([0, 0, 2])],),
        (index_variable(2, S, device=device),),
        (mask_not_all_zeros((S,)),),
    ]

    return tuple(SampleInput(
        make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=args)
        for args in test_args)

def sample_inputs_index_put(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    for accumulate in [False, True]:
        # Test with indices arg
        inputs.append(SampleInput(
            make_tensor((S, S,), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                (index_variable(2, S, device=device), ),
                make_tensor((2, S), device, dtype, low=None, high=None)),
            kwargs=dict(accumulate=accumulate)))

        # Test with mask arg
        mask = torch.zeros(S, dtype=torch.bool) if accumulate else mask_not_all_zeros((S,))
        inputs.append(SampleInput(
            make_tensor((S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                (mask, ),
                make_tensor((S,), device, dtype, low=None, high=None),),
            kwargs=dict(accumulate=accumulate)))

    return inputs

# Missing to test the nondeterminism of the operation
# https://github.com/pytorch/pytorch/issues/53352
def sample_inputs_index_add(op_info, device, dtype, requires_grad, **kwargs):
    # These testa are pretty much the same as those from index_copy.
    # Perhaps merge?
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    t = make_arg((S, S))
    s = make_arg((S, S))
    # non-contiguous target
    t_nonctg = t.transpose(0, 1)
    # non-contiguous source
    s_nonctg = s.transpose(0, 1)

    idx = make_arg((S,), dtype=torch.int64, low=0, high=S)
    idx_nonctg = make_arg((S,), dtype=torch.int64, low=0, high=S, noncontiguous=True)
    samples = [SampleInput(tensor, args=(1, idx, source))
               for tensor, idx, source in product([t, t_nonctg], [idx, idx_nonctg], [s, s_nonctg])]
    samples.extend(SampleInput(tensor, args=(1, idx, source), kwargs=dict(alpha=a))
                   for tensor, idx, source, a in product([t, t_nonctg], [idx, idx_nonctg], [s, s_nonctg], [-1, 0, 2]))

    # Add scalar cases
    scalar_sizes = [(), (1,)]
    ts = (make_arg(size) for size in scalar_sizes)
    idxs = (make_arg(size, dtype=torch.int64, low=0, high=1) for size in scalar_sizes)
    ss = (make_arg(size) for size in scalar_sizes)

    samples.extend(SampleInput(t, args=(0, idx, s)) for t, idx, s in product(ts, idxs, ss))
    samples.extend(SampleInput(t, args=(0, idx, s), kwargs=dict(alpha=a)) for t, idx, s, a in product(ts, idxs, ss, [-1, 0, 2]))
    return samples

def sample_inputs_sort(op_info, device, dtype, requires_grad, **kwargs):
    def apply_grad(t):
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            t.requires_grad_(requires_grad)

    def small_3d_unique(dtype, device):
        res = torch.randperm(S * S * S, dtype=torch.int64, device=device).view(S, S, S)
        res = res.to(dtype)
        apply_grad(res)
        return res

    def large_1d_unique(dtype, device):
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype)
        apply_grad(res)
        return res

    samples = []
    # Test case for large tensor.
    largesample = SampleInput(large_1d_unique(dtype, device))
    samples.append(largesample)

    # Test cases for small 3d tensors.
    # Imitates legacy tests from test/test_torch.py
    t = small_3d_unique(dtype, device)
    dims = range(-3, 3)
    flag = [True, False]
    for dim, descending, stable in product(dims, flag, flag):
        # default schema without stable sort
        samples.append(SampleInput(t, args=(dim, descending)))
        # schema with stable sort, no CUDA support yet
        if torch.device(device).type == 'cpu':
            samples.append(
                SampleInput(t, kwargs=dict(dim=dim, descending=descending, stable=stable))
            )

    # Test cases for scalar tensor
    scalar = torch.tensor(1, dtype=dtype, device=device)
    apply_grad(scalar)
    samples.append(SampleInput(scalar))
    samples.append(SampleInput(scalar, args=(0,)))
    samples.append(SampleInput(scalar, args=(0, True)))
    # no CUDA support for stable sort yet
    if not device.startswith('cuda'):
        samples.append(SampleInput(scalar, kwargs=dict(stable=True)))
        samples.append(SampleInput(scalar, kwargs=dict(dim=0, stable=True)))
        samples.append(SampleInput(scalar, kwargs=dict(dim=0, descending=True, stable=True)))
    return samples

def sample_inputs_index_fill(op_info, device, dtype, requires_grad, **kwargs):
    samples = []
    t = make_tensor((S, S, S), device, dtype,
                    low=None, high=None,
                    requires_grad=requires_grad)
    fill_val = torch.tensor(-1 + 1j if t.is_complex() else -1)
    # non-contiguous input
    t01 = t.transpose(0, 1)
    t02 = t.transpose(0, 2)
    t12 = t.transpose(1, 2)
    idx = index_variable(1, S, device=device)
    # non-contiguous index
    idx_nonctg = torch.empty_strided((S,), (2,), device=device, dtype=torch.int64)
    idx_nonctg.copy_(idx)
    for d in range(t.dim()):
        for tensor in [t, t01, t02, t12]:
            samples.append(SampleInput(tensor, args=(d, idx, fill_val)))
            samples.append(SampleInput(tensor, args=(d, -idx - 1, fill_val)))
            samples.append(SampleInput(tensor, args=(d, idx_nonctg, fill_val)))

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    index_tensor = partial(torch.tensor, device=device, dtype=torch.long)

    def unique_idx(numel, max_idx):
        # Generate unique random indices vector of `numel`
        # elements in range [0, max_idx).
        indices = random.sample(range(max_idx), numel)
        return index_tensor(indices)

    samples.append(SampleInput(make_arg((S, S)), args=(0, unique_idx(2, S), 2)))
    samples.append(SampleInput(make_arg((S, S)), args=(0, unique_idx(2, S), make_arg(()))))
    samples.append(SampleInput(make_arg((S, S)), args=(0, index_tensor(0), 2)))
    samples.append(SampleInput(make_arg(()), args=(0, index_tensor([0]), 2)))
    samples.append(SampleInput(make_arg(()), args=(0, index_tensor(0), 2)))

    # Duplicate indices
    samples.append(SampleInput(make_arg((S, S)), args=(0, index_tensor([0, 0]), 2)))
    samples.append(SampleInput(make_arg((S, S)), args=(0, index_tensor([0, 0, 2]), make_arg(()))))

    return samples

def sample_inputs_max_min_binary(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    args_for_binary_op = (
        ((S, S, S), (S, S, S),),
        ((S, S, S), (S,),),
        ((S,), (S, S, S),),
        ((S, 1, S), (S, S),),
        ((S, S), (S, S),),
        ((), (),),
        ((S, S, S), (),),
        ((), (S, S, S),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=(make_tensor(other_tensor, device, dtype,
                                                 low=None, high=None,
                                                 requires_grad=requires_grad),),))
                  for input_tensor, other_tensor in args_for_binary_op)
    return inputs

def sample_inputs_hardswish(self, device, dtype, requires_grad):
    N = 5
    # make sure we are testing -3 -> 3 range. default is -10 -> 10 so maybe unnecessary ?
    tensors = [SampleInput(make_tensor((N * 2, N * 2), device=device, dtype=dtype,
               requires_grad=requires_grad, low=-5, high=5)) for _ in range(1, N)]
    return tensors

def sample_inputs_gelu(self, device, dtype, requires_grad):
    N = 5
    tensors = [SampleInput(make_tensor((N * 2, N * 2), device=device, dtype=dtype,
               requires_grad=requires_grad, low=-3, high=3)) for _ in range(1, N)]
    return tensors

def sample_inputs_max_min_reduction_with_dim(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    args_for_reduction_with_dim = (
        ((S, S, S), (1,),),
        ((S, S, S), (1, True, ),),
        ((), (0,),),
        ((), (0, True,),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=args,))
                  for input_tensor, args in args_for_reduction_with_dim)
    return inputs

def sample_inputs_max_min_reduction_no_dim(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    inputs.append(SampleInput(make_tensor((S, S, S), device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad),))
    inputs.append(SampleInput(make_tensor((), device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad),))
    return inputs

# Generates input tensors for testing reduction ops
def _generate_reduction_inputs(device, dtype, requires_grad):
    yield make_tensor((), device, dtype, requires_grad=requires_grad)
    yield make_tensor((2,), device, dtype, requires_grad=requires_grad)
    yield make_tensor((2, 3), device, dtype, requires_grad=requires_grad, noncontiguous=True)
    yield make_tensor((3, 2, 1, 2, 2), device, dtype, requires_grad=requires_grad)

# Generates a subset of possible dim and keepdim kwargs for a tensor
# with ndim dims appropriate for testing. If supports_multiple_dims
# is True (default) then dim kwarg can be a list of dims.
def _generate_reduction_kwargs(ndim, supports_multiple_dims=True):
    for keepdim in [True, False]:
        # Always test reducing inner and outer most dimensions
        yield {'dim': 0, 'keepdim': keepdim}
        yield {'dim': -1, 'keepdim': keepdim}

        # Also reduce middle dimension
        if ndim > 2:
            yield {'dim': ndim // 2, 'keepdim': keepdim}

        if supports_multiple_dims:
            # Always test reducing all dims
            yield {'dim': tuple(range(ndim)), 'keepdim': keepdim}

            # Test reducing both first and last dimensions
            if ndim > 1:
                yield {'dim': (0, ndim - 1), 'keepdim': keepdim}

            # Test reducing every other dimension starting with the second
            if ndim > 3:
                yield {'dim': tuple(range(1, ndim, 2)), 'keepdim': keepdim}

# Wraps sample_inputs_reduction function to provide the additional supports_multiple_dims args
def sample_inputs_reduction_wrapper(supports_multiple_dims):
    # Generates sample inputs for reduction ops that contain the input tensor
    # and dim and keepdim kwargs. If a reduction op needs to test additional
    # args/kwargs then create a separate sample_inputs function
    def fn(op_info, device, dtype, requires_grad):
        inputs = []

        for t in _generate_reduction_inputs(device, dtype, requires_grad):
            # Add case without dim and keepdim kwargs
            inputs.append(SampleInput(t))
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims):
                inputs.append(SampleInput(t, kwargs=kwargs))

        return inputs

    return fn

def sample_inputs_reduction_quantile(op_info, device, dtype, requires_grad):
    test_quantiles = (0.5, make_tensor((2,), device, dtype, low=0, high=1))
    test_interpolations = ['linear', 'midpoint']

    inputs = []
    for quantiles in test_quantiles:
        for t in _generate_reduction_inputs(device, dtype, requires_grad):
            # Add case without dim and keepdim kwargs
            inputs.append(SampleInput(t, args=(quantiles,)))
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims=False):
                # Interpolation kwarg for now is only supported when providing both dim and keepdim
                for interpolation in test_interpolations:
                    kwargs['interpolation'] = interpolation
                    inputs.append(SampleInput(t, args=(quantiles,), kwargs=kwargs))

    return inputs

def sample_inputs_leaky_relu(op_info, device, dtype, requires_grad):
    N = 10
    tensors = [SampleInput(make_tensor((N, N), device=device, dtype=dtype,
               requires_grad=requires_grad)) for _ in range(1, N)]
    return tensors

def sample_inputs_topk(op_info, device, dtype, requires_grad, **kwargs):
    def get_tensor_input(size):
        return make_tensor(size, device, dtype, requires_grad=requires_grad)

    inputs = []
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3,)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1, True, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2, True, True)))

    inputs.append(SampleInput(get_tensor_input(()), args=(1,)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0, True, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1, True, True)))

    return inputs

def sample_inputs_outer(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    arg_a = make_tensor((S,), device, dtype, requires_grad=requires_grad)
    arg_b = make_tensor((M,), device, dtype, requires_grad=requires_grad)
    inputs.append(SampleInput(arg_a, args=(arg_b,)))
    return inputs

def sample_inputs_dist(op_info, device, dtype, requires_grad):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    sizes = ((S, S, S), (S,), (S, 1, S), (), (S, S))
    ps = (2, 4)

    def generate_samples():
        for size_x, size_y, p in product(sizes, sizes, ps):
            yield SampleInput(make_arg(size_x), args=(make_arg(size_y), p))

    return list(generate_samples())

# Missing to test the nondeterminism of the operation
# https://github.com/pytorch/pytorch/issues/53352
def sample_inputs_index_copy(op_info, device, dtype, requires_grad, **kwargs):
    def make_arg(shape, low=None, high=None, dtype=dtype):
        return make_tensor(shape, device=device, dtype=dtype,
                           low=low, high=high,
                           requires_grad=requires_grad)

    t = make_arg((S, S))
    s = make_arg((S, S))
    # non-contiguous input
    t01 = t.transpose(0, 1)
    # non-contiguous input
    s01 = s.transpose(0, 1)

    # idx is a permutation of 0...S-1 for this function to be deterministic
    idx = torch.randperm(S, device=device, dtype=torch.int64)
    # non-contiguous index
    idx_nonctg = torch.repeat_interleave(idx, 2, dim=-1)[::2]
    # index_copy_ does not support negative indices
    # idx_neg = -idx - 1
    samples = [SampleInput(tensor, args=(1, idx, source))
               for tensor, idx, source in product([t, t01], [idx, idx_nonctg], [s, s01])]

    # Add scalar cases
    scalar_sizes = [(), (1,)]
    ts = (make_arg(size) for size in scalar_sizes)
    idxs = (make_arg(size, dtype=torch.int64, low=0, high=1) for size in scalar_sizes)
    ss = (make_arg(size) for size in scalar_sizes)

    samples.extend(SampleInput(t, args=(0, idx, s)) for t, idx, s in product(ts, idxs, ss))
    return samples

def sample_inputs_mode(op_info, device, dtype, requires_grad):
    inputs = []
    args = (
        ((S, S, S), (),),
        ((S, S, S), (1, ),),
        ((S, S, S), (1, True, ),),
        ((), (),),
        ((), (0,),),
        ((), (0, True,),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=args,))
                  for input_tensor, args in args)
    return inputs

# Missing to test the nondeterminism of the operation
# https://github.com/pytorch/pytorch/issues/53352
def sample_inputs_put(op_info, device, dtype, requires_grad):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)

    S = 3

    def gen_inputs():
        # Generic inputs
        tgt_gen = (make_arg((S, S), noncontiguous=not ctg) for ctg in (True, False))
        src_gen = (make_arg((S,), noncontiguous=not ctg) for ctg in (True, False))
        idx = torch.randperm(S * S, device=device, dtype=torch.int64)[:S]
        idx_nonctg = torch.repeat_interleave(idx, 2, dim=-1)[::2]
        idx_neg = -idx - 1
        idx_list = [idx, idx_nonctg, idx_neg]
        for tgt, idx, src, acc in product(tgt_gen, idx_list, src_gen, (True, False)):
            yield SampleInput(input=tgt, args=(idx, src, acc))

        # Scalar cases
        scalar_sizes = [(), (1,)]
        tgt_gen = (make_arg(size) for size in scalar_sizes)
        idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
        src_gen = (make_arg(size) for size in scalar_sizes)
        for tgt, idx, src, acc in product(tgt_gen, idx_gen, src_gen, (True, False)):
            yield SampleInput(input=tgt, args=(idx, src, acc))

        # Empty cases
        tgt_sizes = [(0,), (), (1,), (3, 2)]
        tgt_gen = (make_arg(size) for size in tgt_sizes)
        idx = make_idx((0,), high=1)
        src = make_arg((0,))
        for tgt, acc in product(tgt, (True, False)):
            yield SampleInput(input=tgt, args=(idx, src, acc))

    return list(gen_inputs())

def sample_inputs_take(op_info, device, dtype, requires_grad):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)

    S = 3

    def gen_inputs():
        # Generic inputs: take S elements out of S * S
        src_gen = (make_arg((S, S), noncontiguous=not ctg) for ctg in (True, False))
        idx = make_idx((S,), high=S * S)
        idx_nonctg = make_idx((S,), high=S * S, noncontiguous=True)
        idx_neg = -idx - 1
        idx_list = [idx, idx_nonctg, idx_neg]
        for src, idx in product(src_gen, idx_list):
            yield SampleInput(input=src, args=(idx,))

        # Scalar cases
        scalar_sizes = [(), (1,)]
        src_gen = (make_arg(size) for size in scalar_sizes)
        idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
        for src, idx in product(src_gen, idx_gen):
            yield SampleInput(input=src, args=(idx,))

        # Empty cases
        src_sizes = [(0,), (), (1,), (3, 2)]
        src_gen = (make_arg(size) for size in src_sizes)
        idx = make_idx((0,), high=1)
        for src in src_gen:
            yield SampleInput(input=src, args=(idx,))

    return list(gen_inputs())

def sample_movedim_moveaxis(op_info, device, dtype, requires_grad):
    return (
        SampleInput(
            make_tensor((4, 3, 2, 1), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=([0, 1, 2, 3], [3, 2, 1, 0])),
        SampleInput(
            make_tensor((4, 3, 2, 1), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=([0, -1, -2, -3], [-3, -2, -1, -0]))
    )


def sample_repeat_tile(op_info, device, dtype, requires_grad, **kwargs):
    rep_dims = ((), (0, ), (1, ), (0, 2), (1, 1), (2, 3), (2, 3, 2), (0, 2, 3), (2, 1, 1, 1),)
    shapes = ((), (0,), (2,), (3, 0), (3, 2), (3, 0, 1))

    if requires_grad:
        # Tests for variant_consistency_jit, grad, gradgrad
        # are slower. Use smaller bags of `rep_dims` and `shapes`
        # in this case.
        rep_dims = ((), (0, ), (0, 2), (1, 1), (2, 3), (1, 3, 2), (3, 1, 1))  # type: ignore[assignment]
        shapes = ((), (0,), (2,), (3, 2))  # type: ignore[assignment]

    tensors = [make_tensor(shape, device, dtype,
                           low=None, high=None,
                           requires_grad=requires_grad) for shape in shapes]

    samples = []
    for rep_dim, tensor in product(rep_dims, tensors):
        for t in (tensor, tensor.T):
            if op_info.name == 'repeat' and len(rep_dim) >= t.dim():
                # `torch.repeat` errors for `len(rep_dims) < t.dim()`,
                # so we filter such combinations.
                samples.append(SampleInput(t, args=(rep_dim,),))
            elif op_info.name == 'tile':
                samples.append(SampleInput(t, args=(rep_dim,),))

    return samples


def sample_inputs_narrow(op_info, device, dtype, requires_grad, **kwargs):
    shapes_and_args = (
        ((S, S, S), (1, 2, 2)),
        ((S, S, S), (-1, 2, 2)),
        ((S, S, S), (1, 0, 0)),
        ((S, S, S), (-1, 0, 0)),
    )

    def generator():
        for shape, args in shapes_and_args:
            tensor = make_tensor(shape, device, dtype, low=None, high=None,
                                 requires_grad=requires_grad)
            yield SampleInput(tensor, args=args)

    return list(generator())


def sample_unsqueeze(op_info, device, dtype, requires_grad, **kwargs):
    shapes_and_axes = [
        ((3, 4, 5), 0),
        ((3, 4, 5), 1),
        ((3, 4, 5), 3),
        ((3, 4, 5), -1),
        ((3, 4, 5), -3),
        ((), 0)
    ]

    samples = []
    for shape, axis in shapes_and_axes:
        tensor = make_tensor(shape, device, dtype, low=None, high=None,
                             requires_grad=requires_grad)
        samples.append(SampleInput(tensor, args=(axis,),))

    return samples


def sample_inputs_squeeze(op_info, device, dtype, requires_grad, **kwargs):
    shapes_and_args = (
        ((S, 1, S, 1), ()),
        ((1, 1, 1, 1), ()),
        ((S, 1, S, 1), (1,)),
        ((S, 1, S, 1), (-1,)),
        ((S, 1, S, 1), (2,)),
        ((S, 1, S, 1), (-2,)),
        ((), (0, )),
    )

    def generator():
        for shape, args in shapes_and_args:
            tensor = make_tensor(shape, device, dtype, low=None, high=None,
                                 requires_grad=requires_grad)

            yield SampleInput(tensor, args=args)

    return list(generator())


# TODO: reconcile with torch.linalg.det and torch.linalg.slogdet
# Creates matrices with a positive nonzero determinant
def sample_inputs_logdet(op_info, device, dtype, requires_grad, **kwargs):
    def make_nonzero_det(A, *, sign=1, min_singular_value=0.1, **kwargs):
        u, s, vh = torch.linalg.svd(A, full_matrices=False)
        s.clamp_(min=min_singular_value)
        A = (u * s.unsqueeze(-2)) @ vh
        det = A.det()
        if sign is not None:
            if A.dim() == 2:
                if (det < 0) ^ (sign < 0):
                    A[0, :].neg_()
            else:
                cond = ((det < 0) ^ (sign < 0)).nonzero()
                if cond.size(0) > 0:
                    for i in range(cond.size(0)):
                        A[list(cond[i])][0, :].neg_()
        return A

    samples = []

    # cases constructed using make_tensor()
    tensor_shapes = (
        (S, S),
        (1, 1),
        (3, 3, S, S),
        (3, 3, 1, 1)
    )

    for shape in tensor_shapes:
        t = make_tensor(shape, device=device, dtype=dtype)
        d = make_nonzero_det(t).requires_grad_(requires_grad)
        samples.append(SampleInput(d))

    # cases constructed using:
    #  1) make_symmetric_matrices
    #  2) make_symmetric_pd_matrices
    #  3) make_fullrank_matrices_with_distinct_singular_values
    symmetric_shapes = (
        (S, S),
        (3, S, S),
    )


    def _helper(constructor, *shape, **kwargs):
        t = constructor(*shape, device=device, dtype=dtype)
        d = make_nonzero_det(t, **kwargs).requires_grad_(requires_grad)
        samples.append(SampleInput(d))

    for shape in symmetric_shapes:
        _helper(make_symmetric_matrices, *shape)
        _helper(make_symmetric_pd_matrices, *shape)
        _helper(make_fullrank_matrices_with_distinct_singular_values, *shape, min_singular_value=0)

    return tuple(samples)

def np_unary_ufunc_integer_promotion_wrapper(fn):
    # Wrapper that passes PyTorch's default scalar
    #   type as an argument to the wrapped NumPy
    #   unary ufunc when given an integer input.
    #   This mimicks PyTorch's integer->floating point
    #   type promotion.
    #
    # This is necessary when NumPy promotes
    #   integer types to double, since PyTorch promotes
    #   integer types to the default scalar type.

    # Helper to determine if promotion is needed
    def is_integral(dtype):
        return dtype in [np.bool_, bool, np.uint8, np.int8, np.int16, np.int32, np.int64]

    @wraps(fn)
    def wrapped_fn(x):
        # As the default dtype can change, acquire it when function is called.
        # NOTE: Promotion in PyTorch is from integer types to the default dtype
        np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]

        if is_integral(x.dtype):
            return fn(x.astype(np_dtype))
        return fn(x)

    return wrapped_fn

def sample_inputs_spectral_ops(self, device, dtype, requires_grad=False, **kwargs):
    nd_tensor = make_tensor((S, S + 1, S + 2), device, dtype, low=None, high=None,
                            requires_grad=requires_grad)
    tensor = make_tensor((31,), device, dtype, low=None, high=None,
                         requires_grad=requires_grad)

    if self.ndimensional:
        return [
            SampleInput(nd_tensor, kwargs=dict(s=(3, 10), dim=(1, 2), norm='ortho')),
            SampleInput(nd_tensor, kwargs=dict(norm='ortho')),
            SampleInput(nd_tensor, kwargs=dict(s=(8,))),
            SampleInput(tensor),

            *(SampleInput(nd_tensor, kwargs=dict(dim=dim))
                for dim in [-1, -2, -3, (0, -1)]),
        ]
    else:
        return [
            SampleInput(nd_tensor, kwargs=dict(n=10, dim=1, norm='ortho')),
            SampleInput(nd_tensor, kwargs=dict(norm='ortho')),
            SampleInput(nd_tensor, kwargs=dict(n=7)),
            SampleInput(tensor),

            *(SampleInput(nd_tensor, kwargs=dict(dim=dim))
                for dim in [-1, -2, -3]),
        ]

# Metadata class for Fast Fourier Transforms in torch.fft.
class SpectralFuncInfo(OpInfo):
    """Operator information for torch.fft transforms. """

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref=None,  # Reference implementation (probably in np.fft namespace)
                 dtypes=floating_and_complex_types(),
                 ndimensional: bool,  # Whether dim argument can be a tuple
                 sample_inputs_func=sample_inputs_spectral_ops,
                 decorators=None,
                 **kwargs):
        decorators = list(decorators) if decorators is not None else []
        decorators += [
            skipCPUIfNoMkl,
            skipCUDAIfRocm,
            # gradgrad is quite slow
            DecorateInfo(slowTest, 'TestGradients', 'test_fn_gradgrad'),
        ]

        super().__init__(name=name,
                         dtypes=dtypes,
                         decorators=decorators,
                         sample_inputs_func=sample_inputs_func,
                         **kwargs)
        self.ref = ref if ref is not None else _getattr_qual(np, name)
        self.ndimensional = ndimensional


class ShapeFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for Shape manipulating operations like tile and roll"""
    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref,  # a reference function
                 dtypes=floating_types(),
                 dtypesIfCPU=None,
                 dtypesIfCUDA=None,
                 dtypesIfROCM=None,
                 sample_inputs_func=None,
                 **kwargs):
        super(ShapeFuncInfo, self).__init__(name,
                                            dtypes=dtypes,
                                            dtypesIfCPU=dtypesIfCPU,
                                            dtypesIfCUDA=dtypesIfCUDA,
                                            dtypesIfROCM=dtypesIfROCM,
                                            sample_inputs_func=sample_inputs_func,
                                            **kwargs)
        self.ref = ref

def sample_inputs_foreach(self, device, dtype, N, *, noncontiguous=False):
    tensors = [make_tensor((N - i, N - i), device, dtype, noncontiguous=noncontiguous) for i in range(N)]
    return tensors


def get_foreach_method_names(name):
    # get torch inplace reference function
    op_name = "_foreach_" + name
    inplace_op_name = "_foreach_" + name + "_"

    op = getattr(torch, op_name, None)
    inplace_op = getattr(torch, inplace_op_name, None)

    ref = getattr(torch, name, None)
    ref_inplace = getattr(torch.Tensor, name + "_", None)
    return op, inplace_op, ref, ref_inplace

class ForeachFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for foreach functions"""
    def __init__(self,
                 name,
                 dtypes=floating_and_complex_types(),
                 dtypesIfCPU=all_types_and_complex(),
                 dtypesIfCUDA=floating_and_complex_types_and(torch.half),
                 dtypesIfROCM=None,
                 safe_casts_outputs=True,
                 sample_inputs_func=sample_inputs_foreach,
                 **kwargs):
        super().__init__(
            "_foreach_" + name,
            dtypes=dtypes,
            dtypesIfCPU=dtypesIfCPU,
            dtypesIfCUDA=dtypesIfCUDA,
            dtypesIfROCM=dtypesIfROCM,
            safe_casts_outputs=safe_casts_outputs,
            sample_inputs_func=sample_inputs_func,
            **kwargs
        )

        foreach_method, foreach_method_inplace, torch_ref_method, torch_ref_inplace = get_foreach_method_names(name)
        self.method_variant = foreach_method
        self.inplace_variant = foreach_method_inplace
        self.ref = torch_ref_method
        self.ref_inplace = torch_ref_inplace


def sample_inputs_linalg_cholesky_inverse(op_info, device, dtype, requires_grad=False):
    # Generate Cholesky factors of positive-definite (non-singular) Hermitian (symmetric) matrices
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix
    inputs = (
        torch.zeros(0, 0, dtype=dtype, device=device),  # 0x0 matrix
        torch.zeros(0, 2, 2, dtype=dtype, device=device),  # zero batch of matrices
        random_hermitian_pd_matrix(S, dtype=dtype, device=device),  # single matrix
        random_hermitian_pd_matrix(S, 2, dtype=dtype, device=device),  # batch of matrices
    )
    test_cases = (torch.linalg.cholesky(a) for a in inputs)
    out = []
    for a in test_cases:
        a.requires_grad = requires_grad
        out.append(SampleInput(a))
        out.append(SampleInput(a, kwargs=dict(upper=True)))
    return out

def sample_inputs_linalg_lstsq(op_info, device, dtype, requires_grad=False, **kwargs):
    from torch.testing._internal.common_utils import random_well_conditioned_matrix
    out = []
    for batch in ((), (3,), (3, 3)):
        shape = batch + (3, 3)
        # NOTE: inputs are not marked with `requires_grad` since
        # linalg_lstsq is not differentiable
        a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
        b = make_tensor(shape, device, dtype, low=None, high=None)
        out.append(SampleInput(a, args=(b,)))
    return out

def sample_inputs_householder_product(op_info, device, dtype, requires_grad, **kwargs):
    """
    This function generates input for torch.linalg.householder_product (torch.orgqr).
    The first argument should be a square matrix or batch of square matrices, the second argument is a vector or batch of vectors.
    Empty, square, rectangular, batched square and batched rectangular input is generated.
    """
    # Each column of the matrix is getting multiplied many times leading to very large values for
    # the Jacobian matrix entries and making the finite-difference result of grad check less accurate.
    # That's why gradcheck with the default range [-9, 9] fails and [-2, 2] is used here.
    samples = (
        SampleInput(make_tensor((S, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        SampleInput(make_tensor((S + 1, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        SampleInput(make_tensor((2, 1, S, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((2, 1, S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        SampleInput(make_tensor((2, 1, S + 1, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((2, 1, S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        SampleInput(make_tensor((0, 0), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(make_tensor((0,), device, dtype, low=None, high=None, requires_grad=requires_grad),)),

        SampleInput(make_tensor((S, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((0,), device, dtype, low=None, high=None, requires_grad=requires_grad),)),
    )

    return samples

def sample_inputs_ormqr(op_info, device, dtype, requires_grad):
    # create a helper function wrapping `make_tensor`
    make_input = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    def gen_inputs():
        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        tf = [True, False]
        for batch, (m, n), left, transpose in product(batches, product(ns, ns), tf, tf):
            reflectors = make_input((*batch, m, n))
            tau = make_input((*batch, min(m, n)))
            other_matrix_shape = (m, n) if left else (n, m)
            other = make_input((*batch, *other_matrix_shape))
            kwargs = {"left": left, "transpose": transpose}
            yield SampleInput(reflectors, args=(tau, other,), kwargs=kwargs)

    return tuple(gen_inputs())

def sample_inputs_linalg_cholesky(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates always positive-definite input for torch.linalg.cholesky using
    random_hermitian_pd_matrix.
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix

    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 0]
    out = []
    for batch, n in product(batches, ns):
        a = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
        a.requires_grad = requires_grad
        out.append(SampleInput(a))
    return out

def sample_inputs_symeig(op_info, device, dtype, requires_grad=False):
    out = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)

    for o in out:
        o.kwargs = {"upper": bool(np.random.choice([True, False])),
                    "eigenvectors": True}
        # A gauge-invariant function
        o.output_process_fn_grad = lambda output: (output[0], abs(output[1]))
    return out

def sample_inputs_linalg_eig(op_info, device, dtype, requires_grad=False):
    """
    This function generates input for torch.linalg.eigh with UPLO="U" or "L" keyword argument.
    """
    def out_fn(output):
        return output[0], abs(output[1])

    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.output_process_fn_grad = out_fn

    return samples

def sample_inputs_linalg_eigh(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.eigh/eigvalsh with UPLO="U" or "L" keyword argument.
    """
    def out_fn(output):
        if isinstance(output, tuple):
            # eigh function
            return output[0], abs(output[1])
        else:
            # eigvalsh function
            return output

    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.kwargs = {"UPLO": np.random.choice(["L", "U"])}
        sample.output_process_fn_grad = out_fn

    return samples


def sample_inputs_linalg_slogdet(op_info, device, dtype, requires_grad=False):
    def out_fn(output):
        return output[1]

    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.output_process_fn_grad = out_fn

    return samples


def sample_inputs_linalg_pinv_hermitian(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.pinv with hermitian=True keyword argument.
    """
    out = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad, **kwargs)
    for o in out:
        o.kwargs = {"hermitian": True}
    return out

def sample_inputs_linalg_solve(op_info, device, dtype, requires_grad=False, vector_rhs_allowed=True, **kwargs):
    """
    This function generates always solvable input for torch.linalg.solve
    Using random_fullrank_matrix_distinct_singular_value gives a non-singular (=invertible, =solvable) matrices 'a'.
    The first input to torch.linalg.solve is generated as the itertools.product of 'batches' and 'ns'.
    The second input is generated as the product of 'batches', 'ns' and 'nrhs'.
    In total this function generates 18 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices.
    'ns' gives 0x0 and 5x5 matrices.
    and 'nrhs' controls the number of vectors to solve for:
        () - using 1 as the number of vectors implicitly
        (1,) - same as () but explicit
        (3,) - solve for 3 vectors.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    'vector_rhs_allowed' controls whether to include nrhs = () to the list of SampleInputs.
    torch.solve / triangular_solve / cholesky_solve (opposed to torch.linalg.solve) do not allow
    1D tensors (vectors) as the right-hand-side.
    Once torch.solve / triangular_solve / cholesky_solve and its testing are removed,
    'vector_rhs_allowed' may be removed here as well.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    batches = [(), (0, ), (2, )]
    ns = [5, 0]
    if vector_rhs_allowed:
        nrhs = [(), (1,), (3,)]
    else:
        nrhs = [(1,), (3,)]
    out = []
    for n, batch, rhs in product(ns, batches, nrhs):
        a = random_fullrank_matrix_distinct_singular_value(n, *batch, dtype=dtype, device=device)
        a.requires_grad = requires_grad
        b = torch.randn(*batch, n, *rhs, dtype=dtype, device=device)
        b.requires_grad = requires_grad
        out.append(SampleInput(a, args=(b,)))
    return out


def sample_inputs_legacy_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates always solvable input for legacy solve functions
    (the ones that are not in torch.linalg module).
    The difference from sample_inputs_linalg_solve is that here the right-hand-side of A x = b equation
    should have b.ndim >= 2, vectors are not allowed.
    Also the arguments order is swapped.
    """
    out = sample_inputs_linalg_solve(
        op_info, device, dtype, requires_grad=requires_grad, vector_rhs_allowed=False
    )

    # Reverses tensor order
    for sample in out:
        sample.input, sample.args = sample.args[0], (sample.input,)

    return out


def sample_inputs_lu(op_info, device, dtype, requires_grad=False, **kwargs):
    # not needed once OpInfo tests support Iterables
    def generate_samples():
        batch_shapes = ((), (3,), (3, 3))
        for batch_shape, get_infos in product(batch_shapes, (True, False)):
            shape = batch_shape + (S, S)
            input = make_tensor(shape, device, dtype, requires_grad=requires_grad, low=None, high=None)
            yield SampleInput(input, args=(True, get_infos))

    return list(generate_samples())


def sample_inputs_lu_unpack(op_info, device, dtype, requires_grad=False, **kwargs):
    # not needed once OpInfo tests support Iterables
    def generate_samples():
        for lu_sample in sample_inputs_lu(op_info, device, dtype, requires_grad, **kwargs):
            lu_data, pivots = lu_sample.input.lu()
            yield SampleInput(lu_data, args=(pivots,))

            # generate rectangular inputs
            lu_data_shape = lu_data.shape
            batch_shape = lu_data_shape[:-2]
            n = lu_data_shape[-2]

            for shape_inc in ((1, 0), (0, 1)):
                lu_data, pivots = make_tensor(
                    batch_shape + (n + shape_inc[0], n + shape_inc[1]),
                    device, dtype,
                    requires_grad=False,
                    low=None, high=None
                ).lu()
                lu_data.requires_grad_(requires_grad)
                yield SampleInput(lu_data, args=(pivots,))

    return list(generate_samples())


def sample_inputs_roll(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    args = ((0, 0), (1, 2), (0, 2), (2, 0), (-1, 0), (10000, 1), (2,), ((1, 2, -1), (0, 1, 2)))

    def generator():
        for arg in args:
            yield SampleInput(make_arg((S, S, S)), args=arg)

    return list(generator())


def sample_inputs_rot90(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    args = ((1, (0, 1),),
            (1, (1, 2),),
            (1, (1, -1),),
            ())

    def generator():
        for arg in args:
            yield SampleInput(make_arg((S, S, S)), args=arg)

    return list(generator())


def sample_inputs_std_var(op_info, device, dtype, requires_grad, **kwargs):
    tensor_nd = make_tensor((S, S, S), device=device, dtype=dtype,
                            low=None, high=None, requires_grad=requires_grad)
    tensor_1d = make_tensor((S,), device=device, dtype=dtype,
                            low=None, high=None, requires_grad=requires_grad)

    return [
        SampleInput(tensor_nd),
        SampleInput(tensor_nd, kwargs=dict(dim=1)),
        SampleInput(tensor_nd, kwargs=dict(dim=1, unbiased=True, keepdim=True)),
        SampleInput(tensor_1d, kwargs=dict(dim=0, unbiased=True, keepdim=True)),
        SampleInput(tensor_1d, kwargs=dict(dim=0, unbiased=False, keepdim=False)),

        SampleInput(tensor_nd, kwargs=dict(dim=(1,), correction=S // 2)),
        SampleInput(tensor_nd, kwargs=dict(dim=None, correction=0, keepdim=True)),
    ]


def _sample_inputs_svd(op_info, device, dtype, requires_grad=False, is_linalg_svd=False):
    """
    This function generates input for torch.svd with distinct singular values so that autograd is always stable.
    Matrices of different size:
        square matrix - S x S size
        tall marix - S x (S-2)
        wide matrix - (S-2) x S
    and batched variants of above are generated.
    Each SampleInput has a function 'output_process_fn_grad' attached to it that is applied on the output of torch.svd
    It is needed for autograd checks, because backward of svd doesn't work for an arbitrary loss function.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    # svd and linalg.svd returns V and V.conj().T, respectively. So we need to slice
    # along different dimensions when needed (this is used by
    # test_cases2:wide_all and wide_all_batched below)
    if is_linalg_svd:
        def slice_V(v):
            return v[..., :(S - 2), :]

        def uv_loss(usv):
            u00 = usv[0][0, 0]
            v00_conj = usv[2][0, 0]
            return u00 * v00_conj
    else:
        def slice_V(v):
            return v[..., :, :(S - 2)]

        def uv_loss(usv):
            u00 = usv[0][0, 0]
            v00_conj = usv[2][0, 0].conj()
            return u00 * v00_conj

    test_cases1 = (  # some=True (default)
        # loss functions for complex-valued svd have to be "gauge invariant",
        # i.e. loss functions shouldn't change when sigh of the singular vectors change.
        # the simplest choice to satisfy this requirement is to apply 'abs'.
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            lambda usv: usv[1]),  # 'check_grad_s'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            lambda usv: abs(usv[0])),  # 'check_grad_u'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            lambda usv: abs(usv[2])),  # 'check_grad_v'
        # this test is important as it checks the additional term that is non-zero only for complex-valued inputs
        # and when the loss function depends both on 'u' and 'v'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            uv_loss),  # 'check_grad_uv'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device)[:(S - 2)],
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2][..., :, :(S - 2)]))),  # 'wide'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device)[:, :(S - 2)],
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2]))),  # 'tall'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device),
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2]))),  # 'batched'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device)[..., :(S - 2), :],
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2]))),  # 'wide_batched'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device)[..., :, :(S - 2)],
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2]))),  # 'tall_batched'
    )
    test_cases2 = (  # some=False
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device)[:(S - 2)],
            lambda usv: (abs(usv[0]), usv[1], abs(slice_V(usv[2])))),  # 'wide_all'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device)[:, :(S - 2)],
            lambda usv: (abs(usv[0][:, :(S - 2)]), usv[1], abs(usv[2]))),  # 'tall_all'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device)[..., :(S - 2), :],
            lambda usv: (abs(usv[0]), usv[1], abs(slice_V(usv[2])))),  # 'wide_all_batched'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device)[..., :, :(S - 2)],
            lambda usv: (abs(usv[0][..., :, :(S - 2)]), usv[1], abs(usv[2]))),  # 'tall_all_batched'
    )

    out = []
    for a, out_fn in test_cases1:
        a.requires_grad = requires_grad
        if is_linalg_svd:
            kwargs = {'full_matrices': False}
        else:
            kwargs = {'some': True}
        out.append(SampleInput(a, kwargs=kwargs, output_process_fn_grad=out_fn))

    for a, out_fn in test_cases2:
        a.requires_grad = requires_grad
        if is_linalg_svd:
            kwargs = {'full_matrices': True}
        else:
            kwargs = {'some': False}
        out.append(SampleInput(a, kwargs=kwargs, output_process_fn_grad=out_fn))

    return out


def sample_inputs_permute(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = [((1, 2, 3, 4), (0, 2, 3, 1)),
             ((1, 2, 3, 4), (0, -2, -1, 1)),
             ((), ()),
             ((1, 2, 3, 4), (2, 1, 3, 0))]

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=(args,))

    return list(generator())


# Based on erstwhile method_tests tests & some tensor_op_tests for pow
def sample_inputs_pow(op_info, device, dtype, requires_grad, **kwargs):
    samples = []

    if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        test_cases = (
            ((2, 2), 0, 5, 1e-3, requires_grad, (2, 2), 0, 1, 0.1, requires_grad, False),
            ((2, 2), 0, 5, 1e-3, requires_grad, (1,), 0, 1, 0.1, requires_grad, False),
            ((), 1e-3, 1e-3 + 1, 0, requires_grad, (), 0.1, 1.1, 0, False, False),
            ((2, 2), 0, 5, 1e-3, requires_grad, (), 0.1, 1.1, 1, False, False),
        )
        tests_require_resizing = (
            ((1,), 0, 5, 1e-3, requires_grad, (2, 2), 0, 1, 0.1, requires_grad, requires_grad),
            ((2, 1, 2), 0, 5, 1e-3, requires_grad, (1, 2, 1), 0, 1, 0.1, requires_grad, requires_grad),
            ((), 1e-3, 1e-3 + 1, 0, requires_grad, (1, S, 1), 0, 1, 0.1, requires_grad, requires_grad),
        )
        cases = test_cases + tests_require_resizing
        samples = list(SampleInput(make_tensor(shape_b, low=low_b, high=high_b,
                                               requires_grad=b_grad, device=device,
                                               dtype=dtype) + additive_b,
                                   args=(make_tensor(shape_e, low=low_e, high=high_e,
                                                     requires_grad=e_grad, device=device,
                                                     dtype=dtype) + additive_e,),
                                   broadcasts_input=broadcasts_input)
                       for shape_b, low_b, high_b, additive_b, b_grad, shape_e, low_e,
                       high_e, additive_e, e_grad, broadcasts_input in cases)
        tensor_scalar_inputs = (
            ((2, 2), 0, 5, 1e-3, requires_grad, (3.14,)),
            ((), 1e-3, 1e-3 + 1, 0, requires_grad, (3.14,))
        )
        more_samples = list(SampleInput(make_tensor(shape, dtype=dtype, device=device,
                                                    high=high, low=low,
                                                    requires_grad=b_grad) + additive,
                                        args=exp)
                            for shape, low, high, additive, b_grad, exp in tensor_scalar_inputs)
        samples = [*samples, *more_samples]
    elif dtype in [torch.complex64, torch.complex128]:
        args_tuple = (
            ((2, 2), 0, 5, requires_grad, (3.14,)),
            ((), 0, 1, requires_grad, (3.14,)),
            ((), 0, 1, requires_grad, (3.14j,))
        )
        samples = list(SampleInput(make_tensor(shape, dtype=dtype, device=device,
                                               high=high, low=low,
                                               requires_grad=b_grad) + 1e-3 * (1 + 1j),
                                   args=arg)
                       for shape, low, high, b_grad, arg in args_tuple)
    elif dtype == torch.bool:
        arg_tuple = (0, 1, 1., 2.3)
        samples = list(SampleInput(make_tensor((2, 2), device=device, dtype=dtype,
                                               requires_grad=requires_grad),
                                   args=(arg,))
                       for arg in arg_tuple)
        dtypes_list = [torch.float64, torch.float32, torch.int64, torch.int32]
        more_samples = list(SampleInput(make_tensor((2, 2), device, dtype=torch.bool,
                                                    requires_grad=requires_grad),
                                        args=(make_tensor((2, 2), device, dtype=dtype,
                                                          requires_grad=requires_grad),))
                            for dtype in dtypes_list)
        samples = [*samples, *more_samples]
        samples.append(SampleInput(make_tensor((2, 2, 2), device, dtype=torch.bool,
                                               requires_grad=requires_grad),
                                   args=(make_tensor((2, 1), device, dtype=torch.float64,
                                                     requires_grad=requires_grad),)))
    else:
        exp_tuple = (1, 2, 3)
        samples = list(SampleInput(make_tensor((2, 2), device, dtype,
                                               requires_grad=requires_grad),
                                   args=(arg,))
                       for arg in exp_tuple)
        samples.append(SampleInput(make_tensor((2, 2), device, dtype,
                                               requires_grad=requires_grad),
                                   args=(make_tensor((2, 2), device, dtype,
                                                     requires_grad=requires_grad),)))
    return tuple(samples)

def sample_inputs_svd(op_info, device, dtype, requires_grad=False, **kwargs):
    return _sample_inputs_svd(op_info, device, dtype, requires_grad, is_linalg_svd=False)

def sample_inputs_linalg_svd(op_info, device, dtype, requires_grad=False, **kwargs):
    return _sample_inputs_svd(op_info, device, dtype, requires_grad, is_linalg_svd=True)

def sample_inputs_linalg_svdvals(op_info, device, dtype, requires_grad=False, **kwargs):
    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 2, 0]
    samples = []
    for batch, (m, n) in product(batches, product(ns, ns)):
        a = make_tensor((*batch, m, n), device, dtype, low=None, high=None, requires_grad=requires_grad)
        samples.append(SampleInput(a))
    return samples

def sample_inputs_hardshrink_hardtanh(op_info, device, dtype, requires_grad=False, **kwargs):
    N = 10
    tensors = [SampleInput(make_tensor((N, N), device=device, dtype=dtype,
               requires_grad=requires_grad)) for _ in range(1, N)]
    return tensors

def sample_inputs_eig(op_info, device, dtype, requires_grad=False, **kwargs):
    eigvecs = make_tensor((S, S), device=device, dtype=dtype,
                          low=None, high=None)
    eigvals = make_tensor((S,), device=device, dtype=dtype,
                          low=None, high=None)
    # we produce only diagonazible inputs which do not have
    # complex eigenvalues for real inputs, as there is no
    # backward implementation for real inputs with complex
    # eigenvalues yet.
    input = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.inverse()
    input.requires_grad_(requires_grad)

    def process_output(eigpair):
        eigvals, eigvecs = eigpair
        if dtype.is_complex:
            # eig produces eigenvectors which are normalized to 1 norm.
            # Note that if v is an eigenvector, so is v * e^{i \phi},
            # and |v| = |v * e^{i \phi}| = 1.
            # This, however, makes the eigenvector backward computation process
            # rather unstable unless the objective function is gauge-invariant,
            # that is if f(z) == f(|z|), for example.
            # Hence for complex inputs we ignore the phases and return only
            # the absolute values.
            return eigvals, eigvecs.abs()
        else:
            return eigvals, eigvecs

    return [
        SampleInput(
            input,
            kwargs=dict(eigenvectors=True),
            output_process_fn_grad=process_output
        ),
    ]


def sample_inputs_einsum(op_info, device, dtype, requires_grad=False, **kwargs):
    x = make_tensor((3,), device, dtype, requires_grad=requires_grad)
    y = make_tensor((4,), device, dtype, requires_grad=requires_grad)
    A = make_tensor((2, 3,), device, dtype, requires_grad=requires_grad, noncontiguous=True)
    B = make_tensor((1, 3,), device, dtype, requires_grad=requires_grad)
    C = make_tensor((1, 2, 3,), device, dtype, requires_grad=requires_grad)
    D = make_tensor((1, 3, 4,), device, dtype, requires_grad=requires_grad, noncontiguous=True)
    E = make_tensor((4, 4,), device, dtype, requires_grad=requires_grad)
    H = make_tensor((3, 3,), device, dtype, requires_grad=requires_grad, noncontiguous=True)
    I = make_tensor((1, 3, 1,), device, dtype, requires_grad=requires_grad)

    inputs = []

    # Vector operations
    inputs.append(SampleInput([x], args=('i->',)))                      # sum
    inputs.append(SampleInput([x, y], args=('i,j->ij',)))               # outer

    # Matrix operations
    inputs.append(SampleInput([A], args=("ij->i",)))                    # col sum
    inputs.append(SampleInput([A, B], args=("ij,kj->ik",)))             # matmul
    inputs.append(SampleInput([A, E], args=("ij,Ab->ijAb",)))           # matrix outer product

    # Tensor operations
    inputs.append(SampleInput([C, D], args=("aij,ajk->aik",)))          # batch matmul
    inputs.append(SampleInput([D, E], args=("aij,jk->aik",)))           # tensor matrix contraction
    inputs.append(SampleInput([C, B], args=("ijk,ik->j",)))             # non contiguous

    # Test diagonals
    inputs.append(SampleInput([I], args=('iji->j',)))                   # non-contiguous trace

    # Test ellipsis
    inputs.append(SampleInput([H], args=("i...->...",)))
    inputs.append(SampleInput([C, x], args=('...ik, ...j -> ij',)))

    return inputs


def sample_inputs_linalg_qr(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.qr
    The input is generated as the itertools.product of 'batches' and 'ns'.
    """
    batches = [(), (0,), (2, ), (1, 1)]
    ns = [5, 2, 0]
    out = []
    for batch, (m, n) in product(batches, product(ns, ns)):
        a = torch.randn(*batch, m, n, dtype=dtype, device=device, requires_grad=requires_grad)
        out.append(SampleInput(a))
    return out

def sample_inputs_geqrf(op_info, device, dtype, requires_grad=False):
    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 2, 0]
    samples = []
    for batch, (m, n) in product(batches, product(ns, ns)):
        # TODO: CUDA path doesn't work with batched or empty inputs
        if torch.device(device).type == 'cuda' and (batch != () or m == 0 or n == 0):
            continue
        a = make_tensor((*batch, m, n), device, dtype, low=None, high=None, requires_grad=requires_grad)
        samples.append(SampleInput(a))
    return samples

def sample_inputs_flip(op_info, device, dtype, requires_grad):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((S, M, S), (S, 0, M))
    all_dims = ((0, 1, 2), (0,), (0, 2), (-1,), ())

    def gen_samples():
        for size, dims in product(sizes, all_dims):
            yield SampleInput(make_arg(size), kwargs={"dims": dims})

    return list(gen_samples())

def sample_inputs_fliplr_flipud(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((S, 0, M), device, dtype, low=None, high=None, requires_grad=requires_grad)
    )
    return [SampleInput(tensor) for tensor in tensors]

def sample_inputs_fmod_remainder(op_info, device, dtype, requires_grad, *, autodiffed=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    if autodiffed:
        samples = (
            ((S, S, S), 1.5, False),
            ((), 1.5, False),
        )
    else:
        cases = (
            ((S, S, S), (), False),
            ((S, S, S), (S, S, S), False),
            ((S, S, S), (S,), False),
        )

        # Sample inputs with scalars as torch tensors
        cases_with_tensor_scalar = (
            ((), torch.tensor(1, dtype=dtype, device=device, requires_grad=False), False),
        )

        # Sample inputs with broadcasting
        cases_with_broadcasting = (
            ((S,), (S, S, S), True),
            ((S, 1, S), (S, S, S), True),
            ((), (S, S, S), True),
        )

        samples = cases + cases_with_tensor_scalar + cases_with_broadcasting  # type: ignore[assignment]

    def generator():
        for shape, arg_other, broadcasts_input in samples:
            if isinstance(arg_other, tuple):
                arg = make_arg(arg_other, requires_grad=False, exclude_zero=True)
            else:
                # shape_other is scalar or torch.tensor
                arg = arg_other
            yield(SampleInput(make_arg(shape), args=(arg,), broadcasts_input=broadcasts_input))

    return list(generator())

# TODO: clamp shares tensors among its sample inputs --- we should prohibit this!
def sample_inputs_clamp(op_info, device, dtype, requires_grad, **kwargs):
    x = make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    lb = make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    ub = make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad)

    def detach(tensor):
        return tensor.clone().detach_().requires_grad_(requires_grad)

    return [
        SampleInput(detach(x), args=(lb, ub)),
        SampleInput(detach(x), args=(detach(lb[0]), detach(ub[0]))),
        SampleInput(detach(x), args=(detach(lb[:, :1]),)),
    ]

def sample_inputs_clamp_scalar(op_info, device, dtype, requires_grad):
    tensors = (
        make_tensor((2, 3, 2), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((2, 0, 3), device, dtype, low=None, high=None, requires_grad=requires_grad),
    )
    if dtype is torch.uint8:
        min_max_vals = ((2, 5), (3, 7))
    else:
        min_max_vals = ((0, 1), (-1, 1))
    output = [SampleInput(tensor, args=vals) for tensor, vals in product(tensors, min_max_vals)]
    output += [SampleInput(tensors[0], args=(0.5, None)), SampleInput(tensors[0], args=(None, 0.5))]
    empty_tensor = make_tensor((), device=device, dtype=dtype, low=None, high=None, requires_grad=requires_grad)
    output += [SampleInput(empty_tensor, args=(0.0, 1.0)), ]
    return output

def sample_kwargs_clamp_scalar(device, dtype, input):
    if dtype is torch.uint8:
        min_val, max_val = (random.randint(1, 3), random.randint(4, 8))
    elif dtype.is_floating_point:
        min_val, max_val = (random.uniform(-8, 0), random.uniform(1, 8))  # type: ignore[assignment]
    else:
        min_val, max_val = (random.randint(-8, 0), random.randint(1, 8))
    return {'min': min_val, 'max': max_val}, {'a_min': min_val, 'a_max': max_val}

def sample_inputs_cross(op_info, device, dtype, requires_grad, **kwargs):
    sample0 = SampleInput(make_tensor((S, 3), device=device, dtype=dtype, requires_grad=requires_grad),
                          args=(make_tensor((S, 3), device=device, dtype=dtype, requires_grad=requires_grad),))
    sample1 = SampleInput(make_tensor((S, 3, S), device=device, dtype=dtype, requires_grad=requires_grad),
                          args=(make_tensor((S, 3, S), device=device, dtype=dtype, requires_grad=requires_grad),),
                          kwargs={'dim': 1})

    return (sample0, sample1)

def sample_inputs_cumprod(op_info, device, dtype, requires_grad, **kwargs):
    def make_arg(shape):
        # shrink values to be in the interval [-1, +1] for better precision in gradgradcheck
        return make_tensor(shape, device, dtype, low=-1, high=+1, requires_grad=requires_grad)

    def prod_zeros(dim_select):
        assert len(dim_select) == 2
        result = make_arg(3 * (S,))
        with torch.no_grad():
            result.narrow(dim_select[0], 0, 1).narrow(dim_select[1], 1, 1).zero_()
            result.narrow(dim_select[0], 2, 1).narrow(dim_select[1], 3, 1).zero_()
            result.narrow(dim_select[0], 4, 1).narrow(dim_select[1], 3, 1).zero_()
        return result

    # will not be needed once OpInfo tests suport Iterables
    def sample_generator():
        for dim in range(3):
            yield SampleInput(make_arg((S, S, S)), args=(dim,))
        # Scalar tensors and empty tensor
        for size in [(), (1,), (0,)]:
            yield SampleInput(make_arg(size), args=(0,))

        yield SampleInput(prod_zeros([0, 1]), args=(1,))
        yield SampleInput(prod_zeros([0, 2]), args=(1,))
        yield SampleInput(prod_zeros([1, 2]), args=(1,))

        # test dtype kwarg
        yield SampleInput(prod_zeros([1, 2]), args=(1,), kwargs={'dtype': dtype})

    return list(sample_generator())

def sample_inputs_view_as_complex(op_info, device, dtype, requires_grad, **kwargs):
    return [SampleInput(make_tensor((S, 2), device, dtype, requires_grad=requires_grad),)]

def sample_inputs_view_as_real(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((), device, dtype, requires_grad=requires_grad)
    )
    return [SampleInput(tensor) for tensor in tensors]

def sample_inputs_copysign(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor(*shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    cases = [
        # no broadcast
        ((S, S, S), (S, S, S), False),
        # broadcast rhs
        ((S, S, S), (S, S), False),

        # scalar
        ((S, S), 3.14, False),
        # scalar positive zero
        ((S, S), 0.0, False),
        # scalar negative zero
        ((S, S), -0.0, False),
    ]

    # broadcast lhs
    cases.append(((S, S), (S, S, S), True))
    # broadcast all
    cases.append(((S, 1, S), (M, S), True))

    def generator():
        for input_shape, arg_val, broadcasts_input in cases:
            if isinstance(arg_val, tuple):
                arg = _make_tensor(*arg_val)
            else:
                # arg_val is scalar
                arg = arg_val

            yield SampleInput(_make_tensor(*input_shape), args=(arg, ), broadcasts_input=broadcasts_input)

    return list(generator())

def sample_inputs_prod(op_info, device, dtype, requires_grad):
    def make_arg(shape):
        # shrink values to be in the interval [-1, +1] for better precision in gradgradcheck
        return make_tensor(shape, device, dtype, low=-1, high=+1, requires_grad=requires_grad)

    def prod_single_zero():
        result = make_arg(2 * (S,))
        with torch.no_grad():
            result[0, 1] = 0
        return result

    # will not be needed once OpInfo tests support Iterables
    def sample_generator():
        for sample in sample_inputs_cumprod(op_info, device, dtype, requires_grad):
            yield SampleInput(sample.input)  # only Tensor, ignore other inputs
            yield sample
            sample.kwargs['keepdim'] = True
            yield sample
        yield SampleInput(prod_single_zero())
        yield SampleInput(make_arg((3, 3, 3)), args=(1,))
        yield SampleInput(make_arg((3, 3, 3)), args=(1,), kwargs={'keepdim': True})

        # test zero scalar tensor
        zero = make_arg(())
        with torch.no_grad():
            zero.zero_()
        yield SampleInput(zero)
        yield SampleInput(zero, args=(0,))
        yield SampleInput(zero, args=(0,), kwargs={'keepdim': True})

    return list(sample_generator())

def sample_inputs_diag(op_info, device, dtype, requires_grad, **kwargs):
    vec_sample = SampleInput(make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad))

    tensors = (
        make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((3, 5), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((5, 3), device, dtype, low=None, high=None, requires_grad=requires_grad),
    )

    args = ((), (2,), (-2,), (1,), (2,))

    samples = []
    for tensor, arg in product(tensors, args):
        samples.append(SampleInput(tensor, args=arg))

    return samples + [vec_sample]

def sample_inputs_diagonal_diag_embed(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    # Shapes for 2D Tensors
    shapes_2d = ((M, M), (3, 5), (5, 3))

    # Shapes for 3D Tensors
    shapes_3d = ((M, M, M),)

    args_2d = ((), (2,), (-2,), (1,))
    args_3d = ((1, 1, 2), (2, 0, 1), (-2, 0, 1))

    def generator():
        for shape, arg in chain(product(shapes_2d, args_2d), product(shapes_3d, args_3d)):
            yield SampleInput(make_arg(shape), args=arg)

    return list(generator())


def sample_inputs_to_sparse(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return (SampleInput(make_arg((S, S)), args=(), output_process_fn_grad=lambda x: x.to_dense()),
            SampleInput(make_arg((S, S)), args=(1,), output_process_fn_grad=lambda x: x.to_dense()),)


def sample_inputs_log_softmax(op_info, device, dtype, requires_grad, with_dtype=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    if with_dtype:
        cases = (((S, S, S), (1, torch.float64)),)
    else:
        cases = (((S, S, S), (1,)),)  # type:ignore[assignment]

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_logit(op_info, device, dtype, requires_grad, **kwargs):
    low, high = op_info.domain

    # Note: Operator is very sensitive at points near the
    # start and end of domain and leads to NaN for float16
    # if domain_eps is 1e-5.
    domain_eps = op_info._domain_eps if dtype != torch.float16 else 3e-2

    low = low + domain_eps
    high = high - domain_eps

    samples = (
        SampleInput(make_tensor((S, S, S), device, dtype, low=low, high=high, requires_grad=requires_grad)),
        SampleInput(make_tensor((S, S, S), device, dtype, low=low,
                                high=high, requires_grad=requires_grad), args=(0.2,)),
        SampleInput(make_tensor((), device, dtype, low=low, high=high, requires_grad=requires_grad)),
        SampleInput(make_tensor((), device, dtype, low=low,
                                high=high, requires_grad=requires_grad), args=(0.2,)),
    )

    return samples

def sample_inputs_floor_divide(op_info, device, dtype, requires_grad, **kwargs):
    lhs = make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    rhs = make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    # Avoid integer divide by 0
    if not (dtype.is_floating_point or dtype.is_complex):
        rhs[rhs == 0] = 1

    return [
        SampleInput(lhs, args=(rhs,)),
        SampleInput(lhs, args=(rhs[0],)),
        SampleInput(lhs, args=(3.14,)),
    ]

def sample_inputs_isin(op_info, device, dtype, requires_grad):
    element = make_tensor((L,), device, dtype, low=None, high=None, requires_grad=requires_grad)
    indices = torch.randint(0, L, size=[S])
    test_elements = element[indices].clone()
    return [
        SampleInput(element, args=(test_elements,))
    ]

def sample_inputs_masked_scatter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def samples_generator():
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, make_arg((S, S))))
        yield SampleInput(make_arg((S, S)), args=(torch.randn((S,), device=device) > 0, make_arg((S, S))))
        yield SampleInput(make_arg((S, S)), args=(bernoulli_scalar().to(device), make_arg((S, S))))
        yield SampleInput(make_arg((S,)),
                          args=(torch.randn(S, S, device=device) > 0, make_arg((S, S))),
                          broadcasts_input=True)

    samples = tuple(samples_generator())
    return samples


def sample_inputs_masked_fill(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def sample_generator():
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, 10))
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, make_arg(())))
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, device=device) > 0, 10))
        yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, 10))
        yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, make_arg(())))
        yield SampleInput(make_arg((S, S)), args=(torch.randn((), device=device) > 0, 10))

        yield SampleInput(make_arg((S,)),
                          args=(torch.randn(S, S, device=device) > 0, make_arg(())),
                          broadcasts_input=True)
        yield SampleInput(make_arg((S,)),
                          args=(torch.randn(S, S, device=device) > 0, 10),
                          broadcasts_input=True)

    samples = tuple(sample_generator())
    return samples

def sample_inputs_masked_select(op_info, device, dtype, requires_grad, **kwargs):
    samples = (
        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn(M, M, device=device) > 0,)),

        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M,), device=device) > 0,)),

        SampleInput(make_tensor((M,), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M, M), device=device) > 0,)),

        SampleInput(make_tensor((M, 1, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M, M), device=device) > 0,)),

        SampleInput(make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.tensor(1, device=device, dtype=torch.bool),)),

        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.tensor(1, device=device, dtype=torch.bool),)),

        SampleInput(make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M, M), device=device) > 0,)),
    )

    return samples

def sample_inputs_matrix_exp(op_info, device, dtype, requires_grad, **kwargs):
    samples = (
        SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad)),
        SampleInput(make_tensor((S, S, S), device, dtype, requires_grad=requires_grad)),
    )

    return samples

def sample_inputs_matmul(op_info, device, dtype, requires_grad):
    test_cases = (((L,), (L,)),
                  ((S, M), (M,)),
                  ((M,), (M, S)),
                  ((S, M), (M, S)),
                  ((S, S, M), (M,)),
                  ((S, S, M), (M, S)),
                  ((M,), (S, M, S)),
                  ((S, M), (S, M, S)),
                  ((S, S, M, M), (S, S, M, S)),
                  ((S, S, M, M), (M,)),
                  ((M,), (S, S, M, S)))
    sample_inputs = []
    for lhs_shape, rhs_shape in test_cases:
        lhs = make_tensor(lhs_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        rhs = make_tensor(rhs_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        if op_info.name == 'matmul':
            sample_inputs.append(SampleInput(lhs, args=(rhs,)))
        elif op_info.name == '__rmatmul__':
            sample_inputs.append(SampleInput(rhs, args=(lhs,)))
        else:
            raise RuntimeError("`op_info.name` must be 'matmul' or '__rmatmul__'")
    return tuple(sample_inputs)


def sample_inputs_polar(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    samples = (
        SampleInput(_make_tensor_helper((S, S), low=0), args=(_make_tensor_helper((S, S)),)),
        SampleInput(_make_tensor_helper((), low=0), args=(_make_tensor_helper(()),)),
    )

    return samples

def sample_inputs_complex(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor_helper(shape):
        return make_tensor(shape, device, dtype, requires_grad=requires_grad)

    samples = (
        SampleInput(_make_tensor_helper((S, S)), args=(_make_tensor_helper((S, S)),)),
        SampleInput(_make_tensor_helper(()), args=(_make_tensor_helper(()),)),
    )

    return samples


def sample_inputs_polygamma(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)

    def generator():
        for shape, n in product(tensor_shapes, ns):
            yield SampleInput(make_arg(shape), args=(n,))

    return list(generator())


def sample_inputs_mvlgamma(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)

    # Since the accepted lower bound for input
    # to mvlgamma depends on `p` argument,
    # the following function computes the lower bound
    # which we pass to `make_tensor`.
    def compute_min_val(p):
        return (p - 1.) / 2

    def generator():
        for shape, n in product(tensor_shapes, ns):
            min_val = compute_min_val(n)
            if not dtype.is_floating_point:
                # Round-up minimum value for integral dtypes
                min_val += 1
            yield SampleInput(make_arg(shape, low=min_val), args=(n,))

    return list(generator())


# Since `mvlgamma` has multiple entries,
# there are multiple common skips for the additional
# entries. Following function is a helper to that end.
def skips_mvlgamma(skip_redundant=False):
    skips = (
        # outside domain values are hard error for mvlgamma op.
        SkipInfo('TestUnaryUfuncs', 'test_float_domains'),
    )
    if not skip_redundant:
        # Redundant tests
        skips = skips + (  # type: ignore[assignment]
            SkipInfo('TestGradients'),
            SkipInfo('TestJit'),
            SkipInfo('TestCommon'),
        )
    return skips


# To test reference numerics against multiple values of argument `p`,
# we make multiple OpInfo entries with each entry corresponding to different value of p.
# We run the op tests from test_ops.py only for `p=1` to avoid redundancy in testing.
# Class `MvlGammaInfo` already contains the basic information related to the operator,
# it only takes arguments like `domain`, `skips` and `sample_kwargs`, which
# differ between the entries.
class MvlGammaInfo(UnaryUfuncInfo):
    def __init__(self, variant_test_name, domain, skips, sample_kwargs):
        super(MvlGammaInfo, self).__init__(
            'mvlgamma',
            ref=reference_mvlgamma if TEST_SCIPY else _NOTHING,
            variant_test_name=variant_test_name,
            domain=domain,
            decorators=(precisionOverride({torch.float16: 5e-2}),),
            dtypes=all_types(),
            dtypesIfCUDA=all_types_and(torch.half),
            sample_inputs_func=sample_inputs_mvlgamma,
            supports_out=False,
            safe_casts_outputs=True,
            skips=skips,
            sample_kwargs=sample_kwargs)


def sample_inputs_entr(op_info, device, dtype, requires_grad, **kwargs):
    low, _ = op_info.domain

    if requires_grad:
        low = 0 + op_info._domain_eps

    return (SampleInput(make_tensor((L,), device, dtype,
                                    low=low,
                                    requires_grad=requires_grad)),
            SampleInput(make_tensor((), device, dtype,
                                    low=low,
                                    requires_grad=requires_grad)))


def sample_inputs_zeta(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = (SampleInput(make_arg((S,), low=1, requires_grad=requires_grad),
                           args=(make_arg((S,), low=2, requires_grad=False),)),
               SampleInput(make_arg((S,), low=1, requires_grad=requires_grad),
                           args=(3.,)),
               )

    return samples


# TODO: Consolidate `i0e` with sample_inputs_unary when `make_tensor`,
#       supports `exclude` argument.
#       For more context: https://github.com/pytorch/pytorch/pull/56352#discussion_r633277617
def sample_inputs_i0_i1(op_info, device, dtype, requires_grad, **kwargs):

    samples = (SampleInput(make_tensor((S,), device, dtype,
                                       requires_grad=requires_grad)),
               SampleInput(make_tensor((), device, dtype,
                                       requires_grad=requires_grad)))

    if requires_grad and op_info.op == torch.special.i0e:
        # NOTE: `i0e`'s first-order gradient is not continous
        # at `0`, hence we don't test `i0e` with any input being `0`.
        # TODO: Remove this when `make_tensor` supports excluding `0`.
        with torch.no_grad():
            for sample in samples:
                t = sample.input
                t[t == 0] = torch.finfo(dtype).eps  # type: ignore[index]
    elif requires_grad and op_info.op != torch.special.i0e:
        # Special Case for gradient
        # Sample with `0` in the input
        t = make_tensor((S,), device, dtype,
                        requires_grad=requires_grad)

        with torch.no_grad():
            t[0] = 0

        samples += (SampleInput(t),)  # type: ignore[assignment]

    return samples


def sample_inputs_rsub(op_info, device, dtype, requires_grad, variant='tensor', **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _samples_with_alpha_helper(args, alphas, filter_fn=lambda arg_alpha: True):
        filtered_product = filter(filter_fn, product(args, alphas))  # type: ignore[var-annotated]
        return (SampleInput(input, args=(arg,), kwargs=dict(alpha=alpha))
                for (input, arg), alpha in filtered_product)

    int_alpha, float_alpha, complex_alpha = 2, 0.1, 1 + 0.6j

    if variant == 'tensor':
        samples = (
            SampleInput(_make_tensor_helper((S, S)), args=(_make_tensor_helper((S, S)),)),
            SampleInput(_make_tensor_helper((S, S)), args=(_make_tensor_helper((S,)),)),
            SampleInput(_make_tensor_helper((S,)), args=(_make_tensor_helper((S, S)),)),
            SampleInput(_make_tensor_helper(()), args=(_make_tensor_helper(()),)),
            SampleInput(_make_tensor_helper(()), args=(_make_tensor_helper((S,)),)),
            SampleInput(_make_tensor_helper((S,)), args=(_make_tensor_helper(()),)),
        )

        if dtype.is_complex:
            alphas = [int_alpha, float_alpha, complex_alpha]
        elif dtype.is_floating_point:
            alphas = [int_alpha, float_alpha]
        else:
            alphas = [int_alpha]

        args = ((_make_tensor_helper((S, S)), _make_tensor_helper((S, S))),
                (_make_tensor_helper((S, S)), _make_tensor_helper((S,))),
                (_make_tensor_helper(()), _make_tensor_helper(())))
        samples += tuple(_samples_with_alpha_helper(args, alphas))  # type: ignore[assignment]
    elif variant == 'scalar':
        # Scalar Other
        samples = (SampleInput(_make_tensor_helper((S, S)), args=(0.5,)),
                   SampleInput(_make_tensor_helper(()), args=(0.5,)),
                   SampleInput(_make_tensor_helper((S, S)), args=(1.5j,)),
                   SampleInput(_make_tensor_helper(()), args=(1.5j,)),
                   SampleInput(_make_tensor_helper((S, S)), args=(0.4 + 1.2j,)),
                   SampleInput(_make_tensor_helper(()), args=(1.2 + 1.76j,)))

        scalar_args = [(_make_tensor_helper((S, S)), 0.5), (_make_tensor_helper(()), 0.5),
                       (_make_tensor_helper((S, S)), 2.7j), (_make_tensor_helper(()), 2.7j),
                       (_make_tensor_helper((S, S)), 1 - 2.7j), (_make_tensor_helper(()), 1 + 2.7j)]

        alphas = [int_alpha, float_alpha, complex_alpha]

        def filter_fn(arg_alpha):
            arg, alpha = arg_alpha
            if isinstance(alpha, complex):
                if dtype.is_complex or isinstance(arg[1], complex):
                    return True
                else:
                    # complex alpha is valid only if either `self` or `other` is complex
                    return False

            # Non-Complex Alpha
            return True

        # Samples with alpha (scalar version) covers the following cases
        # self    | other   | alpha
        # -----------------------------------------
        # real    | real    | real (int and float)
        # real    | complex | real and complex
        # complex | real    | real and complex
        # complex | complex | real and complex
        #
        # It does not cover
        # real    | real    | complex
        # x = torch.randn(2, requires_grad=True, dtype=torch.float64)
        # torch.rsub(x, 1, alpha=1. + 1.6j)
        # RuntimeError: value cannot be converted to type double without overflow: (-1,-1.6)

        samples += tuple(_samples_with_alpha_helper(scalar_args, alphas, filter_fn=filter_fn))  # type: ignore[assignment]
    else:
        raise Exception("Invalid variant!")

    return samples

def sample_inputs_cumulative_ops(op_info, device, dtype, requires_grad, supports_dtype_kwargs=True, **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    samples = [
        SampleInput(_make_tensor_helper((S, S, S)), args=(0,)),
        SampleInput(_make_tensor_helper((S, S, S)), args=(1,)),
        SampleInput(_make_tensor_helper(()), args=(0,)),
    ]

    if supports_dtype_kwargs:
        # NOTE: if `dtype` is not same as input, then inplace variants fail with
        # `provided dtype must match the dtype of self tensor in cumsum`
        samples.append(SampleInput(_make_tensor_helper((S, S, S)), args=(1,), kwargs={'dtype': dtype}))

    return samples


def sample_inputs_unfold(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((), (0, 1, 1)),
        ((S, S, S, S), (0, 3, 1)),
        ((S, S, S, S), (1, 3, 1)),
        ((S, S, S, S), (2, 3, 1)),
        ((S, S, S, S), (3, 3, 1)),
        ((S, S, S, S), (0, 3, 2)),
        ((S, S, S, S), (1, 3, 2)),
        ((S, S, S, S), (2, 3, 2)),
        ((S, S, S, S), (3, 3, 2)),
        ((S, S, S, S), (0, 4, 1)),
        ((S, S, S, S), (1, 4, 1)),
        ((S, S, S, S), (2, 4, 1)),
        ((S, S, S, S), (3, 4, 1)),
        ((M,), (0, 3, 1)),
        ((M,), (0, 3, 2)),
        ((M,), (0, 3, 3)),
        ((1000,), (0, 3, 11)),
        ((1000,), (0, 2, 27)),
        ((10, 10), (0, 1, 2)),
        ((10, 10), (1, 2, 3)),
        ((10, 10), (1, 2, 2)),
        ((S, S, S), (2, 3, 2)),
    )

    sample_inputs = []
    for shape, arguments in test_cases:
        sample_inputs += [SampleInput(make_tensor(shape, device, dtype,
                                      low=None, high=None,
                                      requires_grad=requires_grad),
                                      args=arguments)]
    return sample_inputs


def sample_inputs_atan2(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (
        ((S, S, S), (S, S, S), False),
        ((), (), False),
        ((S, S, S), (S,), False),
        ((S,), (S, S, S), True),
        ((S, 1, S), (S, S), True),
    )

    def generator():
        for x_shape, y_shape, broadcasts_input in cases:
            yield SampleInput(make_arg(x_shape), args=(make_arg(y_shape),),
                              broadcasts_input=broadcasts_input)

    return list(generator())


def sample_inputs_split(op_info, device, dtype, requires_grad, *, list_args=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    if list_args:
        cases = (
            ((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],)),
            ((S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)], 2),),
            ((S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)], -2),)
        )
    else:
        cases = (  # type: ignore[assignment]
            ((S, S, S), (2,)),
            ((S, S, S), (S, 1)),
        )

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_split_with_sizes(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],)),
             ((S, S, S), ([int(S / 3), S - int(S / 3), 0],)),
             ((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)], 2)),
             ((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)], -2)),
             )

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_msort(op_info, device, dtype, requires_grad):
    def apply_grad(t):
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            t.requires_grad_(requires_grad)

    def large_1d_unique(dtype, device):
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype)
        apply_grad(res)
        return res

    samples = []
    # Test case for large tensor.
    largesample = SampleInput(large_1d_unique(dtype, device))

    sample = SampleInput(make_tensor((S, M, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad))

    return [largesample, sample]

def sample_inputs_lerp(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    samples = (
        # no broadcast
        SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 0.4)),
        # broadcast rhs
        SampleInput(make_arg((S, S)), args=(make_arg((S,)), 0.4)),
        # scalar tensor
        SampleInput(make_arg(()), args=(make_arg(()), 0.4)),
        # broadcast rhs scalar-tensor
        SampleInput(make_arg((S, S)), args=(make_arg(()), 0.4)),
        # broadcast rhs with weight tensor
        SampleInput(make_arg((S, S)), args=(make_arg((S,)), make_arg((S, S)))),
        # broadcast rhs and weight tensor
        SampleInput(make_arg((S, S)), args=(make_arg((S, 1)), make_arg((S,)))),
        # broadcast_lhs
        SampleInput(make_arg((S,)), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # scalar broadcast_lhs
        SampleInput(make_arg(()), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # broadcast all
        SampleInput(make_arg((S, 1)), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # tensor broadcast all
        SampleInput(make_arg((S, 1)), args=(make_arg((S, S)), make_arg((S, 1))),
                    broadcasts_input=True),
    )

    if dtype.is_complex:
        samples = samples + (  # type: ignore[assignment]
            # no broadcast
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 1.2 + 0.1j)),
            # broadcast rhs
            SampleInput(make_arg((S, S)), args=(make_arg((S,)), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 5.4 + 9j)),
            # scalar tensor
            SampleInput(make_arg(()), args=(make_arg(()), 0.4j)),
            SampleInput(make_arg(()), args=(make_arg(()), 6.1 + 0.004j)),
            # broadcast rhs scalar-tensor
            SampleInput(make_arg((S, S)), args=(make_arg(()), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg(()), 1 + 2j)),
        )

    return samples

def sample_inputs_tensordot(self, device, dtype, requires_grad, **kwargs):
    cases = (
        ((2, 2, 2), (2, 2, 2), (2)),
        ((2, 2, 1), (2, 1, 2), ([0, 1], [2, 0])),
    )
    samples = []
    for first_shape, second_shape, dims in cases:
        samples.append(SampleInput(make_tensor(first_shape, device, dtype,
                                   requires_grad=requires_grad),
                       args=(make_tensor(second_shape, device, dtype,
                             requires_grad=requires_grad),),
                       kwargs=dict(dims=dims,)))
    return tuple(samples)

def sample_inputs_kron(op_info, device, dtype, requires_grad):
    test_cases = (
        ((S, S), (M, L)),
    )

    sample_inputs = []
    for input_shape, other_shape in test_cases:
        input = make_tensor(input_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        other = make_tensor(other_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        sample = SampleInput(input, args=(other,))
        sample_inputs.append(sample)
    return tuple(sample_inputs)

def sample_inputs_inner(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, ), device, dtype, requires_grad=requires_grad),
            args=(
                make_tensor((S, ), device, dtype, requires_grad=requires_grad),
            )
        ),
        SampleInput(
            make_tensor((), device, dtype, requires_grad=requires_grad),
            args=(
                make_tensor((S, S), device, dtype, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_scatter(op_info, device, dtype, requires_grad):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        return gather_variable(shape, index_dim, max_indices, device=device)

    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (
        (_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))),
        (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))),
        (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor(()), (0, zero.clone().detach(), _tensor(()))),
        (_tensor(()), (0, zero.clone().detach(), 2.5)),
    )

    samples = []
    for tensor, args in test_cases:
        samples.append(SampleInput(tensor, args=args))

        if not requires_grad:
            samples.append(SampleInput(
                tensor.clone().detach(),
                args=args, kwargs={'reduce': 'add'}
            ))

            if dtype.is_floating_point:
                samples.append(SampleInput(
                    tensor.clone().detach(),
                    args=args, kwargs={'reduce': 'multiply'}
                ))

    return samples

def sample_inputs_scatter_add(op_info, device, dtype, requires_grad):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        return gather_variable(shape, index_dim, max_indices, device=device)

    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (
        (_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))),
        (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))),
        (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor(()), (0, zero.clone().detach(), _tensor(()))),
    )

    return [SampleInput(tensor, args=args) for tensor, args in test_cases]


def sample_inputs_ravel(op_info, device, dtype, requires_grad, **kwargs):
    samples = (SampleInput(make_tensor((S, S, S), device, dtype,
                                       low=None, high=None,
                                       requires_grad=requires_grad)),
               SampleInput(make_tensor((), device, dtype,
                                       low=None, high=None,
                                       requires_grad=requires_grad)),)

    return samples


def sample_inputs_tril_triu(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((M, M), ()),
             ((M, M), (2,),),
             ((S, M, M), ()),
             ((S, M, M), (2,)),
             ((3, 3, S, S), ()),)

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_clone(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    def generator():
        yield SampleInput(make_arg((S, M, S)))
        yield SampleInput(make_arg(()))

    return list(generator())


def sample_inputs_contiguous(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    def generator():
        yield SampleInput(make_arg((S, S)))
        yield SampleInput(make_arg((S, S), noncontiguous=True))

    return list(generator())


def sample_inputs_resize_ops(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)
    cases = (((S, S, S), (S * S, S)),
             ((), ()),
             ((), (1, 1, 1)),
             )

    def generator():
        for shape, args_or_shape in cases:
            # Update `args` based on operator
            if op_info.name == 'resize_':
                # resize_ takes shape/tuple of ints,
                args = (args_or_shape, )
            elif op_info.name == 'resize_as_':
                # resize_as_ takes another tensor
                args = (make_arg(shape, requires_grad=False), )  # type:ignore[assignment]
            else:
                raise ValueError("sample_inputs_resize_ops is being used with incorrect operator")

            yield(SampleInput(make_arg(shape, requires_grad=requires_grad), args=args))

    return list(generator())


def sample_inputs_view_reshape(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((S, S, S), (S * S, S)),
             ((S * S, S), (S, S, S)),
             ((S,), (S,)),
             ((), ()),
             ((), (1,)))

    def generator():
        for case in cases:
            shape, args = case
            yield(SampleInput(make_arg(shape), args=(args, )))

    return list(generator())


def sample_inputs_view_as_reshape_as(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)

    cases = (((S, S, S), (S * S, S)),
             ((), ()),
             ((), (1, 1)),
             )

    def generator():
        for case in cases:
            shape, shape_other = case
            yield(SampleInput(make_arg(shape, requires_grad=requires_grad),
                              args=(make_arg(shape_other, requires_grad=False), )))

    return list(generator())


def sample_inputs_select(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((S, S, S), (1, 2)),
             ((S, S, S), (-1, 2)),
             ((S, S, S), (-1, -1)),
             ((S, S, S), (1, -1)),
             ((S,), (0, 2))
             )

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_rbinops(op_info, device, dtype, requires_grad, supports_dtype_kwargs=True, **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    scalar: Union[int, float, complex] = 3

    if dtype.is_floating_point:
        scalar = 3.14
    elif dtype.is_complex:
        scalar = 3.14j

    samples = [
        SampleInput(_make_tensor_helper((S, S, S)), args=(scalar,)),
        SampleInput(_make_tensor_helper(()), args=(scalar,)),
    ]

    return samples


def sample_inputs_expand(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((S, 1, 1), (S, S, S)),
             ((S, 1, S), (S, S, S)),
             ((S, 1), (S, S, S)),
             ((1,), (S, S, S)),
             ((1, S), (1, 1, S)),
             ((), ()),
             ((), (1, 3, 2)),
             )

    def generator():
        for case in cases:
            shape, args = case
            yield(SampleInput(make_arg(shape), args=(args, )))

    return list(generator())


def sample_inputs_expand_as(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)

    cases = (((S, 1, 1), (S, S, S)),
             ((), ()),
             ((), (1, 1)),
             )

    def generator():
        for shape, shape_other in cases:
            yield(SampleInput(make_arg(shape, requires_grad=requires_grad),
                              args=(make_arg(shape_other, requires_grad=False), )))

    return list(generator())


def sample_inputs_where(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    def make_bool_mask(shape):
        # Make sure atleast one element is nonzero,
        # except for empty tensor
        mask_t = make_tensor(shape, dtype=torch.bool, device=device, requires_grad=False)

        if mask_t.numel() == 0:
            return mask_t
        elif mask_t.numel() == 1:
            mask_t.fill_(True)
            return mask_t

        if mask_t.sum() == 0:
            def random_index(shape):
                return tuple(map(lambda max_idx: random.randint(0, max_idx), shape))

            mask_t[random_index(mask_t.shape)] = True
            return mask_t

        return mask_t

    cases = (((M, M), (M, M), (M, M), False),
             ((M, 1, M), (M, M), (M, M, 1), True),
             ((), (), (), False),
             ((M, 1, M), (), (M, M, 1), True),
             ((), (M, M), (), True),)

    def generator():
        for shape, mask_shape, other_shape, broadcasts_input in cases:
            yield SampleInput(make_arg(shape),
                              args=(make_bool_mask(mask_shape), make_arg(other_shape)),
                              broadcasts_input=broadcasts_input)

    return list(generator())


def sample_inputs_chunk(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)

    cases = (((S, S, S), (2,)),
             ((S, S, S), (S, 1)),
             ((S, S, S), (S, -1)))

    def generator():
        for case in cases:
            shape, args = case
            yield(SampleInput(make_arg(shape, requires_grad=requires_grad), args=args))

    return list(generator())

def sample_inputs_kthvalue(op_info, device, dtype, requires_grad, **kwargs):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    test_cases = [
        (_tensor((S, S, S)), (2,)),
        (_tensor((S, S, S)), (2, 1,)),
        (_tensor((S, S, S)), (2, -1,)),
        (_tensor((S, S, S)), (2, 1, True,)),
        (_tensor((S, S, S)), (2, -1, True,)),
        (_tensor((S,)), (2, 0,)),
        (_tensor((S,)), (2, 0, True,)),
        (_tensor(()), (1,)),
        (_tensor(()), (1, 0,)),
        (_tensor(()), (1, 0, True))
    ]

    return [SampleInput(tensor, args=args) for tensor, args in test_cases]

foreach_unary_op_db: List[OpInfo] = [
    ForeachFuncInfo('exp'),
    ForeachFuncInfo('acos'),
    ForeachFuncInfo('asin'),
    ForeachFuncInfo('atan'),
    ForeachFuncInfo('cos'),
    ForeachFuncInfo('cosh'),
    ForeachFuncInfo('log'),
    ForeachFuncInfo('log10'),
    ForeachFuncInfo('log2'),
    ForeachFuncInfo('tan'),
    ForeachFuncInfo('tanh'),
    ForeachFuncInfo('sin'),
    ForeachFuncInfo('sinh'),

    ForeachFuncInfo(
        'neg',
        dtypes=all_types_and_complex(),
        dtypesIfCPU=all_types_and_complex(),
        dtypesIfCUDA=all_types_and_complex(),
        sample_inputs_func=sample_inputs_foreach,
        safe_casts_outputs=False,
    ),

    ForeachFuncInfo(
        'sqrt',
        dtypes=floating_types(),
        dtypesIfCPU=floating_and_complex_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'ceil',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'erf',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'erfc',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'expm1',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'floor',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'log1p',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'round',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'frac',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'reciprocal',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'sigmoid',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'trunc',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'abs',
        dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
        dtypesIfCPU=all_types_and_complex_and(torch.bfloat16, torch.half),
        dtypesIfCUDA=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
        safe_casts_outputs=False,
        supports_forward_ad=True,
    ),
]

def reference_sign(x):
    if x.dtype == np.bool_:
        # `np.sign` doesn't support `bool`.
        # >>> np.sign(True)
        # ufunc 'sign' did not contain a loop
        # with signature matching types dtype('bool') -> dtype('bool')
        return np.sign(x, dtype=np.uint8).astype(np.bool_)
    return np.sign(x)


def reference_sgn(x):
    # NumPy doesn't have an equivalent to `torch.sgn` when the dtype is complex.
    # For complex inputs, `np.sign` returns sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j.
    # while `torch.sgn` returns, 0 if abs(input) == 0 else input/abs(input)
    if x.dtype not in [np.complex64, np.complex128]:
        return reference_sign(x)

    out = (x / np.abs(x))
    if out.ndim == 0:
        # Handle x == 0 case
        if (x == 0):
            # Can't assign to np.complex object
            # So make a new one.
            return np.array(complex(0, 0), dtype=x.dtype)
        return out

    # Handle x == 0 case
    mask = (x == 0)
    out[mask] = complex(0, 0)
    return out


def reference_sigmoid(x):
    # 'scipy.special.expit' not supported for the input types
    if x.dtype in [np.complex64, np.complex128]:
        return (1 / (1 + np.exp(-x)))
    return scipy.special.expit(x)


def reference_lgamma(x):
    # scipy.special.gammaln returns `-inf` when input is `-inf`.
    # While Pytorch, C and C++, all return `inf` when input is `-inf`.
    # Reference:
    # https://en.cppreference.com/w/cpp/numeric/math/lgamma
    # https://en.cppreference.com/w/c/numeric/math/lgamma

    # To handle the above discrepancy,
    # we replace -inf with inf so values
    # that were originally -inf map to inf as expected
    if x.dtype.kind == 'f':
        x = np.where(x == float('-inf'), np.array(float('inf'), dtype=x.dtype), x)

    out = scipy.special.gammaln(x)

    if x.dtype == np.float16:
        # `scipy.special.gammaln` returns output of float32 when input is float16,
        # while `torch.lgamma` preserves `float16`. But due to smaller range of float16,
        # Pytorch version outputs `inf` while SciPy returns finite values.
        out = out.astype(np.float16)

    return out

def reference_polygamma(x, n):
    # WEIRD `scipy.special.polygamma` behavior
    # >>> scipy.special.polygamma(0, np.array(501, dtype=np.float32)).dtype
    # dtype('float64')
    # >>> scipy.special.polygamma(0, np.array([501], dtype=np.float32)).dtype
    # dtype('float32')
    #
    # Thus we cast output to the default torch dtype.
    np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]
    return scipy.special.polygamma(n, x).astype(np_dtype)


def reference_mvlgamma(x, d):
    if x.dtype == np.float16:
        return scipy.special.multigammaln(x, d).astype(np.float16)

    return scipy.special.multigammaln(x, d)


def gradcheck_wrapper_hermitian_input(op, input, *args, **kwargs):
    """Gradcheck wrapper for functions that take Hermitian matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the Hermitian property of the input.
    """
    return op(input + input.conj().transpose(-2, -1), *args, **kwargs)


def gradcheck_wrapper_triangular_input(op, input, *args, upper=False, **kwargs):
    """Gradcheck wrpper for functions that take lower or upper triangular matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the triangular property of the input.
    """
    return op(input.triu() if upper else input.tril(), upper)


# Operator database (sorted alphabetically)
op_db: List[OpInfo] = [
    UnaryUfuncInfo('abs',
                   aliases=('absolute', ),
                   ref=np.abs,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat]),
                       # Reference: https://github.com/pytorch/pytorch/issues/49224
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.int8], active_if=TEST_WITH_ASAN),
                       # TODO: Fix test_out_arg_all_dtypes as torch.empty_like(expected_output) where expected_output=op(input)
                       # We can break the logic of the loop over all possible types but it is OK.
                       # https://github.com/pytorch/pytorch/blob/master/test/test_unary_ufuncs.py#L440-L449
                       SkipInfo('TestUnaryUfuncs', 'test_out_arg_all_dtypes',
                                dtypes=[torch.cfloat, torch.cdouble]),
                   ),
                   supports_inplace_autograd=False,
                   assert_autodiffed=True,
                   supports_forward_ad=True),
    # NOTE: CPU complex acos produces incorrect outputs (https://github.com/pytorch/pytorch/issues/42952)
    UnaryUfuncInfo('acos',
                   aliases=('arccos', ),
                   ref=np.arccos,
                   domain=(-1, 1),
                   handles_complex_extremals=False,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "rsqrt_cpu" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-1,
                                                  torch.complex64: 1e-2}),),
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestGradients', 'test_fn_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_method_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_inplace_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_forward_mode_AD',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                   )),
    # NOTE: the derivative for inplace acosh is not implemented
    UnaryUfuncInfo('acosh',
                   aliases=('arccosh', ),
                   ref=np.arccosh,
                   domain=(1, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "rsqrt_cuda" not implemented for 'BFloat16'
                   backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   supports_inplace_autograd=False,
                   supports_forward_ad=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       # Reference: https://github.com/pytorch/pytorch/issues/50692
                       SkipInfo('TestGradients', 'test_fn_grad',
                                device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_method_grad',
                                device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_forward_mode_AD',
                                dtypes=[torch.cdouble]),
                   )),
    OpInfo('add',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           assert_autodiffed=True,
           sample_inputs_func=partial(sample_inputs_binary_pwise, alpha=2),
           supports_inplace_autograd=False,
           supports_forward_ad=True),
    OpInfo('mul',
           aliases=('multiply',),
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool),
           assert_autodiffed=True,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_binary_pwise),
    OpInfo('sub',
           aliases=('subtract',),
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16),
           assert_autodiffed=True,
           sample_inputs_func=partial(sample_inputs_binary_pwise, alpha=2),
           supports_inplace_autograd=False),
    OpInfo('addmm',
           # This addmm OpInfo is for when alpha and beta are not both equal to 1.
           # alpha=beta=1 is tested in the following opinfo, because that special case will
           # trigger addmm being decomposed by a jit pass.
           dtypes=floating_and_complex_types_and(torch.float16),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_addmm),
    OpInfo('addmm',
           # When alpha=beta=1 as compile-time constants, JIT will decompose addmm into mm and add.
           variant_test_name='decomposed',
           dtypes=floating_and_complex_types_and(torch.float16),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           autodiff_nonfusible_nodes=['aten::add', 'aten::mm'],
           sample_inputs_func=partial(sample_inputs_addmm, alpha=1, beta=1)),
    OpInfo('addmv',
           dtypes=floating_types(),
           dtypesIfCPU=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128,
                                           *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half),
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           skips=(
               # issue may fix: https://github.com/pytorch/pytorch/issues/55589
               # AssertionError: UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               SkipInfo('TestCommon', 'test_out', dtypes=(torch.float32,)),
               # Reference: https://github.com/pytorch/pytorch/issues/55589
               SkipInfo('TestCommon', 'test_variant_consistency_eager'),
           ),
           sample_inputs_func=sample_inputs_addmv),
    OpInfo('addbmm',
           dtypes=floating_types(),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if SM53OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half),
           backward_dtypesIfROCM=floating_types_and(torch.half),
           supports_forward_ad=True,
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+
               #   and SM53+
               SkipInfo('TestCommon', 'test_dtypes', active_if=IS_WINDOWS),
               # addbmm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
               # https://github.com/pytorch/pytorch/issues/55907
               SkipInfo('TestCommon', 'test_variant_consistency_eager'),
           ),
           sample_inputs_func=sample_inputs_addbmm),
    OpInfo('baddbmm',
           dtypes=floating_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128,
                                           *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_types_and(torch.float16,
                                                    *[torch.bfloat16] if SM53OrLater else [],
                                                    torch.complex64, torch.complex128),
           supports_forward_ad=True,
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+
               #   and SM53+
               SkipInfo('TestCommon', 'test_dtypes', active_if=IS_WINDOWS),
               # baddbmm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           ),
           sample_inputs_func=sample_inputs_baddbmm),
    OpInfo('dot',
           dtypes=all_types_and_complex_and(torch.float16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_dot_vdot,
           supports_forward_ad=True,
           ),
    OpInfo('vdot',
           dtypes=all_types_and_complex_and(torch.float16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_dot_vdot,
           supports_forward_ad=True,
           ),
    OpInfo('bmm',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if SM53OrLater else []),
           assert_autodiffed=True,
           supports_forward_ad=True,
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+
               #   and SM53+
               SkipInfo('TestCommon', 'test_dtypes', active_if=IS_WINDOWS),
               # bmm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           ),
           sample_inputs_func=sample_inputs_bmm),
    OpInfo('mv',
           dtypes=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           skips=(
               # bmm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_mv),
    OpInfo('addr',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           backward_dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
           backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           # Reference: https://github.com/pytorch/pytorch/issues/50747
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/50747
               SkipInfo('TestCommon', 'test_variant_consistency_eager',
                        dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16)),
           ),
           sample_inputs_func=sample_inputs_addr,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('addcmul',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.float16, torch.bfloat16),
           assert_autodiffed=True,
           supports_forward_ad=True,
           supports_inplace_autograd=False,
           skips=(
               # TODO: update sample inputs with for_inplace_variant kwarg to support this test
               SkipInfo('TestCommon', 'test_variant_consistency_eager'),),
           sample_inputs_func=sample_inputs_addcmul_addcdiv),
    OpInfo('addcdiv',
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           skips=(
               # TODO: update sample inputs with for_inplace_variant kwarg to support this test
               SkipInfo('TestCommon', 'test_variant_consistency_eager'),),
           sample_inputs_func=sample_inputs_addcmul_addcdiv),
    OpInfo('amax',
           ref=lambda a, dim=None, keepdim=False, **kwargs: np.amax(a, axis=dim, keepdims=keepdim, **kwargs),
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_amax_amin,),
    OpInfo('amin',
           ref=lambda a, dim=None, keepdim=False, **kwargs: np.amin(a, axis=dim, keepdims=keepdim, **kwargs),
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_amax_amin),
    OpInfo('argmax',
           dtypes=all_types_and(torch.float16, torch.bfloat16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_argmax_argmin,),
    OpInfo('argmin',
           dtypes=all_types_and(torch.float16, torch.bfloat16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_argmax_argmin,),
    UnaryUfuncInfo('asin',
                   aliases=('arcsin', ),
                   ref=np.arcsin,
                   domain=(-1, 1),
                   supports_sparse=True,
                   supports_forward_ad=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    # NOTE: derivative for inplace asinh is not implemented
    UnaryUfuncInfo('asinh',
                   aliases=('arcsinh', ),
                   ref=np.arcsinh,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   supports_inplace_autograd=False,
                   supports_forward_ad=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       # Complex gradcheck tests asinh at points 0 + ix for x > 1 which are points
                       # where asinh is not differentiable
                       SkipInfo('TestGradients', 'test_forward_mode_AD',
                                dtypes=complex_types()),
                   )),
    UnaryUfuncInfo('atan',
                   aliases=('arctan', ),
                   ref=np.arctan,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    OpInfo('atan2',
           dtypes=all_types_and(torch.bool),
           dtypesIfCPU=all_types_and(torch.bool),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_atan2,
           ),
    UnaryUfuncInfo('atanh',
                   aliases=('arctanh', ),
                   ref=np.arctanh,
                   domain=(-1, 1),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   supports_inplace_autograd=False,
                   supports_forward_ad=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cfloat],
                                active_if=IS_WINDOWS),
                   )),
    OpInfo('broadcast_to',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_broadcast_to),
    UnaryUfuncInfo('bitwise_not',
                   ref=np.bitwise_not,
                   dtypes=integral_types_and(torch.bool),
                   supports_autograd=False),
    OpInfo('bitwise_left_shift',
           op=torch.bitwise_left_shift,
           dtypesIfCPU=all_types(),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_bitwise_shift),
    OpInfo('bitwise_right_shift',
           op=torch.bitwise_right_shift,
           dtypesIfCPU=all_types(),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_bitwise_shift),
    OpInfo('cdist',
           dtypes=floating_types(),
           supports_out=False,
           supports_gradgrad=False,
           assert_autodiffed=False,
           sample_inputs_func=sample_inputs_cdist,
           ),
    UnaryUfuncInfo('ceil',
                   ref=np.ceil,
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   assert_autodiffed=True),
    OpInfo('cholesky',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # Gradcheck for complex generates invalid inputs for this function
               SkipInfo('TestGradients', 'test_forward_mode_AD', dtypes=complex_types()),)),
    OpInfo('cholesky_inverse',
           dtypes=floating_and_complex_types(),
           backward_dtypes=floating_types(),
           # TODO: RuntimeError: cholesky_inverse does not support automatic differentiation for outputs
           # with complex dtype.
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky_inverse,
           gradcheck_wrapper=gradcheck_wrapper_triangular_input,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           skips=(
               # TODO: FIXME: cholesky_inverse throws an error in forward when requires_grad=True
               #   for complex tensors
               SkipInfo('TestCommon', 'test_dtypes'),
               # cholesky_inverse does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),)),
    OpInfo('chunk',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_chunk,
           supports_out=False),
    OpInfo('clone',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_clone,
           supports_forward_ad=True,
           supports_out=False),
    OpInfo('contiguous',
           op=lambda x, *args, **kwargs: x.contiguous(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_contiguous,
           supports_forward_ad=True,
           skips=(
               # JIT has issue when op is passed as lambda
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('symeig',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_symeig,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    # NOTE: clamp has seperate opinfos for scalar min/max (unary op) vs. tensors
    OpInfo('clamp',
           aliases=('clip',),
           dtypes=all_types_and(torch.half, torch.bfloat16),
           dtypesIfCPU=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_clamp),
    UnaryUfuncInfo('clamp',
                   variant_test_name='scalar',
                   aliases=('clip', ),
                   decorators=(precisionOverride({torch.bfloat16: 7e-2, torch.float16: 1e-2}),),
                   ref=np.clip,
                   dtypes=all_types_and(torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/54841
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                   ),
                   sample_kwargs=sample_kwargs_clamp_scalar,
                   sample_inputs_func=sample_inputs_clamp_scalar),
    UnaryUfuncInfo('positive',
                   ref=np.positive,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   supports_out=False,
                   supports_forward_ad=True,
                   ),
    UnaryUfuncInfo('conj',
                   ref=np.conj,
                   dtypes=all_types_and_complex_and(torch.bool,
                                                    torch.bfloat16, torch.half),
                   supports_sparse=True,
                   supports_forward_ad=True,
                   supports_out=False),
    UnaryUfuncInfo('conj_physical',
                   ref=np.conj,
                   dtypes=all_types_and_complex_and(torch.bool,
                                                    torch.bfloat16, torch.half),
                   supports_forward_ad=True,
                   skips=(
                       SkipInfo('TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, )),
                   )),
    OpInfo('resolve_conj',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_view_as_real,
           supports_forward_ad=True,
           supports_out=False,
           ),
    OpInfo('view_as_real',
           dtypes=complex_types(),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_view_as_real,
           test_conjugated_samples=False,
           ),
    OpInfo('view_as_complex',
           dtypes=floating_types_and(torch.half),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_view_as_complex),
    OpInfo('complex',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_complex,
           supports_forward_ad=True,
           ),
    OpInfo('copysign',
           dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_copysign,
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           ),
    UnaryUfuncInfo('cos',
                   ref=np.cos,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   handles_large_floats=False,
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                   )),
    UnaryUfuncInfo('cosh',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.cosh),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/48641
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.int8]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                   )),
    OpInfo('cross',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.half),
           sample_inputs_func=sample_inputs_cross,
           supports_forward_ad=True,
           skips=(
               # AssertionError: UserWarning not triggered :
               # Resized a non-empty tensor but did not warn about it.
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('cumsum',
           dtypesIfCPU=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16),
           supports_forward_ad=True,
           skips=(
               # cumsum does not handle correctly out= dtypes
               SkipInfo('TestCommon', 'test_out'),
           ),
           sample_inputs_func=sample_inputs_cumulative_ops),
    OpInfo('cumprod',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.float16, torch.bfloat16),
           supports_forward_ad=True,
           skips=(
               # cumprod does not handle correctly out= dtypes
               SkipInfo('TestCommon', 'test_out',
                        dtypes=[torch.float32]),
           ),
           # gradgradcheck fails in fast_mode=True: #56275
           sample_inputs_func=sample_inputs_cumprod,
           gradcheck_fast_mode=False),
    OpInfo('cummax',
           dtypesIfCPU=all_types_and(torch.bool),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_cumulative_ops, supports_dtype_kwargs=False),
           supports_forward_ad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('cummin',
           dtypesIfCPU=all_types_and(torch.bool),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_cumulative_ops, supports_dtype_kwargs=False),
           supports_forward_ad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    UnaryUfuncInfo('deg2rad',
                   ref=np.radians,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/51283#issuecomment-770614273
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16]),
                   ),
                   safe_casts_outputs=True),
    OpInfo('diff',
           op=torch.diff,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_diff),
    OpInfo('div',
           aliases=('divide',),
           variant_test_name='no_rounding_mode',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_binary_pwise, rhs_exclude_zero=True),
           supports_forward_ad=True,
           assert_autodiffed=True),
    OpInfo('div',
           aliases=('divide',),
           variant_test_name='trunc_rounding',
           dtypes=all_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_binary_pwise, extra_kwargs={
                                      "rounding_mode": 'trunc'}, rhs_exclude_zero=True),
           supports_forward_ad=True,
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/59174
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           assert_autodiffed=True),
    OpInfo('div',
           aliases=('divide',),
           variant_test_name='floor_rounding',
           dtypes=all_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_binary_pwise, extra_kwargs={
                                      "rounding_mode": 'floor'}, rhs_exclude_zero=True),
           supports_forward_ad=True,
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/59174
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           assert_autodiffed=True),
    OpInfo('true_divide',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_forward_ad=True,
           sample_inputs_func=partial(sample_inputs_binary_pwise, rhs_exclude_zero=True)),
    UnaryUfuncInfo('exp',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.exp),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/50093#pullrequestreview-561791547
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal', dtypes=[torch.bfloat16]),
                       # Reference: https://github.com/pytorch/pytorch/issues/48010
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                   ),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   safe_casts_outputs=True),
    OpInfo('expand',
           op=lambda self, shape: self.expand(shape),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_expand,
           skips=(
               # Because expand does not have a function variant.
               SkipInfo('TestJit', 'test_variant_consistency_jit'),),
           supports_forward_ad=True,
           supports_out=False),
    OpInfo('expand_as',
           op=lambda self, other: self.expand_as(other),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_expand_as,
           skips=(
               # Because expand_as does not have a function variant.
               SkipInfo('TestJit', 'test_variant_consistency_jit'),),
           supports_out=False),
    OpInfo('diag',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCPU=all_types_and_complex_and(torch.bool),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_diag),
    OpInfo('diag_embed',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_diagonal_diag_embed),
    OpInfo('diagonal',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           sample_inputs_func=sample_inputs_diagonal_diag_embed),
    OpInfo('eq',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('fmax',
           op=torch.fmax,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_max_min_binary,),
    OpInfo('fmin',
           op=torch.fmin,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_max_min_binary,),
    OpInfo('fmod',
           dtypes=all_types_and(torch.float16),
           sample_inputs_func=sample_inputs_fmod_remainder),
    OpInfo('fmod',
           variant_test_name='autodiffed',
           dtypes=all_types_and(torch.float16, torch.bool),
           assert_autodiffed=True,
           sample_inputs_func=partial(sample_inputs_fmod_remainder, autodiffed=True)),
    OpInfo('remainder',
           dtypesIfCPU=all_types_and(torch.float16),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_fmod_remainder),
    OpInfo('remainder',
           variant_test_name='autodiffed',
           dtypesIfCPU=all_types_and(torch.float16, torch.bool),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bool, torch.bfloat16),
           assert_autodiffed=True,
           sample_inputs_func=partial(sample_inputs_fmod_remainder, autodiffed=True)),
    UnaryUfuncInfo('frac',
                   ref=lambda x: np.modf(x)[0],
                   dtypes=floating_types_and(torch.bfloat16, torch.float16),
                   dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   # Reference for disabling extremals
                   # https://github.com/pytorch/pytorch/issues/51948
                   handles_extremals=False),
    SpectralFuncInfo('fft.fft',
                     aten_name='fft_fft',
                     ref=np.fft.fft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types()),
    SpectralFuncInfo('fft.fftn',
                     aten_name='fft_fftn',
                     ref=np.fft.fftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     decorators=[precisionOverride(
                         {torch.float: 1e-4, torch.cfloat: 1e-4})],),
    SpectralFuncInfo('fft.hfft',
                     aten_name='fft_hfft',
                     ref=np.fft.hfft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.rfft',
                     aten_name='fft_rfft',
                     ref=np.fft.rfft,
                     ndimensional=False,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.rfftn',
                     aten_name='fft_rfftn',
                     ref=np.fft.rfftn,
                     ndimensional=True,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False,
                     decorators=[precisionOverride({torch.float: 1e-4})],),
    SpectralFuncInfo('fft.ifft',
                     aten_name='fft_ifft',
                     ref=np.fft.ifft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types()),
    SpectralFuncInfo('fft.ifftn',
                     aten_name='fft_ifftn',
                     ref=np.fft.ifftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    SpectralFuncInfo('fft.ihfft',
                     aten_name='fft_ihfft',
                     ref=np.fft.ihfft,
                     ndimensional=False,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_types(),
                     check_batched_grad=False),
    SpectralFuncInfo('fft.irfft',
                     aten_name='fft_irfft',
                     ref=np.fft.irfft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.irfftn',
                     aten_name='fft_irfftn',
                     ref=np.fft.irfftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False,
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    UnaryUfuncInfo('floor',
                   ref=np.floor,
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   assert_autodiffed=True),
    OpInfo('flip',
           op=torch.flip,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_flip,
           supports_out=False),
    OpInfo('fliplr',
           op=torch.fliplr,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_fliplr_flipud,
           supports_out=False),
    OpInfo('flipud',
           op=torch.flipud,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_fliplr_flipud,
           supports_out=False),
    UnaryUfuncInfo('i0',
                   ref=np_unary_ufunc_integer_promotion_wrapper(
                       scipy.special.i0) if TEST_SCIPY else _NOTHING,
                   aliases=('special.i0',),
                   decorators=(precisionOverride({torch.bfloat16: 3e-1,
                                                  torch.float16: 5e-1}),),
                   backward_dtypesIfCPU=floating_types(),
                   backward_dtypesIfCUDA=floating_types(),
                   backward_dtypesIfROCM=floating_types(),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   sample_inputs_func=sample_inputs_i0_i1),
    UnaryUfuncInfo('special.i0e',
                   aten_name='special_i0e',
                   ref=scipy.special.i0e if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.bfloat16: 3e-1,
                                                  torch.float16: 3e-1}),),
                   backward_dtypesIfCPU=floating_types(),
                   backward_dtypesIfCUDA=floating_types(),
                   backward_dtypesIfROCM=floating_types(),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   sample_inputs_func=sample_inputs_i0_i1,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('special.i1',
                   aten_name='special_i1',
                   ref=np_unary_ufunc_integer_promotion_wrapper(scipy.special.i1) if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.float: 1e-4}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool),
                   sample_inputs_func=sample_inputs_i0_i1,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('special.i1e',
                   aten_name='special_i1e',
                   ref=scipy.special.i1e if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool),
                   sample_inputs_func=sample_inputs_i0_i1,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('special.ndtr',
                   aten_name='special_ndtr',
                   decorators=(precisionOverride({torch.bfloat16: 5e-3,
                                                  torch.float16: 5e-4}),),
                   ref=scipy.special.ndtr if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.float16),
                   safe_casts_outputs=True),
    OpInfo('floor_divide',
           dtypes=all_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_floor_divide,
           supports_autograd=False,
           ),
    UnaryUfuncInfo('frexp',
                   op=torch.frexp,
                   ref=np.frexp,
                   dtypes=floating_types_and(torch.half),
                   # skip testing torch.frexp as it is not supported by ROCm platform yet
                   decorators=[skipCUDAIfRocm],
                   supports_out=False,
                   supports_forward_ad=True,
                   skips=(
                       # skips below tests as torch.frexp returns tuple-like (mantissa, exponent) as outputs,
                       # while theses tests currently requires output to a single tensor.
                       SkipInfo('TestUnaryUfuncs', 'test_batch_vs_slicing'),
                       SkipInfo('TestUnaryUfuncs', 'test_contig_vs_every_other'),
                       SkipInfo('TestUnaryUfuncs', 'test_contig_vs_transposed'),
                       SkipInfo('TestUnaryUfuncs', 'test_non_contig_expand'),
                       SkipInfo('TestUnaryUfuncs', 'test_variant_consistency'),

                       # skips test_reference_numerics due to error in Windows CI.
                       # The np.frexp returns exponent as np.intc dtype on Windows platform,
                       # and np.intc does not have the correspond torch dtype
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                active_if=IS_WINDOWS),
                   )),
    OpInfo('ge',
           aliases=('greater_equal',),
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('geqrf',
           dtypes=floating_and_complex_types(),
           dtypesIfCPU=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_geqrf,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],),
    OpInfo('gt',
           aliases=('greater',),
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    UnaryUfuncInfo('imag',
                   ref=np.imag,
                   dtypes=complex_types(),
                   supports_out=False,
                   supports_forward_ad=True,
                   # TODO(@anjali411): Test this once neg bit is added.
                   test_conjugated_samples=False,
                   skips=(
                       # Skip since real and imag don't have out variants.
                       SkipInfo('TestUnaryUfuncs', 'test_out_arg_all_dtypes'),
                   )),
    OpInfo('gradient',
           dtypes=floating_and_complex_types_and(torch.int8, torch.int16,
                                                 torch.int32, torch.int64,
                                                 torch.bfloat16, torch.half),
           supports_out=False,
           skips=(
               # following tests give a runtime error with undefined value tensor
               # see discussion : https://github.com/pytorch/pytorch/issues/56660
               SkipInfo('TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, torch.complex64)),
           ),
           supports_inplace_autograd=False,
           sample_inputs_func=sample_inputs_gradient),
    OpInfo('inverse',
           op=torch.inverse,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('isin',
           dtypesIfCPU=all_types(),
           dtypesIfCUDA=all_types_and(torch.half),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_isin),
    OpInfo('kthvalue',
           dtypes=all_types(),
           dtypesIfCPU=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_kthvalue),
    OpInfo('le',
           aliases=('less_equal',),
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('linalg.det',
           op=torch.linalg.det,
           aliases=('det', ),
           dtypes=floating_and_complex_types(),
           # det doesn't support complex autograd, https://github.com/pytorch/pytorch/issues/57358
           backward_dtypes=floating_types(),
           aten_name='linalg_det',
           sample_inputs_func=sample_inputs_linalg_det,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           supports_inplace_autograd=False,
           skips=(
               # linalg.det throwns an error when given complex inputs that require grad
               SkipInfo('TestCommon', 'test_dtypes'),
               # The following tests fail only on ROCm. This is probably
               # related to the fact that the current linalg.det backward is
               # unstable if the matrix has repeated singular values, see
               # https://github.com/pytorch/pytorch/issues/53364
               SkipInfo('TestGradients', 'test_fn_grad', device_type='cuda',
                        dtypes=(torch.float64,), active_if=TEST_WITH_ROCM),
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda',
                        dtypes=(torch.float64,), active_if=TEST_WITH_ROCM),
               SkipInfo('TestJit', 'test_variant_consistency_jit', device_type='cuda',
                        dtypes=(torch.float64, torch.float32), active_if=TEST_WITH_ROCM),
           )),
    OpInfo('linalg.cholesky',
           aten_name='linalg_cholesky',
           dtypes=floating_and_complex_types(),
           # TODO: RuntimeError: While computing batched gradients,
           # got: vmap: Calling Tensor.as_strided is not supported
           # unless the batch dims being vmapped over are at the front of the tensor (in memory layout).
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # Gradcheck for complex generates invalid inputs for this function
               SkipInfo('TestGradients', 'test_forward_mode_AD', dtypes=complex_types()),)
           ),
    OpInfo('linalg.cholesky_ex',
           aten_name='linalg_cholesky_ex',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.cond',
           aten_name='linalg_cond',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_cond,
           check_batched_gradgrad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           ),
    OpInfo('linalg.eig',
           aten_name='linalg_eig',
           op=torch.linalg.eig,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_eig,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.eigvals',
           aten_name='linalg_eigvals',
           op=torch.linalg.eigvals,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.eigh',
           aten_name='linalg_eigh',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_eigh,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.eigvalsh',
           aten_name='linalg_eigvalsh',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_eigh,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],),
    OpInfo('linalg.householder_product',
           aten_name='linalg_householder_product',
           op=torch.linalg.householder_product,
           aliases=('orgqr', ),
           dtypes=floating_and_complex_types(),
           # TODO: backward uses in-place operations that vmap doesn't like
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_householder_product,
           decorators=[skipCUDAIfNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.lstsq',
           aten_name='linalg_lstsq',
           op=torch.linalg.lstsq,
           dtypes=floating_and_complex_types(),
           supports_out=True,
           sample_inputs_func=sample_inputs_linalg_lstsq,
           supports_autograd=False,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           skips=(
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('linalg.matrix_power',
           aliases=('matrix_power',),
           aten_name='linalg_matrix_power',
           dtypes=floating_and_complex_types(),
           supports_inplace_autograd=False,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, skipCUDAIfRocm],
           sample_inputs_func=sample_inputs_linalg_matrix_power,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('linalg.multi_dot',
           # Need this lambda because gradcheck does not work with TensorList inputs
           aten_name='linalg_multi_dot',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, *[torch.bfloat16] if CUDA11OrLater else []),
           supports_inplace_autograd=False,
           # Batched grad checks fail for empty input tensors (see https://github.com/pytorch/pytorch/issues/53407)
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_multi_dot,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           ),
    OpInfo('linalg.norm',
           op=torch.linalg.norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_norm,
           aten_name='linalg_norm',
           skips=(
               # linalg.norm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('linalg.matrix_norm',
           aten_name='linalg_matrix_norm',
           dtypes=floating_and_complex_types(),
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_matrix_norm,
           skips=(
               # linalg.matrix_norm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('linalg.qr',
           aten_name='linalg_qr',
           op=torch.linalg.qr,
           dtypes=floating_and_complex_types(),
           # batched gradients do not work for empty inputs
           # https://github.com/pytorch/pytorch/issues/50743#issuecomment-767376085
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_qr,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.slogdet',
           aten_name='linalg_slogdet',
           op=torch.linalg.slogdet,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_slogdet,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.vector_norm',
           op=torch.linalg.vector_norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_vector_norm,
           aten_name='linalg_vector_norm',
           skips=(
               # linalg.vector_norm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    UnaryUfuncInfo('log',
                   ref=np.log,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('log10',
                   ref=np.log10,
                   domain=(0, float('inf')),
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   assert_autodiffed=True,
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('log1p',
                   ref=np.log1p,
                   aliases=('special.log1p',),
                   domain=(-1, float('inf')),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   assert_autodiffed=True),
    UnaryUfuncInfo('log2',
                   ref=np.log2,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.cfloat, torch.cdouble]),
                   )),
    OpInfo('logaddexp',
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.bfloat16),
           dtypesIfROCM=floating_types_and(torch.bfloat16),
           supports_forward_ad=True,
           sample_inputs_func=lambda op_info, device, dtype, requires_grad=False, **kwargs:
           (SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad),
                        args=(make_tensor((S, S), device, dtype, requires_grad=requires_grad),)),)),
    OpInfo('logaddexp2',
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.bfloat16),
           dtypesIfROCM=floating_types_and(torch.bfloat16),
           supports_forward_ad=True,
           sample_inputs_func=lambda op_info, device, dtype, requires_grad=False, **kwargs:
           (SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad),
                        args=(make_tensor((S, S), device, dtype, requires_grad=requires_grad),)),)),
    UnaryUfuncInfo('logical_not',
                   ref=np.logical_not,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 5e-1}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   supports_autograd=False,
                   skips=(
                       # The function variant always returns BoolTensor
                       # while the inplace variant preserves the input dtype.
                       # >>> t = torch.randn(3)
                       # >>> torch.logical_not(t)
                       # tensor([False, False, False])
                       # >>> torch.logical_not(t).dtype
                       # torch.bool
                       # >>> t.logical_not_().dtype
                       # torch.float32
                       SkipInfo('TestUnaryUfuncs', 'test_variant_consistency',
                                dtypes=all_types_and_complex_and(torch.half, torch.bfloat16)),
                       SkipInfo('TestCommon', 'test_variant_consistency_eager',
                                dtypes=all_types_and_complex_and(torch.half, torch.bfloat16)),
                   )),
    OpInfo('lt',
           aliases=('less',),
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('lu',
           op=torch.lu,
           dtypes=floating_and_complex_types(),
           supports_inplace_autograd=False,
           check_batched_gradgrad=False,
           supports_out=False,
           sample_inputs_func=sample_inputs_lu,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # we skip jit tests because lu_backward is impelemented as autograd.Function,
               # which does not support autograd with scripting
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
               # Skip operator schema test because this is a functional and not an operator
               SkipInfo('TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('lu_unpack',
           op=torch.lu_unpack,
           dtypes=floating_and_complex_types(),
           supports_inplace_autograd=False,
           # we use in-place operations which cannot be avoided.
           # This cases vmap failures, hence we skip batched gradient checks
           check_batched_grad=False,
           supports_out=True,
           sample_inputs_func=sample_inputs_lu_unpack,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),
           )),
    OpInfo('masked_fill',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_fill,
           supports_forward_ad=True,
           supports_out=False),
    OpInfo('masked_scatter',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_scatter,
           supports_forward_ad=True,
           supports_out=False),
    OpInfo('masked_select',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_masked_select),
    OpInfo('matrix_exp',
           dtypesIfCPU=floating_and_complex_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_matrix_exp,
           supports_out=False,
           ),
    OpInfo('matmul',
           dtypes=floating_types(),
           dtypesIfCPU=all_types_and_complex(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half, torch.bfloat16),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_matmul,
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+
               #   and SM53+
               SkipInfo('TestCommon', 'test_dtypes', active_if=IS_WINDOWS),
               # matmul does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
               SkipInfo('TestCommon', 'test_conj_view', device_type='cpu'),
           )),
    OpInfo('max',
           op=torch.max,
           variant_test_name='binary',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_binary,
           supports_forward_ad=True,
           assert_autodiffed=True,),
    OpInfo('max',
           op=torch.max,
           variant_test_name='reduction_with_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_reduction_with_dim,
           supports_forward_ad=True,
           skips=(
               # max does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),)),
    OpInfo('max',
           op=torch.max,
           variant_test_name='reduction_no_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_max_min_reduction_no_dim,),
    OpInfo('median',
           dtypes=all_types(),
           dtypesIfCPU=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16),
           # TODO: some signatures of median do support out
           supports_out=False,
           sample_inputs_func=sample_inputs_reduction_wrapper(False)),
    OpInfo('nanmedian',
           dtypes=all_types(),
           dtypesIfCPU=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16),
           # TODO: some signatures of nanmedian do support out
           supports_out=False,
           sample_inputs_func=sample_inputs_reduction_wrapper(False)),
    OpInfo('var_mean',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_reduction_wrapper(False),
           backward_dtypes=floating_types_and(torch.half),
           backward_dtypesIfCUDA=floating_types_and(torch.half),
           # TODO: some signatures of var_mean do support out
           supports_out=False,
           supports_forward_ad=True,
           skips=(
               # TODO: FIXME: complex inputs requiring grad error in forward
               SkipInfo('TestCommon', 'test_dtypes'),
               # TODO: review with var_mean tests in test_autograd.py
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
               SkipInfo('TestGradients', 'test_fn_grad'),
               SkipInfo('TestGradients', 'test_fn_gradgrad'),
               SkipInfo('TestGradients', 'test_forward_mode_AD'))),
    OpInfo('std_mean',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_reduction_wrapper(False),
           backward_dtypes=floating_types_and(torch.half),
           backward_dtypesIfCUDA=floating_types_and(torch.half),
           # TODO: some signatures of std_mean do support out
           supports_out=False,
           supports_forward_ad=True,
           skips=(
               # TODO: FIXME: complex inputs requiring grad error in forward
               SkipInfo('TestCommon', 'test_dtypes'),
               # TODO: fix along with var_mean autograd tests
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
               SkipInfo('TestGradients', 'test_fn_grad'),
               SkipInfo('TestGradients', 'test_fn_gradgrad'),
               SkipInfo('TestGradients', 'test_forward_mode_AD'))),
    OpInfo('min',
           op=torch.min,
           variant_test_name='binary',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_binary,
           supports_forward_ad=True,
           assert_autodiffed=True,),
    OpInfo('min',
           op=torch.min,
           variant_test_name='reduction_with_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_reduction_with_dim,
           supports_forward_ad=True,
           skips=(
               # min does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('min',
           op=torch.min,
           variant_test_name='reduction_no_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_max_min_reduction_no_dim,),
    OpInfo('sum',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool),
           supports_out=False,
           sample_inputs_func=sample_inputs_reduction_wrapper(supports_multiple_dims=True)),
    OpInfo('nansum',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           dtypesIfCPU=all_types_and(torch.float16, torch.bool),
           supports_out=False,
           sample_inputs_func=sample_inputs_reduction_wrapper(supports_multiple_dims=True)),
    # TODO(@heitorschueroff) Add test for dtype kwarg
    OpInfo('mean',
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           assert_autodiffed=True,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_reduction_wrapper(supports_multiple_dims=True),
           # Need to skip out test because one of the overload for mean does not support it
           # TODO(@heitorschueroff) fix this when implementing ReductionInfo
           skips=(SkipInfo('TestCommon', 'test_out'),)),
    OpInfo('quantile',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_reduction_quantile),
    OpInfo('nanquantile',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_reduction_quantile),
    OpInfo('maximum',
           op=torch.maximum,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_max_min_binary,),
    OpInfo('minimum',
           op=torch.minimum,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_max_min_binary,),
    OpInfo('nn.functional.hardswish',
           aten_name="hardswish",
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_hardswish,
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_gradgrad=False,
           supports_forward_ad=True,
           supports_out=False,
           autodiff_nonfusible_nodes=["aten::hardswish"]),
    OpInfo('nn.functional.leaky_relu',
           aliases=None,
           aten_name="leaky_relu",
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_leaky_relu,
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_autograd=True,
           assert_autodiffed=True,
           supports_gradgrad=True,
           supports_out=False,
           autodiff_nonfusible_nodes=["aten::leaky_relu"]),
    OpInfo('topk',
           dtypes=all_types(),
           dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_topk,
           skips=(
               # Topk is not raising a warning when the out is resized
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('nn.functional.hardshrink',
           aten_name="hardshrink",
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_hardshrink_hardtanh,
           supports_gradgrad=True,
           supports_out=False,
           autodiff_nonfusible_nodes=["aten::hardshrink"]),
    OpInfo('nn.functional.hardtanh',
           aten_name="hardtanh",
           dtypesIfCPU=floating_types_and(torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16),
           backward_dtypesIfCPU=all_types(),
           dtypesIfCUDA=floating_types_and(torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(torch.float16),
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_hardshrink_hardtanh,
           supports_gradgrad=True,
           supports_out=False,
           autodiff_nonfusible_nodes=["aten::hardtanh"],
           ),
    OpInfo('nn.functional.gelu',
           aten_name="gelu",
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_gelu,
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_gradgrad=True,
           supports_out=False,
           autodiff_nonfusible_nodes=["aten::gelu"]),
    OpInfo('nn.functional.relu6',
           aten_name="relu6",
           dtypes=all_types(),
           dtypesIfCPU=all_types_and(torch.bfloat16),
           backward_dtypesIfCPU=floating_types(),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(torch.float16),
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_hardshrink_hardtanh,
           supports_gradgrad=True,
           supports_out=False,
           autodiff_nonfusible_nodes=["aten::relu6"]),
    OpInfo('mm',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_mm,
           skips=(
               # mm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('mode',
           op=torch.mode,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_mode,),
    MvlGammaInfo(variant_test_name='mvlgamma_p_1',
                 domain=(1, float('inf')),
                 skips=skips_mvlgamma(),
                 sample_kwargs=lambda device, dtype, input: ({'p': 1}, {'d': 1})),
    MvlGammaInfo(variant_test_name='mvlgamma_p_3',
                 domain=(2, float('inf')),
                 skips=skips_mvlgamma(skip_redundant=True) + (
                     SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard', dtypes=(torch.float16,)),
                 ),
                 sample_kwargs=lambda device, dtype, input: ({'p': 3}, {'d': 3})),
    MvlGammaInfo(variant_test_name='mvlgamma_p_5',
                 domain=(3, float('inf')),
                 skips=skips_mvlgamma(skip_redundant=True) + (
                     SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard', dtypes=(torch.float16,)),
                 ),
                 sample_kwargs=lambda device, dtype, input: ({'p': 5}, {'d': 5})),
    OpInfo('ne',
           aliases=('not_equal',),
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('narrow',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_narrow),
    UnaryUfuncInfo('neg',
                   aliases=('negative', ),
                   ref=np.negative,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   assert_autodiffed=True,),
    OpInfo('dist',
           op=torch.dist,
           dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_dist,
           skips=(
               # dist does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('outer',
           op=torch.outer,
           aliases=('ger', ),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_outer,),
    OpInfo('ormqr',
           op=torch.ormqr,
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_ormqr,
           decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack]),
    OpInfo('permute',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_permute),
    OpInfo('pow',
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
           # Due to AVX2 curently not being fully supported for Float16, log_vml_cpu can't be enabled
           # for Float16, causing this test to fail. pow's autograd for Float16 is thus currently
           # unsupported on CPU.
           backward_dtypes=all_types_and_complex_and(torch.bfloat16, torch.bool),
           backward_dtypesIfCUDA=all_types_and_complex_and(torch.bfloat16, torch.half),
           sample_inputs_func=sample_inputs_pow,
           supports_inplace_autograd=False,
           assert_autodiffed=True,
           ),
    OpInfo('float_power',
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
           backward_dtypes=all_types_and_complex_and(torch.bfloat16, torch.bool),
           backward_dtypesIfCUDA=all_types_and_complex_and(torch.bfloat16, torch.half),
           sample_inputs_func=sample_inputs_pow,
           supports_inplace_autograd=False,
           skips=(
               SkipInfo('TestCommon', 'test_conj_view', device_type='cuda'),),),
    OpInfo('prod',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           skips=(
               # prod does not support the (Tensor, *, out) overload
               SkipInfo('TestCommon', 'test_out',
                        dtypes=[torch.float32]),
           ),
           sample_inputs_func=sample_inputs_prod,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('qr',
           op=torch.qr,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_qr,
           # batched gradients do not work for empty inputs
           # https://github.com/pytorch/pytorch/issues/50743#issuecomment-767376085
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    UnaryUfuncInfo('rad2deg',
                   ref=np.degrees,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/51283#issuecomment-770614273
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16]),
                   ),
                   safe_casts_outputs=True),
    UnaryUfuncInfo('real',
                   ref=np.real,
                   dtypes=complex_types(),
                   supports_out=False,
                   skips=(
                       # Skip since real and imag don't have out variants.
                       SkipInfo('TestUnaryUfuncs', 'test_out_arg_all_dtypes'),
                   )),
    OpInfo('roll',
           ref=np.roll,
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           sample_inputs_func=sample_inputs_roll),
    OpInfo('rot90',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           sample_inputs_func=sample_inputs_rot90),
    UnaryUfuncInfo('round',
                   ref=np.round,
                   aliases=('special.round',),
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   assert_autodiffed=True,),
    UnaryUfuncInfo('sin',
                   ref=np.sin,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),)),
    UnaryUfuncInfo('sinc',
                   ref=np_sinc_with_fp16_as_fp32,
                   aliases=('special.sinc',),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2,
                                                  torch.float16: 1e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/49133
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.cfloat]),
                   )),
    UnaryUfuncInfo('sinh',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.sinh),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   decorators=(precisionOverride({torch.float16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       # Reference: https://github.com/pytorch/pytorch/issues/48641
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.int8]),
                   )),
    UnaryUfuncInfo('sign',
                   ref=reference_sign,
                   dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/41245
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),
                   )),
    UnaryUfuncInfo('sgn',
                   ref=reference_sgn,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/41245
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),
                       # Reference: https://github.com/pytorch/pytorch/issues/53958
                       # Test fails in comparison on Nan as the `equal_nan` is True for
                       # comparing the CPU tensors.
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.complex64, torch.complex128]),
                       # Reference: https://github.com/pytorch/pytorch/issues/48486
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.complex64])
                   )),
    OpInfo('split',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=partial(sample_inputs_split, list_args=False),
           supports_out=False,
           assert_autodiffed=True),
    OpInfo('split',
           variant_test_name='list_args',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=partial(sample_inputs_split, list_args=True),
           supports_out=False),
    OpInfo('split_with_sizes',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_split_with_sizes,
           supports_out=False,
           assert_autodiffed=True),
    OpInfo('__radd__',
           op=torch.Tensor.__radd__,
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_rbinops,
           supports_out=False,
           skips=(SkipInfo('TestJit', 'test_variant_consistency_jit',),),
           assert_autodiffed=True,
           supports_forward_ad=True,
           autodiff_nonfusible_nodes=['aten::add'],),
    OpInfo('__rdiv__',
           op=torch.Tensor.__rdiv__,
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_rbinops,
           supports_out=False,
           skips=(SkipInfo('TestJit', 'test_variant_consistency_jit',),),
           assert_autodiffed=True,
           autodiff_nonfusible_nodes=['aten::mul', 'aten::reciprocal'],),
    OpInfo('__rmul__',
           op=torch.Tensor.__rmul__,
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_rbinops,
           supports_out=False,
           skips=(SkipInfo('TestJit', 'test_variant_consistency_jit',),),
           assert_autodiffed=True,
           supports_forward_ad=True,
           autodiff_nonfusible_nodes=['aten::mul'],),
    OpInfo('__rmatmul__',
           op=torch.Tensor.__rmatmul__,
           dtypes=floating_types(),
           dtypesIfCPU=all_types_and_complex(),
           dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else [],
                                           torch.complex64, torch.complex128),
           backward_dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_matmul,
           supports_out=False,
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+
               #   and SM53+
               SkipInfo('TestCommon', 'test_dtypes', active_if=IS_WINDOWS),
               SkipInfo('TestJit', 'test_variant_consistency_jit',),
           )),
    OpInfo('__rmod__',
           op=torch.Tensor.__rmod__,
           dtypes=all_types_and(torch.bfloat16, torch.half),
           dtypesIfCPU=floating_types_and(torch.half,),
           dtypesIfCUDA=all_types_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_rbinops,
           supports_out=False,
           skips=(SkipInfo('TestJit', 'test_variant_consistency_jit',),),
           # Support autograd after torch.remainder(Tensor, Tensor) supports
           # autograd of the second argument.
           # https://github.com/pytorch/pytorch/pull/58476/files#r637167630
           supports_autograd=False,
           assert_autodiffed=True,
           autodiff_nonfusible_nodes=['aten::remainder'],),
    OpInfo('__rpow__',
           op=torch.Tensor.__rpow__,
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           # Reference: https://github.com/pytorch/pytorch/issues/54774
           # "log2" "_vml_cpu" not implemented for Half
           backward_dtypesIfCPU=all_types_and_complex_and(torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_rbinops,
           supports_out=False,
           skips=(
               SkipInfo('TestJit', 'test_variant_consistency_jit',),),
           assert_autodiffed=True,
           autodiff_nonfusible_nodes=['aten::pow'],),
    OpInfo('__rsub__',
           op=torch.Tensor.__rsub__,
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
           sample_inputs_func=sample_inputs_rbinops,
           supports_out=False,
           skips=(SkipInfo('TestJit', 'test_variant_consistency_jit',),),
           assert_autodiffed=True,
           autodiff_nonfusible_nodes=['aten::rsub'],),
    OpInfo('rsub',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
           variant_test_name='rsub_tensor',
           supports_out=False,
           supports_inplace_autograd=False,
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/53797
               # JIT doesn't understand complex literals
               SkipInfo('TestJit', 'test_variant_consistency_jit',
                        dtypes=[torch.cfloat, torch.cdouble]),
           ),
           sample_inputs_func=partial(sample_inputs_rsub, variant='tensor'),),
    OpInfo('rsub',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
           variant_test_name='rsub_scalar',
           supports_out=False,
           supports_inplace_autograd=False,
           sample_inputs_func=partial(sample_inputs_rsub, variant='scalar'),
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/53797
               # JIT doesn't understand complex literals
               SkipInfo('TestJit', 'test_variant_consistency_jit',
                        dtypes=all_types_and_complex_and(torch.bfloat16, torch.half)),),
           assert_autodiffed=True,),
    OpInfo('select',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_select,
           supports_out=False),
    UnaryUfuncInfo('signbit',
                   ref=np.signbit,
                   dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   supports_autograd=False,),
    OpInfo('solve',
           op=torch.solve,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_legacy_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('std',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           backward_dtypesIfCPU=floating_and_complex_types_and(torch.half),
           sample_inputs_func=sample_inputs_std_var,
           # TODO: std does support out in some signatures
           supports_out=False,
           assert_autodiffed=True,
           ),
    UnaryUfuncInfo('tan',
                   ref=np.tan,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.float64],
                                active_if=TEST_WITH_ROCM),
                   )),
    UnaryUfuncInfo('tanh',
                   ref=np.tanh,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "tanh_backward_cpu" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                   )),
    OpInfo('tensor_split',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_tensor_split,),
    OpInfo('hsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_hsplit,),
    OpInfo('vsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_vsplit,),
    OpInfo('dsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_dsplit,),
    OpInfo('triangular_solve',
           op=torch.triangular_solve,
           dtypes=floating_and_complex_types(),
           supports_out=False,
           sample_inputs_func=sample_inputs_legacy_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    UnaryUfuncInfo('trunc',
                   aliases=('fix', ),
                   ref=np.trunc,
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
                   assert_autodiffed=True),
    UnaryUfuncInfo('exp2',
                   aliases=('special.exp2', ),
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.exp2),
                   dtypes=all_types_and(torch.bool, torch.half),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('expm1',
                   aliases=('special.expm1', ),
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.expm1),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/48926#issuecomment-739734774
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                   )),
    UnaryUfuncInfo('nan_to_num',
                   ref=np.nan_to_num,
                   dtypes=all_types_and(torch.half, torch.bool),
                   dtypesIfCUDA=all_types_and(torch.half, torch.bool, torch.bfloat16),
                   # Passing numpy_kwargs via sample_kwargs, as numpy does comparison
                   # with BFloat16 in float, since it currently doesn't support BFloat16.
                   # Ref: https://github.com/pytorch/pytorch/issues/57982#issuecomment-839150556
                   sample_kwargs=lambda device, dtype, input: ({},
                                                               {'posinf': torch.finfo(torch.bfloat16).max,
                                                                'neginf': torch.finfo(torch.bfloat16).min})
                   if dtype is torch.bfloat16 else ({}, {})),
    UnaryUfuncInfo('reciprocal',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.reciprocal),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/45690
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.cfloat, torch.cdouble]),
                       # Reference: https://github.com/pytorch/pytorch/pull/49102#issuecomment-744604601
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.bfloat16]),
                   )),
    UnaryUfuncInfo('rsqrt',
                   ref=lambda x: np.reciprocal(np.sqrt(x)),
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.half: 5e-2}),),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   handles_complex_extremals=False),
    UnaryUfuncInfo('sqrt',
                   ref=np.sqrt,
                   supports_sparse=True,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   decorators=(precisionOverride({torch.bfloat16: 7e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/47358
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_MACOS),
                       # Reference: https://github.com/pytorch/pytorch/pull/47293#issuecomment-721774436
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16])),
                   safe_casts_outputs=True,
                   handles_complex_extremals=False),
    UnaryUfuncInfo('square',
                   ref=np.square,
                   dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                   decorators=(precisionOverride({torch.complex64: 3e-4, torch.bfloat16: 3e-1}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/52549
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.cfloat, torch.cdouble]),
                       # >>> t = torch.tensor(complex(-0.01, float("inf")))
                       # >>> np.square(t.numpy())
                       # (-inf-infj)
                       # >>> t.square()
                       # tensor(-inf-infj)
                       # >>> t.cuda().square()
                       # tensor(inf+nanj, device='cuda:0')
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]),
                       # Reference: https://github.com/pytorch/pytorch/pull/52551#issuecomment-782596181
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16]),
                   ),),
    OpInfo('lerp',
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_lerp,
           supports_forward_ad=True,
           assert_autodiffed=True),
    OpInfo('linalg.inv',
           aten_name='linalg_inv',
           op=torch.linalg.inv,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_invertible,
           check_batched_gradgrad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           ),
    OpInfo('linalg.inv_ex',
           aten_name='linalg_inv_ex',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_invertible,
           check_batched_gradgrad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           ),
    UnaryUfuncInfo('angle',
                   ref=np.angle,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_complex_to_float=True),
    OpInfo('linalg.solve',
           aten_name='linalg_solve',
           op=torch.linalg.solve,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.matrix_rank',
           aten_name='linalg_matrix_rank',
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.matrix_rank',
           aten_name='linalg_matrix_rank',
           variant_test_name='hermitian',
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.pinv',
           aten_name='linalg_pinv',
           op=torch.linalg.pinv,
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.pinv',
           aten_name='linalg_pinv',
           variant_test_name='hermitian',
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('eig',
           op=torch.eig,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_eig,
           decorators=[
               skipCUDAIfNoMagma,
               skipCPUIfNoLapack,
               skipCUDAIfRocm
           ],),
    OpInfo('einsum',
           # we need this lambda because SampleInput expects tensor input as the first argument
           # TODO(@heitorschueroff) update SampleInput to handle such cases
           op=lambda tensors, equation: torch.einsum(equation, tensors),
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.half),
           supports_out=False,
           sample_inputs_func=sample_inputs_einsum,
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+
               #   and SM53+
               SkipInfo('TestCommon', 'test_dtypes', active_if=IS_WINDOWS),
               # test does not work with passing lambda for op
               # there's a test `test_einsum` in `test_jit.py` to handle this case
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('svd',
           op=torch.svd,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_svd,
           decorators=[
               skipCUDAIfNoMagmaAndNoCusolver,
               skipCUDAIfRocm,
               skipCPUIfNoLapack,
           ]),
    OpInfo('linalg.svd',
           op=torch.linalg.svd,
           aten_name='linalg_svd',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_svd,
           decorators=[
               skipCUDAIfNoMagmaAndNoCusolver,
               skipCUDAIfRocm,
               skipCPUIfNoLapack,
           ]),
    OpInfo('linalg.svdvals',
           op=torch.linalg.svdvals,
           aten_name='linalg_svdvals',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_svdvals,
           check_batched_gradgrad=False,
           decorators=[
               skipCUDAIfNoMagmaAndNoCusolver,
               skipCPUIfNoLapack]),
    OpInfo('polar',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_polar),
    # TODO(@kshitij12345): Refactor similar to `mvlgamma` entries.
    # To test reference numerics against multiple values of argument `n`,
    # we make multiple OpInfo entries with each entry corresponding to different value of n (currently 0 to 4).
    # We run the op tests from test_ops.py only for `n=0` to avoid redundancy in testing.
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_0',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Probably related to the way the function is
                       # scripted for JIT tests (or maybe not).
                       # RuntimeError:
                       # Arguments for call are not valid.
                       # The following variants are available:
                       #   aten::polygamma(int n, Tensor self) -> (Tensor):
                       #   Expected a value of type 'Tensor' for argument 'self' but instead found type 'int'.
                       #   aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> (Tensor(a!)):
                       #   Expected a value of type 'Tensor' for argument 'self' but instead found type 'int'.
                       # The original call is:
                       #   File "<string>", line 3
                       # def the_method(i0):
                       #     return torch.polygamma(i0, 1)
                       #            ~~~~~~~~~~~~~~~ <--- HERE
                       SkipInfo('TestJit', 'test_variant_consistency_jit'),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 0}, {'n': 0})),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_1',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       SkipInfo('TestGradients'),
                       SkipInfo('TestJit'),
                       SkipInfo('TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal'),
                   ),
                   sample_kwargs=lambda device, dtype, input: ({'n': 1}, {'n': 1})),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_2',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       SkipInfo('TestGradients'),
                       SkipInfo('TestJit'),
                       SkipInfo('TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=TEST_WITH_ROCM),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 2}, {'n': 2})),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_3',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       SkipInfo('TestGradients'),
                       SkipInfo('TestJit'),
                       SkipInfo('TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=TEST_WITH_ROCM),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 3}, {'n': 3})),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_4',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.float16: 5e-4, torch.float32: 5e-4}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       SkipInfo('TestGradients'),
                       SkipInfo('TestJit'),
                       SkipInfo('TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=TEST_WITH_ROCM),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 4}, {'n': 4})),
    OpInfo('ravel',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_ravel,
           ),
    OpInfo('reshape',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_view_reshape,
           supports_out=False,
           ),
    OpInfo('reshape_as',
           op=lambda x, other: x.reshape_as(other),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_view_as_reshape_as,
           skips=(
               # Because reshape_as does not have a function variant.
               SkipInfo('TestJit', 'test_variant_consistency_jit'),),
           supports_out=False,
           ),
    OpInfo('view',
           op=lambda x, shape: x.view(shape),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           skips=(
               # Because view does not have a function variant.
               SkipInfo('TestJit', 'test_variant_consistency_jit'),),
           sample_inputs_func=sample_inputs_view_reshape,
           ),
    OpInfo('view_as',
           op=lambda x, other: x.view_as(other),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           skips=(
               # Because view_as does not have a function variant.
               SkipInfo('TestJit', 'test_variant_consistency_jit'),),
           sample_inputs_func=sample_inputs_view_as_reshape_as,
           ),
    OpInfo('pinverse',
           op=torch.pinverse,
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_out=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('gather',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_gather,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_forward_ad=True,
           ),
    OpInfo('index_fill',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_index_fill),
    OpInfo('index_copy',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_index_copy,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('index_select',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_index_select,
           supports_forward_ad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('index_add',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_index_add,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('__getitem__',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_inplace_autograd=False,
           op=torch.Tensor.__getitem__,
           sample_inputs_func=sample_inputs_getitem,
           skips=(SkipInfo('TestJit', 'test_variant_consistency_jit'),)),
    OpInfo('index_put',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_inplace_autograd=True,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_index_put,
           skips=(
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('sort',
           dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=all_types_and(torch.float16),
           sample_inputs_func=sample_inputs_sort,
           skips=(
               # sort does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('put',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           check_batched_gradgrad=False,  # vmap complains of the sizes
           sample_inputs_func=sample_inputs_put),
    OpInfo('take',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           check_batched_grad=False,  # vmap complains of the sizes
           sample_inputs_func=sample_inputs_take),
    OpInfo('scatter',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_scatter,),
    OpInfo('scatter_add',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_scatter_add,
           supports_out=False),
    OpInfo('stack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_stack,
           assert_autodiffed=True,
           skips=(
               # stack does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),),),
    OpInfo('hstack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           supports_forward_ad=True,
           skips=(
               # hstack does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),),),
    OpInfo('hypot',
           dtypes=floating_types(),
           dtypesIfCPU=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_hypot,
           ),
    OpInfo('histogram',
           dtypes=_dispatch_dtypes(),  # histogram is only implemented on CPU
           dtypesIfCPU=floating_types(),
           sample_inputs_func=sample_inputs_histogram,
           supports_autograd=False,
           skips=(
               # JIT tests don't work with Tensor keyword arguments
               # https://github.com/pytorch/pytorch/issues/58507
               SkipInfo('TestJit', 'test_variant_consistency_jit'),),),
    OpInfo('vstack',
           aliases=('row_stack',),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           supports_forward_ad=True,
           skips=(
               # vstack does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
               # RuntimeError: _fn() Expected a value of type
               #   'Tensor (inferred)' for argument 't0' but instead found type 'tuple'.
               SkipInfo('TestJit', 'test_jit_alias_remapping'))),
    OpInfo('dstack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           skips=(
               # dstack does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),)),
    OpInfo('unfold',
           op=lambda x, *args: x.unfold(*args),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           check_batched_gradgrad=False,
           skips=(
               # torch.unfold does not exist so we get a RuntimeError.
               SkipInfo('TestJit', 'test_variant_consistency_jit',
                        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
               # Skip operator schema test because this is a functional and not an operator
               SkipInfo('TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           ),
           sample_inputs_func=sample_inputs_unfold),
    OpInfo('msort',
           dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=all_types_and(torch.float16),
           check_batched_gradgrad=False,
           skips=(
               #  msort does not correctly warn when resizing out= inputs.
               SkipInfo('TestCommon', 'test_out',
                        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
           ),
           sample_inputs_func=sample_inputs_msort),
    OpInfo('movedim',
           aliases=('moveaxis',),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_movedim_moveaxis,
           skips=(
               # Expected a value of type 'int' for argument 'source'
               #   but instead found type 'list'.
               SkipInfo('TestJit', 'test_jit_alias_remapping'),
           )),
    OpInfo('renorm',
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_renorm),
    ShapeFuncInfo('repeat',
                  op=lambda x, dims: x.repeat(dims),
                  ref=np.tile,
                  dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                  supports_out=False,
                  skips=(
                      # torch.repeat does not exist so we get a RuntimeError.
                      SkipInfo('TestJit', 'test_variant_consistency_jit',
                               dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
                  ),
                  sample_inputs_func=sample_repeat_tile),
    OpInfo('squeeze',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_squeeze),
    OpInfo('fill_',
           op=lambda x, scalar: torch.fill_(x.clone(), scalar),
           method_variant=None,
           inplace_variant=torch.Tensor.fill_,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           skips=(
               # JIT has issue when op is passed as lambda
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           sample_inputs_func=sample_inputs_fill_),
    OpInfo('resize_',
           op=lambda x, shape: x.clone().resize_(shape),
           method_variant=None,
           inplace_variant=None,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_autograd=False,
           skips=(
               # JIT has issue when op is passed as lambda
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           sample_inputs_func=sample_inputs_resize_ops),
    OpInfo('resize_as_',
           op=lambda x, other: torch.resize_as_(x.clone(), other),
           method_variant=None,
           inplace_variant=torch.Tensor.resize_as_,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_autograd=False,
           skips=(
               # JIT has issue when op is passed as lambda
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           sample_inputs_func=sample_inputs_resize_ops),
    OpInfo('take_along_dim',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           sample_inputs_func=sample_inputs_take_along_dim,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    ShapeFuncInfo('tile',
                  ref=np.tile,
                  dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                  supports_out=False,
                  sample_inputs_func=sample_repeat_tile),
    OpInfo('unsqueeze',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           assert_autodiffed=True,
           sample_inputs_func=sample_unsqueeze),
    OpInfo('var',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           backward_dtypesIfCPU=floating_and_complex_types_and(torch.half),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_std_var,
           # TODO: revisit, some var signatures do support out (see std, too)
           supports_out=False,
           assert_autodiffed=True,
           ),
    OpInfo('xlogy',
           dtypes=all_types_and(torch.bool),
           dtypesIfCPU=all_types_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
           supports_inplace_autograd=True,
           supports_forward_ad=True,
           safe_casts_outputs=True,
           sample_inputs_func=sample_inputs_xlogy),
    OpInfo('zero_',
           op=lambda x: torch.zero_(x.clone()),
           method_variant=None,
           inplace_variant=torch.Tensor.zero_,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           skips=(
               # JIT has issue when op is passed as lambda
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           sample_inputs_func=sample_inputs_zero_),
    OpInfo('special.xlog1py',
           aten_name='special_xlog1py',
           dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
           backward_dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
           safe_casts_outputs=True,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_xlog1py),
    OpInfo('special.zeta',
           aten_name='special_zeta',
           dtypes=all_types_and(torch.bool),
           supports_autograd=False,
           safe_casts_outputs=True,
           sample_inputs_func=sample_inputs_binary_pwise),
    # OpInfo entry to verify the gradient formula of `other`/`q`
    OpInfo('special.zeta',
           op=lambda q, x, **kwargs: torch.special.zeta(x, q, **kwargs),
           aten_name='special_zeta',
           variant_test_name='grad',
           dtypes=all_types_and(torch.bool),
           supports_autograd=True,
           safe_casts_outputs=True,
           skips=(
               # Lambda doesn't work in JIT test
               SkipInfo("TestJit", "test_variant_consistency_jit"),
           ),
           sample_inputs_func=sample_inputs_zeta),
    OpInfo('logsumexp',
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.half),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_logsumexp),
    OpInfo('trace',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_out=False,
           sample_inputs_func=sample_inputs_trace),
    OpInfo('transpose',
           aliases=('swapdims', 'swapaxes'),
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           sample_inputs_func=sample_inputs_transpose_swapdims),
    OpInfo('tril',
           dtypes=all_types_and_complex_and(torch.bool, torch.half),
           sample_inputs_func=sample_inputs_tril_triu),
    OpInfo('triu',
           dtypes=all_types_and_complex_and(torch.bool, torch.half),
           sample_inputs_func=sample_inputs_tril_triu),
    OpInfo('kron',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_inplace_autograd=False,
           sample_inputs_func=sample_inputs_kron),
    OpInfo('inner',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_inner,
           ),
    OpInfo('tensordot',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16),
           safe_casts_outputs=True,
           sample_inputs_func=sample_inputs_tensordot,
           skips=(
               # Currently failing due to an INTERNAL_ASSERT_FAILED error.
               # Reference: https://github.com/pytorch/pytorch/issues/56314
               SkipInfo("TestJit", "test_variant_consistency_jit", dtypes=[torch.float32]),
               # Skip operator schema test because this is a functional and not an operator.
               # Reference: https://github.com/pytorch/pytorch/issues/54574
               SkipInfo('TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )
           ),
    OpInfo('to_sparse',
           op=lambda x, *args: x.to_sparse(*args),
           sample_inputs_func=sample_inputs_to_sparse,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           backward_dtypes=floating_types(),
           backward_dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           skips=(
               # TODO: FIXME: complex inputs requiring grad error in forward
               SkipInfo('TestCommon', 'test_dtypes'),
               # JIT has issue when op is passed as lambda
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           )
           ),
    OpInfo('logcumsumexp',
           dtypes=floating_types_and(),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(),
           skips=(
               # AssertionError: UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               SkipInfo('TestCommon', 'test_out', dtypes=(torch.float32,), device_type='cuda'),
           ),
           sample_inputs_func=sample_inputs_logcumsumexp),
    UnaryUfuncInfo('sigmoid',
                   aliases=('special.expit', ),
                   ref=reference_sigmoid if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.complex64: 1e-1,
                                                  torch.bfloat16: 1e-2}),),
                   skips=(
                       # TODO: FIXME: sigmoid fails on complex inputs that require grad
                       SkipInfo('TestCommon', 'test_dtypes'),
                       # Reference: https://github.com/pytorch/pytorch/issues/56012
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.complex64]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.complex64]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble])),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   assert_autodiffed=True),
    UnaryUfuncInfo('digamma',
                   ref=scipy.special.digamma if TEST_SCIPY else _NOTHING,
                   aliases=('special.psi', 'special.digamma',),
                   decorators=(precisionOverride({torch.float16: 5e-1}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   supports_forward_ad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('special.entr',
                   ref=scipy.special.entr if TEST_SCIPY else _NOTHING,
                   aten_name='special_entr',
                   decorators=(precisionOverride({torch.float16: 1e-1,
                                                  torch.bfloat16: 1e-1}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16, torch.float16]),
                   ),
                   supports_inplace_autograd=False,
                   safe_casts_outputs=True,
                   sample_inputs_func=sample_inputs_entr),
    UnaryUfuncInfo('special.ndtri',
                   ref=scipy.special.ndtri if TEST_SCIPY else _NOTHING,
                   domain=(0, 1),
                   aten_name='special_ndtri',
                   dtypes=all_types_and(torch.bool),
                   safe_casts_outputs=True),
    UnaryUfuncInfo('erf',
                   ref=scipy.special.erf if TEST_SCIPY else _NOTHING,
                   aliases=('special.erf', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('erfc',
                   ref=scipy.special.erfc if TEST_SCIPY else _NOTHING,
                   aliases=('special.erfc', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('erfinv',
                   ref=scipy.special.erfinv if TEST_SCIPY else _NOTHING,
                   aliases=('special.erfinv', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2,
                                                  torch.float32: 1e-4}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   domain=(-1, 1),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/49155#issuecomment-742664611
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                active_if=TEST_SCIPY and distutils.version.LooseVersion(scipy.__version__) < "1.4.0"),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=TEST_SCIPY and distutils.version.LooseVersion(scipy.__version__) < "1.4.0"),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=TEST_SCIPY and distutils.version.LooseVersion(scipy.__version__) < "1.4.0"),
                   )),
    UnaryUfuncInfo('lgamma',
                   ref=reference_lgamma if TEST_SCIPY else _NOTHING,
                   aliases=('special.gammaln', ),
                   decorators=(precisionOverride({torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   # "digamma" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and(torch.bool),
                   supports_forward_ad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/50140#discussion_r552615345
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       # Reference: https://github.com/pytorch/pytorch/pull/50140#issuecomment-756150214
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                   ),
                   safe_casts_outputs=True),
    OpInfo(
        'logdet',
        supports_out=False,
        sample_inputs_func=sample_inputs_logdet,
        decorators=(skipCPUIfNoLapack, skipCUDAIfNoMagma, skipCUDAIfRocm)),
    # `log_softmax` supports different dtypes based on whether `dtype` argument,
    # is passed or not. Hence two OpInfo entries, one with dtype and other without.
    OpInfo(
        'log_softmax',
        supports_out=False,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_log_softmax,
        assert_autodiffed=True),
    OpInfo(
        'log_softmax',
        variant_test_name='dtype',
        supports_out=False,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        sample_inputs_func=partial(sample_inputs_log_softmax, with_dtype=True),
        assert_autodiffed=True),
    UnaryUfuncInfo('logit',
                   ref=scipy.special.logit if TEST_SCIPY else _NOTHING,
                   domain=(0, 1),
                   aliases=('special.logit', ),
                   decorators=(precisionOverride({torch.bfloat16: 5e-1,
                                                  torch.float16: 5e-1}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   sample_inputs_func=sample_inputs_logit,
                   safe_casts_outputs=True),
    OpInfo('where',
           # Currently only the `input` is tested in gradcheck.
           # If we pass `condition` first, none of the input which supports
           # autograd will be tested. Hence the following lambda.
           op=lambda self, condition, other: torch.where(condition, self, other),
           sample_inputs_func=sample_inputs_where,
           supports_out=False,
           skips=(
               # test does not work with passing lambda for op
               SkipInfo('TestJit', 'test_variant_consistency_jit'),
           ),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16)),
    # `torch.norm` has multiple code paths depending on the value of `p`.
    # These paths have different dtype support. Also JIT supports,
    # most variants but not all of them. So we split the OpInfo entries,
    # for `norm` based on the code-paths and JIT support.
    OpInfo('norm',
           sample_inputs_func=sample_inputs_norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           skips=(
               # RuntimeError not raised :
               # Expected RuntimeError when calling with input.device=cpu and out.device=cuda
               SkipInfo('TestCommon', 'test_out'),
           )
           ),
    OpInfo('norm',
           variant_test_name='nuc',
           sample_inputs_func=sample_inputs_norm_nuc,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types(),
           skips=(
               # RuntimeError not raised :
               # Expected RuntimeError when calling with input.device=cpu and out.device=cuda
               SkipInfo('TestCommon', 'test_out'),
               # RuntimeError:
               # Arguments for call are not valid.
               SkipInfo('TestJit', 'test_variant_consistency_jit', dtypes=(torch.complex64,)),
               # RuntimeError: aliasOp != torch::jit::getOperatorAliasMap().end()
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":157,
               # please report a bug to PyTorch.
               SkipInfo('TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),
           )
           ),
    OpInfo('norm',
           variant_test_name='fro',
           sample_inputs_func=sample_inputs_norm_fro,
           dtypes=floating_and_complex_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           skips=(
               # RuntimeError not raised :
               # Expected RuntimeError when calling with input.device=cpu and out.device=cuda
               SkipInfo('TestCommon', 'test_out'),
               # RuntimeError:
               # Arguments for call are not valid.
               SkipInfo('TestJit', 'test_variant_consistency_jit', dtypes=(torch.complex64,)),
               # RuntimeError: aliasOp != torch::jit::getOperatorAliasMap().end()
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":157,
               # please report a bug to PyTorch.
               SkipInfo('TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),
           )
           ),
    OpInfo('norm',
           variant_test_name='inf',
           sample_inputs_func=sample_inputs_norm_inf,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           backward_dtypesIfCPU=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           skips=(
               # following 3 tests failed intermittenly
               SkipInfo('TestJit', 'test_variant_consistency_jit',
                        device_type='cpu', dtypes=(torch.complex64,)),
               SkipInfo('TestGradients', 'test_fn_grad',
                        device_type='cpu', dtypes=(torch.complex128,)),
               SkipInfo('TestGradients', 'test_fn_gradgrad',
                        device_type='cpu', dtypes=(torch.complex128,)),
           )
           ),
    OpInfo('t',
           sample_inputs_func=sample_inputs_t,
           supports_out=False,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           assert_autodiffed=True,),
    UnaryUfuncInfo('special.erfcx',
                   ref=scipy.special.erfcx if TEST_SCIPY else _NOTHING,
                   aten_name='special_erfcx',
                   decorators=(toleranceOverride({torch.float32: tol(atol=0, rtol=4e-6), }),),
                   dtypes=all_types_and(torch.bool),
                   safe_casts_outputs=True),
]

# Common operator groupings
unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo)]
spectral_funcs = [op for op in op_db if isinstance(op, SpectralFuncInfo)]
sparse_unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo) and op.supports_sparse is True]
shape_funcs = [op for op in op_db if isinstance(op, ShapeFuncInfo)]

# TODO: review porting these to make_tensor
def index_variable(shape, max_indices, device=torch.device('cpu')):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape, dtype=torch.double, device=device).mul_(max_indices).floor_().long()
    return index

def gather_variable(shape, index_dim, max_indices, duplicate=False, device=torch.device('cpu')):
    assert len(shape) == 2
    assert index_dim < 2
    batch_dim = 1 - index_dim
    index = torch.zeros(*shape, dtype=torch.long, device=device)
    for i in range(shape[index_dim]):
        index.select(index_dim, i).copy_(
            torch.randperm(max_indices, device=device)[:shape[batch_dim]])
    if duplicate:
        index.select(batch_dim, 0).copy_(index.select(batch_dim, 1))
    return index

def bernoulli_scalar():
    return torch.tensor(0, dtype=torch.bool).bernoulli_()

def mask_not_all_zeros(shape):
    assert len(shape) > 0
    while True:
        result = torch.randn(shape).gt(0)
        if result.sum() > 0:
            return result


# TODO: move all tri/tril/triu testing to tensor creation op test suite and remove
#   these from here
def _compare_trilu_indices(
        self, row, col, offset=0, dtype=torch.long, device='cpu'):
    if row == 0 or col == 0:
        # have to handle this separately as tril and triu does not take
        # empty matrix as input
        self.assertEqual(
            torch.empty(0, 2, dtype=dtype, device=device).transpose(0, 1),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))

        self.assertEqual(
            torch.empty(0, 2, dtype=dtype, device=device).transpose(0, 1),
            torch.triu_indices(row, col, offset, dtype=dtype, device=device))

    else:
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(
            torch.ones(row, col, device='cpu')
                 .tril(offset).nonzero().to(dtype).transpose(0, 1),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))

        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(
            torch.ones(row, col, device='cpu')
                 .tril(offset).nonzero().to(dtype).transpose(0, 1),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))


def _compare_large_trilu_indices(
        self, row, col, offset=0, dtype=torch.long, device='cpu'):
    l = torch.ones(row, col, dtype=dtype, device='cpu').tril(offset) \
             .nonzero()[-100:-1, :].transpose(0, 1).to(device)
    torch.cuda.empty_cache()

    r = torch.tril_indices(
        row, col, offset, dtype=dtype, device=device)[:, -100:-1]
    self.assertEqual(l, r)
    torch.cuda.empty_cache()

    l = torch.ones(row, col, dtype=dtype, device='cpu').triu(offset) \
             .nonzero()[-100:-1, :].transpose(0, 1).to(device)
    torch.cuda.empty_cache()

    r = torch.triu_indices(
        row, col, offset, dtype=dtype, device=device)[:, -100:-1]
    self.assertEqual(l, r)
    torch.cuda.empty_cache()

# (
#   row
#   col
#   offset (optional)
#   dtype (optional)
# )
tri_tests_args = [
    (1, 1),
    (3, 3),
    (3, 3, 1),
    (3, 3, 2),
    (3, 3, 200),
    (3, 3, -1),
    (3, 3, -2),
    (3, 3, -200),
    (0, 3, 0),
    (0, 3, 1),
    (0, 3, -1),
    (3, 0, 0),
    (3, 0, 1),
    (3, 0, -1),
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
    (3, 6, 0),
    (3, 6, 1),
    (3, 6, 3),
    (3, 6, 9),
    (3, 6, -1),
    (3, 6, -3),
    (3, 6, -9),
    (6, 3, 0),
    (6, 3, 1),
    (6, 3, 3),
    (6, 3, 9),
    (6, 3, -1),
    (6, 3, -3),
    (6, 3, -9),
    (258, 253, 1, torch.float32),
    (257, 258, 1, torch.float64),
    (258, 258, 1, torch.short),
    (3, 513, 1, torch.long),
    (513, 3, 1, torch.int),
    (513, 0, 1, torch.double),
    (1024, 1024),
    (1024, 1024, 500, torch.float32),
    (1024, 1024, 1023),
    (1024, 1024, -500),
    (1023, 1025),
    (1025, 1023, 1022),
    (1024, 1024, -500),
    (3, 2028),
    (3, 2028, 1),
    (3, 2028, -1),
    (2028, 3),
    (2028, 1),
    (2028, 1, -1)
]

tri_large_tests_args: List[Tuple[int, ...]] = [
    # Large test cases below are deliberately commented out to speed up CI
    # tests and to avoid OOM error. When modifying implementations of
    # tril_indices and triu_indices, please enable these tests and make sure
    # they pass.
    #
    # (1, 268435455),
    # (5000, 5000),
    # (10000, 10000),
    # (268435455, 1),
    # (134217727, 2, 1),
    # (2, 134217727, 1),
    # (536870901, 1),
    # (1, 536870901),
    # (268435455, 2, 1),
    # (2, 268435455, 1)
]


def run_additional_tri_tests(self, device):
    x = torch.ones(
        3, 3, dtype=torch.long, device=device, layout=torch.strided)
    l = x.tril(0).nonzero().transpose(0, 1)
    u = x.triu(0).nonzero().transpose(0, 1)
    self.assertEqual(l, torch.tril_indices(3, 3, device=device))
    self.assertEqual(
        l, torch.tril_indices(3, 3, device=device, layout=torch.strided))

    self.assertEqual(u, torch.triu_indices(3, 3, device=device))
    self.assertEqual(
        u, torch.triu_indices(3, 3, device=device, layout=torch.strided))

    self.assertRaises(
        RuntimeError,
        lambda: torch.triu_indices(
            1, 1, device=device, layout=torch.sparse_coo))

    self.assertRaises(
        RuntimeError,
        lambda: torch.tril_indices(
            1, 1, device=device, layout=torch.sparse_coo))

# TODO: move into common_utils.py or the test suite(s) that use this
def unpack_variables(args):
    if isinstance(args, tuple):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args


class dont_convert(tuple):
    pass


non_differentiable = collections.namedtuple('non_differentiable', ['tensor'])


# TODO: move into common_utils.py or the test suite(s) that use this
def create_input(call_args, requires_grad=True, non_contiguous=False, call_kwargs=None, dtype=torch.double, device=None):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):
        def maybe_non_contig(tensor):
            return tensor if not non_contiguous else make_non_contiguous(tensor)

        def conjugate(tensor):
            return tensor.conj()

        if isinstance(arg, torch.Size) or isinstance(arg, dont_convert):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            var = conjugate(torch.randn((), dtype=dtype, device=device))
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and not isinstance(arg[0], torch.Tensor):
            return conjugate(maybe_non_contig(torch.randn(*arg, dtype=dtype, device=device))).requires_grad_(requires_grad)
        # double check casting
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                if arg.tensor.dtype == torch.float:
                    return maybe_non_contig(arg.tensor.to(dtype=torch.double, device=device))
                if arg.tensor.dtype == torch.cfloat:
                    return conjugate(maybe_non_contig(arg.tensor.to(dtype=torch.cdouble, device=device)))
                return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
            return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
        elif isinstance(arg, torch.Tensor):
            if arg.dtype == torch.float:
                arg = arg.double()
            if arg.dtype == torch.cfloat:
                arg = arg.to(torch.cdouble)
            if arg.is_complex() != dtype.is_complex:
                raise RuntimeError("User provided tensor is real for a test that runs with complex dtype, ",
                                   "which is not supported for now")
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of v afterwards
            v = conjugate(maybe_non_contig(arg)).detach().to(device=device).clone()
            v.requires_grad = requires_grad and (v.is_floating_point() or v.is_complex())
            return v
        elif callable(arg):
            return map_arg(arg(dtype=dtype, device=device))
        else:
            return arg
    args_out = tuple(map_arg(arg) for arg in call_args)
    kwargs_out = {k: map_arg(v) for k, v in call_kwargs.items()} if call_kwargs else {}
    return args_out, kwargs_out
