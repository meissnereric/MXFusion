# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================


import mxnet as mx
import numpy as np
from mxnet.operator import CustomOp, CustomOpProp
from .util import parse_string_to_tuple

class MakeDiagonalOp(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(MakeDiagonalOp, self).__init__(**kwargs)

    def forward(self, is_train, req, in_data, out_data, aux):
        a = in_data[0]
        n = a.shape[-1]
        if req[0] != 'null':
            if req[0] == 'write':
                b = out_data[0]
            else:
                b = mx.nd.zeros_like(out_data[0])
            index = mx.nd.arange(start=0, stop=n, step=1, dtype=np.int32)
            identity = mx.nd.one_hot(index, depth=n, dtype=a.dtype)
            dim_diff = len(b.shape) - len(identity.shape)
            if dim_diff > 0:
                res_shape = (1,)*dim_diff + identity.shape
                identity = mx.nd.reshape(identity, shape=res_shape)
            mx.nd.broadcast_to(identity, shape=out_data[0].shape, out=b)
            b *= mx.nd.expand_dims(a, axis=-1)
            if req[0] != 'write':
                self.assign(out_data[0], req[0], b)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        b_grad = out_grad[0]
        n = b_grad.shape[-1]
        # Extract diagonal of b_grad
        index = mx.nd.arange(start=0, stop=n, step=1, dtype=np.int32)
        identity = mx.nd.one_hot(index, depth=n, dtype=b_grad.dtype)
        dim_diff = len(b_grad.shape) - len(identity.shape)
        if dim_diff > 0:
            res_shape = (1,)*dim_diff + identity.shape
            identity = mx.nd.reshape(identity, shape=res_shape)
        bindex = mx.nd.broadcast_to(identity, shape=out_data[0].shape)
        a_grad = mx.nd.sum(b_grad*bindex, axis=-1)
        self.assign(in_grad[0], req[0], a_grad)


@mx.operator.register("make_diagonal")
class MakeDiagonalOpProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MakeDiagonalOpProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['a']

    def list_outputs(self):
        return ['b']

    def infer_shape(self, in_shape):
        a_shape = in_shape[0]
        out_shape = a_shape[:-1]+[a_shape[-1], a_shape[-1]]
        return [a_shape], [out_shape], []

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return MakeDiagonalOp(**kwargs)


def make_diagonal(F, x, name="make_diagonal"):
    return F.Custom(x, name=name, op_type="make_diagonal")


class BroadcastToWithSamplesOp(CustomOp):
    def __init__(self, isSamples, shape, **kwargs):
        self.isSamples = isSamples
        self.shape = shape
        super(BroadcastToWithSamplesOp, self).__init__(**kwargs)

    def forward(self, is_train, req, in_data, out_data, aux):
        a = in_data[0]
        n_dim = len(self.shape)
        if self.isSamples:
            t_shape = (a.shape[0],) + (1,) * (n_dim - len(a.shape)) + a.shape[1:]
        else:
            t_shape = (1,) * (n_dim - len(a.shape)) + a.shape

        a_reshape = mx.nd.reshape(a, shape=t_shape)
        out = mx.nd.broadcast_to(a_reshape, shape=self.shape)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        a_shape = in_data[0].shape
        if self.isSamples:
            grad = mx.nd.reshape(out_grad[0], shape=(a_shape[0], -1,) + a_shape[1:])
            a_grad = mx.nd.sum(grad, axis=1)
        else:
            grad = mx.nd.reshape(out_grad[0], shape=(-1,) + a_shape)
            a_grad = mx.nd.sum(grad, axis=0)
        self.assign(in_grad[0], req[0], a_grad)


@mx.operator.register("broadcast_to_w_samples")
class BroadcastToWithSamplesOpProp(CustomOpProp):
    def __init__(self, **kwargs):
        self.isSamples = kwargs['isSamples'].lower() == 'true'
        self.shape = parse_string_to_tuple(kwargs['shape'])
        super(BroadcastToWithSamplesOpProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['out']

    def infer_shape(self, in_shapes):
        return in_shapes, (self.shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes, **kwargs):
        return BroadcastToWithSamplesOp(isSamples=self.isSamples,
                                        shape=self.shape, **kwargs)


def broadcast_to_w_samples(F, data, shape, isSamples=True):
    if F is mx.nd:
        n_dim = len(shape)
        if isSamples:
            num_samples = max(data.shape[0], shape[0])
            t_shape = (data.shape[0],) + (1,) * (n_dim - len(data.shape)) + data.shape[1:]
            shape = (num_samples,) + shape[1:]
        else:
            t_shape = (1,) * (n_dim - len(data.shape)) + data.shape

        data_reshape = F.reshape(data, shape=t_shape)
        return F.broadcast_to(data_reshape, shape=shape)
    else:
        return F.Custom(data, op_type="broadcast_to_w_samples",
                        isSamples=isSamples, shape=shape)


class ScoreFuncGradOp(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(ScoreFuncGradOp, self).__init__(**kwargs)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])
        self.assign(in_grad[1], req[1], out_grad[0]*in_data[0])


@mx.operator.register("score_func_grad")
class ScoreFuncGradOpProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ScoreFuncGradOpProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['p', 'q']

    def list_outputs(self):
        return ['l']

    def infer_shape(self, in_shape):
        return in_shape, (in_shape[0],), ()

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return ScoreFuncGradOp(**kwargs)


def score_func_grad(p, q, name="score_func_grad"):
    return mx.nd.Custom(p, q, name=name, op_type="score_func_grad")

class LogSumExpOp(mx.operator.CustomOp):
    """Implementation of log sum exp for numerical stability
    """
    def __init__(self, axis):
        self.axis = axis

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        max_x = mx.nd.max_axis(x, axis=self.axis, keepdims=True)
        sum_x = mx.nd.sum(mx.nd.exp(x - max_x), axis=self.axis, keepdims=True)
        y = mx.nd.log(sum_x) + max_x
        y = y.reshape(out_data[0].shape)
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = out_grad[0]
        x = in_data[0]
        max_x = mx.nd.max_axis(x, axis=self.axis, keepdims=True)
        y = y.reshape(max_x.shape)
        x = mx.nd.exp(x - max_x)
        prob = x / mx.nd.sum(x, axis=self.axis, keepdims=True)
        self.assign(in_grad[0], req[0], prob * y)


@mx.operator.register("logsumexp")
class LogSumExpProp(mx.operator.CustomOpProp):
    def __init__(self, axis, keepdims=False):
        super(LogSumExpProp, self).__init__(need_top_grad=True)
        self.axis = int(axis)
        self.keepdims = keepdims in ('True',)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        oshape = []
        for i, x in enumerate(data_shape):
            if i == self.axis:
                if self.keepdims:
                    oshape.append(1)
            else:
                oshape.append(x)
        return [data_shape], [tuple(oshape)], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogSumExpOp(self.axis)


def logsumexp(in_sym, axis, keepdims=False, name="logsumexp"):
    return mx.nd.Custom(in_sym, name=name,
                            op_type="logsumexp",
                            axis=axis, keepdims=keepdims)
