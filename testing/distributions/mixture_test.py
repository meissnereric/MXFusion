import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array, get_num_samples
from mxfusion.components.distributions import Mixture, Normal
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestMixtureDistribution(object):

    @pytest.mark.parametrize("dtype, mixing_prob, mixing_prob_isSamples, rv, rv_isSamples, num_samples, one_hot_encoding, normalization", [
        (np.float64, np.random.rand(5,3)+1e-2, True, np.random.rand(5,3)+1e-2, True, np.random.rand(5,3,3)+1e-2, True, 5, True, True),
        (np.float64, np.random.rand(3)+1e-2, False, np.random.rand(5,3)+1e-2, True, 5, True, False),
        ])
    def test_log_pdf(self, dtype, mixing_prob, mixing_prob_isSamples, rv, rv_isSamples, var, var_isSamples, num_samples, one_hot_encoding, normalization):

        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape
        n_dim = 1 + len(rv.shape) if not rv_isSamples else len(rv.shape)
        mixing_prob_np = numpy_array_reshape(mixing_prob, mixing_prob_isSamples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_isSamples, n_dim)
        rv_full_shape = (num_samples,)+rv_shape
        rv_np = np.broadcast_to(rv_np, rv_full_shape)
        mixing_prob_np = np.broadcast_to(mixing_prob_np, rv_full_shape[:-1]+(3,))

        if normalization:
            log_pdf_np = np.log(np.exp(mixing_prob_np)/np.exp(mixing_prob_np).sum(-1, keepdims=True)).reshape(-1, 3)
        else:
            log_pdf_np = mixing_prob_np.reshape(-1, 3)
        if one_hot_encoding:
            log_pdf_np = (rv_np.reshape(-1, 3)*log_pdf_np).sum(-1).reshape(rv_np.shape[:-1])
        else:
            bool_idx = np.arange(3)[None,:] == rv_np.reshape(-1,1)
            log_pdf_np = log_pdf_np[bool_idx].reshape(rv_np.shape[:-1])


        # compute the mixture of gaussians manually using scipy density functions + mixing distribution


        components = [Normal.define_variable().factor, Normal.define_variable().factor, Normal.define_variable().factor]
        cat = Mixture.define_variable(mixing_prob=mixing_prob, components=components, normalization=normalization, shape=rv_shape, dtype=dtype).factor
        mixing_prob_mx = mx.nd.array(mixing_prob, dtype=dtype)
        if not mixing_prob_isSamples:
            mixing_prob_mx = add_sample_dimension(mx.nd, mixing_prob_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_isSamples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {cat.mixing_prob.uuid: mixing_prob_mx,
                     cat.random_variable.uuid: rv_mx,
                     cat.component_0_mean.uuid: rv_mx,
                     cat.component_0_variance.uuid: mixing_prob_mx,
                     cat.component_1_mean.uuid: mixing_prob_mx,
                     cat.component_1_variance.uuid: mixing_prob_mx,
                     cat.component_2_mean.uuid: mixing_prob_mx,
                     cat.component_2_variance.uuid: mixing_prob_mx
                     }
        log_pdf_rt = cat.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    @pytest.mark.parametrize(
        "dtype, mixing_prob, mixing_prob_isSamples, rv_shape, num_samples, one_hot_encoding, normalization",[
        (np.float64, np.random.rand(5,4,3)+1e-2, True, (4,3), 5, True, True),
        (np.float64, np.random.rand(4,3)+1e-2, False, (4,3), 5, True, True),
        ])
    def test_draw_samples(self, dtype, mixing_prob, mixing_prob_isSamples, rv_shape, num_samples, one_hot_encoding, normalization):
        n_dim = 1 + len(rv_shape)
        mixing_prob_np = numpy_array_reshape(mixing_prob, mixing_prob_isSamples, n_dim)
        rv_full_shape = (num_samples,) + rv_shape
        mixing_prob_np = np.broadcast_to(mixing_prob_np, rv_full_shape[:-1] + (3,))


        rand_np = np.random.randint(0, 3, size=rv_full_shape[:-1])
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand_np.flatten(), dtype=dtype))

        if one_hot_encoding:
            rand_np = np.identity(3)[rand_np].reshape(*rv_full_shape)
        else:
            rand_np = np.expand_dims(rand_np, axis=-1)
        rv_samples_np = rand_np

        components = [Normal.define_variable().factor, Normal.define_variable().factor, Normal.define_variable().factor]
        cat = Mixture.define_variable(mixing_prob=mixing_prob, components=components, normalization=normalization, shape=rv_shape, rand_gen=rand_gen, dtype=dtype).factor
        mixing_prob_mx = mx.nd.array(mixing_prob, dtype=dtype)
        if not mixing_prob_isSamples:
            mixing_prob_mx = add_sample_dimension(mx.nd, mixing_prob_mx)

        variables = {cat.mixing_prob.uuid: mixing_prob_mx,
                 cat.random_variable.uuid: rv_mx,
                 cat.component_0_mean.uuid: mixing_prob_mx,
                 cat.component_0_variance.uuid: mixing_prob_mx,
                 cat.component_1_mean.uuid: mixing_prob_mx,
                 cat.component_1_variance.uuid: mixing_prob_mx,
                 cat.component_2_mean.uuid: mixing_prob_mx,
                 cat.component_2_variance.uuid: mixing_prob_mx
                 }
        rv_samples_rt = cat.draw_samples(
            F=mx.nd, variables=variables, num_samples=num_samples)
        # import pdb; pdb.set_trace()
        assert is_sampled_array(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples
        assert np.allclose(rv_samples_np, rv_samples_rt.asnumpy())
