"""
two cases: same or different distributions. If same, you just change the parameters and then sample. Otherwise we pick K different distibutions, then has to pick the right distribution and behave accordingly.
"""

import numpy as np
import mxnet as mx
from ...common.config import get_default_MXNet_mode
from ..variables import Variable
from .univariate import UnivariateDistribution
from .distribution import Distribution
from ..variables import get_num_samples
from ...util.customop import broadcast_to_w_samples


class Mixture(UnivariateDistribution):
    """
    Mixture distribution. All distributions it mixes must be of the same class.
    TODO extend to arbitrary mixing.

    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, mixing_prob, components, normalization=True, rand_gen=None, dtype=None, ctx=None, axis=-1):

        self.num_components = len(components)
        self.normalization = normalization
        self.axis = axis

        inputs = [('component_'+str(i)+'_'+name,input) for i, c in enumerate(components) for (name, input) in c.inputs  ] # double enumerate over components and the inputs of the components.
        self.components = components
        if not isinstance(mixing_prob, Variable):
           mixing_prob = Variable(value=mixing_prob)
        inputs.append(('mixing_prob', mixing_prob))
        input_names = [n for (n,_) in inputs]
        output_names = ['random_variable']
        super(Mixture, self).__init__(inputs=inputs, outputs=None,
                                     input_names=input_names,
                                     output_names=output_names,
                                     rand_gen=rand_gen, dtype=dtype, ctx=ctx)


    def log_pdf(self, mixing_prob, random_variable, F=None, **kwargs):
        """
        Computes the logarithm of the probability density function (PDF) of the Gamma distribution.

        :param mixing_prob: the logarithm of the probability being in each of the distributions.
        :type mixing_prob: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the Gamma distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        component_log_probs = []
        for i, component in enumerate(self.components):
            # import pdb; pdb.set_trace()

            variables = {v.uuid: kwargs[name] for name, v in self.inputs if name.startswith('component_'+str(i)+'_')}
            variables.update({component.outputs[0][1].uuid: random_variable})
            component_log_probs.append(component.log_pdf(F, variables))

        if self.normalization:
            mixing_prob = F.log_softmax(mixing_prob, axis=self.axis)


        cat_mixing_probs = F.split(data=mixing_prob, num_outputs=self.num_components, axis=-1)
        paired = [component_prob + cat_prob for (component_prob, cat_prob) in zip(component_log_probs, cat_mixing_probs)]
        paired_stacked = F.stack(paired, axis=0)
        a = F.max_axis(paired_stacked)
        logL = F.log(F.sum(F.exp(paired_stacked - a), axis=0)) + a
        logL = logL * self.log_pdf_scaling

        return logL



    def draw_samples(self, mixing_prob, rv_shape, num_samples=1, F=None, **kwargs):
        """
        Draw samples from the Mixture distribution.
        :param mixing_prob: the logarithm of the probability being in each of the distributions.
        :type mixing_prob: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param nSamples: the number of drawn samples (default: one).
        :int nSamples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Gamma distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        if self.normalization:
            mixing_prob = F.log_softmax(mixing_prob, axis=self.axis)

        mixing_prob = F.reshape(mixing_prob, shape=(1, -1, self.num_components))
        samples = self._rand_gen.sample_multinomial(mixing_prob, get_prob=False)
        samples = F.one_hot(samples, depth=self.num_components)
        samples = F.reshape(samples, shape=(num_samples,) + rv_shape)
        return samples

    @staticmethod
    def define_variable(mixing_prob, components, normalization=True, shape=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Gamma distribution parameterized with a and b parameters.

        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Gamma distribution.
        :rtypes: Variable
        """
        dist = Mixture(mixing_prob=mixing_prob, components=components, normalization=normalization, rand_gen=rand_gen,
                        dtype=dtype, ctx=ctx)
        dist._generate_outputs(shape=shape)
        return dist.random_variable
