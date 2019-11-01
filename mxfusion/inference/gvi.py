# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


from .variational import StochasticVariationalInference
import mxnet as mx
from ..util.customop import logsumexp

class GeneralizedVariationalInference(StochasticVariationalInference):
    """
    The class for the Generalized Variational Inference (GVI) algorithm.

    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param model: the definition of the probabilistic model
    :type model: Model
    :param posterior: the definition of the variational posterior of the probabilistic model
    :param posterior: Posterior
    :param observed: A list of observed variables
    :type observed: [Variable]
    """
    def __init__(self, num_samples, model, posterior, observed, prior_vars, likeihood_vars, alpha, gamma, data_size):
        super(GeneralizedVariationalInference, self).__init__(
            model=model, posterior=posterior, observed=observed)
        self.observed = observed
        self.num_samples = num_samples
        self.data_size = data_size
        self.likelihood_vars = likelihood_vars # TODO parse these correctly from the model using the prior_vars
        self.prior_vars = prior_vars # TODO take these as input from the user
        # self.divergence = divergence
        # self.loss_function = loss_function
        self.alpha = alpha
        self.gamma = gamma

    def compute(self, F, variables):
        """
        Compute the inference algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol
        """

        # Input to algorithm -> p(x_i | theta)
        samples = self.posterior.draw_samples(
            F=F, variables=variables, num_samples=self.num_samples)
        variables.update(samples)

        # import pdb; pdb.set_trace()

        # Step 1: Loss Function. Do inside self.loss_function really, probably make interface take the model, variables, etc.

        # log_p_i_s
        likelihood_loss = self.model.log_pdf_matrix(F=F, variables=variables, targets=self.likeihood_vars)

        # get final likelihood -> last factor in the graph. It's shape is the target shape ~(S, N). This is the shape we use for all other calculations.
        # sum all other likelihoods down into target_shape.
        # sum together all likelihood terms, and use that total term for all calculations.

        x_loss = likelihood_loss[model.leaves()]
        z_loss = z_loss.reshape(x_loss)

        total_loss = x_loss + z_loss # shape = (S, N)

        # TODO construct the right matrix out of these variables?. Sum over all the last dimensions, non-sample ones.

        # log_pi_theta_s
        prior_loss = self.model.log_pdf_matrix(F=F, variables=variables, targets=self.prior_vars)
        # log_q_theta_s
        posterior_loss = self.posterior.log_pdf_matrix(F=F, variables=variables)
        log_n = mx.nd.log(self.data_size) # minibatch size
        log_S = mx.nd.log(self.num_samples) # posterior samples

        # compute p_bar_s # TODO not sure on the "index i" part here in the pseudocode, nor on the axis of logsumexp. Same ? probably.
        # want to logsumexp over all non-0 axes here, since we want to preserve the sample dimension only.
        log_p_bar = (gamma - 1.) / (gamma)  * (logsumexp(gamma * likelihood_loss, axis=1) - log_n)

        # compute l_i_s
        # Adding log_p_bar into the sample axis.
        logL_l_i_s = (gamma - 1)  * likelihood_loss + (gamma - 1.) / (gamma) + log_p_bar

        # compute logL_bar
        # logL_bar should be a scalar now. Make sure logsumexp goes over all axes non-0.
        logL_bar = - (logsumexp(logsumexp(logL_l_i_s, axis=1), axis=0) - log_n - log_S)

        # Step 2: Divergence. Do inside self.divergence.
        log_r_s = (posterior_loss - prior_loss) * (self.alpha - 1.)
        D_tilde = logsumexp(log_r_s, axis=0) - log_S # axis=samples. TODO the losses we get out of .log_pdf are scalar
        D_bar = (1 / (alpha - 1)) / mx.nd.log(D_tilde)  # TODO do we mean to re-log D_tilde here?

        return N * mx.nd.exp(logL_bar) + D_bar
