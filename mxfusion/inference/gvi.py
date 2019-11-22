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
from ..common.exceptions import InferenceError

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
    def __init__(self, num_samples, model, posterior, observed, prior_vars, likelihood_vars,
                 loss_function, divergence, data_size):
        super(GeneralizedVariationalInference, self).__init__(
            num_samples=num_samples, model=model,
            posterior=posterior, observed=observed)
        self.data_size = data_size if isinstance(data_size, mx.ndarray.ndarray.NDArray) else mx.nd.array([data_size])
        self.likelihood_vars = likelihood_vars # TODO parse these correctly from the model using the prior_vars
        self.prior_vars = prior_vars # TODO take these as input from the user
        self.divergence = divergence
        self.loss_function = loss_function

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

        total_likelihood, total_prior_loss, total_posterior_loss = self.compute_total_losses(F, variables)

        log_n = mx.nd.log(self.data_size) # minibatch size
        log_S = mx.nd.log(mx.nd.array([self.num_samples])) # posterior samples

        print("total_likelihood: {}".format(total_likelihood))
        print("total_prior_loss: {}".format(total_prior_loss))
        print("total_posterior_loss: {}".format(total_posterior_loss))

        logL_bar = self.loss_function.compute_loss(total_likelihood, log_n, log_S)

        # Step 2: Divergence. Do inside self.divergence.
        D_bar = self.divergence.compute_divergence(total_posterior_loss, total_prior_loss, log_S)

        # L = mx.nd.exp(logL_bar) + D_bar
        logL = logL_bar + D_bar
        print("logL: {}".format(logL))
        return logL, logL

    def sum_losses(self, loss_dictionary, target_shape, axis_length=2):
        """
        Sums the computed log_pdfs found in the loss_dictionary down to the target shape.
        """
        total_loss = None
        for uuid, loss in loss_dictionary.items():
            if loss.shape != target_shape:
                # Sum over all final dimensions.
                # we take an assumption that the first two dimensions are (S, N, ...) for the likelihood
                # and the rest can be summed over, because they are "independent".
                # This only applies to multi-variate likelihoods.
                target_axes = list(range(len(loss.shape)))[axis_length:]
                ll = mx.nd.sum(loss, axis=target_axes)
            elif loss.shape == target_shape:
                ll = loss
            if total_loss is None:
                total_loss = ll
            else:
                total_loss = total_loss + ll
        return total_loss

    def compute_total_losses(self, F, variables):
        """
        Computes the likelihood, prior, and posterior losses in the correct forms for GVI to proceed.
        These need to be returned separately unlike for the typical KL form (as in StochasticVariationalInference)
         where the prior and likelihood can be computed jointly and returned as a scalar since in GVI other losses and
         divergences don't allow the same trick of summing the two together in log space.
        """

        likelihood_loss = self.model.log_pdf_matrix(F=F, variables=variables, targets=self.likelihood_vars)
        prior_loss = self.model.log_pdf_matrix(F=F, variables=variables, targets=self.prior_vars)
        posterior_loss = self.posterior.log_pdf_matrix(F=F, variables=variables)

        # Find the target shape for likelihoods
        target_shape = None
        for leaf in self.model.leaves:
            if leaf in self.likelihood_vars:
                if target_shape is None:
                    target_shape = likelihood_loss[leaf.factor.uuid].shape[:1]
                    target_variable = likelihood_loss[leaf.factor.uuid]
                    target_varible_uuid  = leaf.factor.uuid
                # elif not (target_shape == likelihood_loss[leaf.uuid].shape): # leaf shapes disagree
                #         throw InferenceError("Final likelihood shapes are different. Please disambiguate. {} {} {}".format(likelihood_vars, target_shape, likelihood_loss[leaf.uuid].shape))

        # log_p_i_s
        total_likelihood = self.sum_losses(likelihood_loss, target_shape, axis_length=2)
        # log_pi_theta_s
        total_prior_loss = self.sum_losses(prior_loss, target_shape, axis_length=1)
        # log_q_theta_s
        total_posterior_loss = self.sum_losses(posterior_loss, target_shape, axis_length=1)

        return total_likelihood, total_prior_loss, total_posterior_loss


class LossFunction():
    def __init__(self):
        pass


class Divergence():
    def __init__(self):
        pass


class RenyiAlpha(Divergence):
    def __init__(self, alpha):
        self.alpha = alpha if isinstance(alpha, mx.ndarray.ndarray.NDArray) else mx.nd.array([alpha])

    def compute_divergence(self, total_posterior_loss, total_prior_loss, log_S):
        log_r_s = (total_posterior_loss - total_prior_loss) * (self.alpha - 1.)
        print("log_r_s: {}".format(log_r_s))

        D_tilde = logsumexp(log_r_s, axis=0, keepdims=True) - log_S # axis=samples
        print("D_tilde: {}".format(D_tilde))
        D_bar = (1 / (self.alpha - 1)) * D_tilde
        print("D_bar: {}".format(D_bar))
        return D_bar

class KullbackLeibler(Divergence):
    def __init__(self):
        pass
    
    def compute_divergence(self, total_posterior_loss, total_prior_loss, log_S):
        return mx.nd.sum(total_posterior_loss - total_prior_loss)/mx.nd.exp(log_S)


class GammaLoss(LossFunction):
    def __init__(self, gamma):
        self.gamma = gamma if isinstance(gamma, mx.ndarray.ndarray.NDArray) else mx.nd.array([gamma])

    def compute_loss(self, total_likelihood, log_n, log_S):
        # compute p_bar_s
        # want to logsumexp over all non-0 axes here, since we want to preserve the sample dimension only.
        log_p_bar = (self.gamma - 1.) / (self.gamma)  * (logsumexp(self.gamma * total_likelihood, axis=1, keepdims=True) - log_n)
        print("log_p_bar: {}".format(log_p_bar))

        # compute l_i_s
        logL_l_i_s = (self.gamma - 1)  * total_likelihood + (self.gamma - 1.) / (self.gamma) + log_p_bar
        print("logL_l_i_s: {}".format(logL_l_i_s))

        # compute logL_bar
        logL_bar = - (logsumexp(logsumexp(logL_l_i_s, axis=1, keepdims=True), axis=0) - log_S)
        print("logL_bar: {}".format(logL_bar))
        return logL_bar
    
    
class StandardLoss(LossFunction):
    def __init__(self):
        pass

    def compute_loss(self, total_likelihood, log_n, log_S):
        return -mx.nd.sum(total_likelihood)/mx.nd.exp(log_S)
        
