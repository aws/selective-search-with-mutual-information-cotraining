import torch.nn as nn
import torch


class FeedForward(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='relu', dropout_rate=0, layer_norm=False,
                 residual_connection=False):
        """This model wraps a (residual) neural network in a easy-access way.
        """
        super().__init__()

        assert num_layers >= 0  # 0 = Linear
        if num_layers > 0:
            assert dim_hidden > 0
        if residual_connection:
            assert dim_hidden == dim_input

        self.residual_connection = residual_connection
        self.stack = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer = []

            if layer_norm:
                layer.append(nn.LayerNorm(dim_input if layer_idx == 0 else dim_hidden))

            layer.append(nn.Linear(dim_input if layer_idx == 0 else dim_hidden,
                                   dim_hidden))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])

            if dropout_rate > 0:
                layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_input if num_layers < 1 else dim_hidden,
                             dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = x + layer(x) if self.residual_connection else layer(x)
        return self.out(x)


class ConditionalDistributionZ(nn.Module):

    def __init__(self, number_clusters, dim_input, num_layers, dim_hidden):
        """This model maps the input (BERT representation of a sentence) to the probability vector for
        query routing or document assignment to the clusters.

        Parameters
        ----------
        number_clusters : int
            The number of clusters to which we are going to assign the documents or route the queries.
        dim_input : int
            The dimension of the input (the representation from BERT base model is 768.)
        num_layers : int
            How many layers are used in this model.
        dim_hidden : int
            How many neurons are in the hidden layer of this model.
            
        """
        super(ConditionalDistributionZ, self).__init__()
        self.number_clusters = number_clusters
        self.softmax = nn.Softmax(dim=-1)
        number_clusters = number_clusters
        self.ff = FeedForward(dim_input, dim_hidden, number_clusters, num_layers)

    def forward(self, inputs):
        logits = self.ff(inputs).view(inputs.size(0), 1, self.number_clusters)
        probability_vector = self.softmax(logits)
        return probability_vector


class MarginalDistributionZ(nn.Module):

    def __init__(self, number_clusters):
        """This is the function approximating the distribution of cluster sizes for document assignment.
        We use `softmax` to make sure the output is a distribution.

        Parameters
        ----------
        number_clusters : int
            The number of clusters to which we are going to assign the documents.

        """
        super(MarginalDistributionZ, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.theta = nn.Embedding(1, number_clusters)

    def forward(self):
        """Return the approximated distribution.
        """
        logits = self.theta.weight
        probability_vector = self.softmax(logits)
        return probability_vector


def get_init_function(init_value):
    """This is for the initialization of the Prior and Posterior model.
    We can change the scale of the initialization by the argument `init_value`.
    """
    def init_function(m):
        if init_value > 0.:
            if hasattr(m, 'weight'):
                m.weight.data.uniform_(-init_value, init_value)
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0.)

    return init_function


def cross_entropy_p_q(p, q):
    """This is the function calculating the cross entropy between two distributions.
    If there are multiple distributions in `p` or in `q`, we calculate the cross entropy correspondingly and take the average.
    """
    if len(q.size()) == 2:
        q = q.repeat(p.size(0), 1, 1)
    return (- (p * torch.log(q)).sum(dim=(1, 2))).mean()
