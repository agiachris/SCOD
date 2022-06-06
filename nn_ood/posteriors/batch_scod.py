from enum import auto
import torch

from autograd_hacks import autograd_hacks
from ..distributions import GaussianFixedDiagVar
from .scod import SCOD


class BatchSCOD(SCOD):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    Accelerated with batched dataset processing and forward pass functionality.
    """

    def __init__(self, model, dist_fam, **kwargs):
        """
        model: base DNN to equip with an uncertainty metric
        dist_fam: distributions.DistFam object representing how to interpret output of model
        args: configuration variables - defaults are in base_config
        """
        super().__init__(model, dist_fam, args=kwargs)
        proj_types = ["batch_posterior_pred"]
        sketch_types = ["batch_random", "batch_srft"]
        assert self.config["proj_type"] in proj_types, "Only parallelized for batch_posterior_pred projection"
        assert self.config["sketch_type"] in sketch_types, "Only parallelized for random and srft sketching"
        assert isinstance(self.dist_fam, GaussianFixedDiagVar), "Only parallelized for GaussianFixedDiagVar"

    def process_dataset(self, dataset, input_keys=None, batch_size=64):
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization

        dataset - torch dataset of (input, target) pairs
        input_keys - list(str) of input keys if the dataset returns batched dictionaries
        batch_size - PyTorch DataLoader batch size for accelerating dataset Fisher computation
        """
        # loop through data in batches
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        sketch = self.sketch_class(N=self.n_params,
                                   M=len(dataset),
                                   r=self.num_eigs,
                                   T=self.num_samples,
                                   gpu=self.gpu)

        autograd_hacks.disable_computation_graph()
        for i, sample in enumerate(dataloader):

            if isinstance(sample, dict):
                assert input_keys is not None, "Require keys to extract inputs"
                assert not self.weighted, "Dataset does not provide labels"
                inputs = [self._prep_vec(sample[k]) for k in input_keys]
                labels = None
            elif isinstance(sample, tuple):
                assert input_keys is None, "Keys cannot be used to extract inputs from a tuple"
                inputs = self._prep_vec(sample[0])
                labels = self._prep_vec(sample[1])

            # get params of output dist
            if isinstance(inputs, list): thetas = self.model(*inputs)
            else: thetas = self.model(inputs)
            thetas = self._format_output(thetas, batch_size=batch_size)
            
            # weight by negative log-likelihood
            weight = 1.
            if self.weighted:
                nll = self.dist_fam.loss(thetas, labels)
                weight = torch.exp(-nll)

            # pre-multipy by sqrt fisher, compute L^(i)_w, add to sketch
            thetas = self.dist_fam.apply_sqrt_F(thetas)  
            L = self._get_weight_jacobian(thetas.mean(0), thetas.size(0))
            sketch.low_rank_update(i, L.transpose(2, 1), weight)
        del L

        eigs, basis = sketch.get_range_basis()
        del sketch

        self.projector.process_basis(eigs, basis)
        self.configured.data = torch.ones(1, dtype=torch.bool)
        self.clear_gradients()

    def forward(self, inputs, 
                      input_keys=None, 
                      n_eigs=None, 
                      Meps=5000, 
                      compute_unc=True,
                      compute_var=False,
                      detach=True):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input

        returns 
            mu = model(inputs) -- shape (N, 1)
            unc = hessian based uncertainty estimates shape (N)
            var = posterior predictive distribution variance
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError

        if n_eigs is None: 
            n_eigs = self.num_eigs

        # Make forward and backward pass differentable processes
        if detach: autograd_hacks.disable_computation_graph()
        else: autograd_hacks.enable_computation_graph()

        if isinstance(inputs, dict):
            assert input_keys is not None, "Require keys to extract inputs"
            inputs = [inputs[k] for k in input_keys]
            mu = self.model(*inputs)
        elif isinstance(inputs, tuple): mu = self.model(*inputs)
        else: mu = self.model(inputs)
        mu = self._format_output(mu)
        
        unc = None
        if compute_unc:
            theta = self.dist_fam.apply_sqrt_F(mu, exact=True)
            L = self._get_weight_jacobian(theta.mean(0), mu.size(0), detach=detach)
            unc = self.projector.compute_distance(L.transpose(2, 1),
                                                  self.proj_type,
                                                  n_eigs=n_eigs,
                                                  Meps=Meps)

        var = None
        if compute_var:
            J = self._get_weight_jacobian(mu.mean(0), mu.size(0), detach=detach)
            var = self.projector.compute_distance(J.transpose(2, 1),
                                                  "batch_posterior_pred_var",
                                                  n_eigs=n_eigs,
                                                  Meps=Meps)

        return dict(output=self.dist_fam.output(mu), unc=unc, var=var)

    def clear_gradients(self):
        self.dist_fam.zero_grad()
        autograd_hacks.clear_model_gradients(self.model)

    def _get_weight_jacobian(self, vec, batch_size, detach=True):
        """
        returns b x d x nparam matrix, with each row of each d x nparam matrix being d(vec[i])/d(weights)
        """
        assert vec.dim() == 1
        grad_vecs = []
        autograd_hacks.clear_model_gradients(self.model)
        for j in range(vec.size(0)):
            vec[j].backward(retain_graph=True, create_graph=not detach)
            autograd_hacks.compute_grad1(self.model)
            g = self._get_grad_vec(batch_size)
            if detach: g = g.detach()
            grad_vecs.append(g)
            autograd_hacks.clear_model_gradients(self.model)

        return torch.stack(grad_vecs).transpose(1, 0)

    def _get_grad_vec(self, batch_size):
        """
        returns gradient of NN parameters flattened into a vector
        assumes backward() has been called so each parameters grad attribute
        has been updated
        """
        grads = [p.grad1.contiguous().view(batch_size, -1) for p in self.trainable_params]
        return torch.cat(grads, dim=1)

    def _prep_vec(self, x):
        if self.gpu: x.cuda()
        return x

    @staticmethod
    def _format_output(x, batch_size=None):
        """
        format model prediction to shape b x d
        """
        if x.dim() == 2 and x.size(0) == 1:
            if batch_size is not None: 
                assert batch_size != 1
            x = x.squeeze(0)
        if x.dim() == 1: 
            x = x.unsqueeze(-1)
        return x
