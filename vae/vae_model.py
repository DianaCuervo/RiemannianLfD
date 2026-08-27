import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
import pickle
from stochman import nnj
import hyperspherical_vae.distributions.von_mises_fisher as vmf
from sklearn.cluster import KMeans

###Create VAE Class --> To reconsider as it migth not be needed
class VAE(nn.Module):
    # (Note: I removed EmbeddedManifold inheritance here assuming you just need standard nn.Module,
    # but keep it if StochMan strictly requires it)

    def __init__(self, layers, batch_size, pos_dof, qua_dof, sigma=1e-6, sigma_z=0.1):
        """
        Create a Variational Auto-Encoder (VAE) neural network integrated with an embedded Riemannian manifold.
            Input:
                layers:     number of neurons in each layer of the VAE's encoder [D, h_1, ..., h_n, d]
                            (D: Ambient space dimension, h_n: nth hidden layer size,
                            d: latent space dimension, tested with 2 dimensions)
                            the reverse is used for the decoder [d, h_n, ..., h_1, D].
                batch_size: training batch size
                sigma:      the scale parameter of the distribution given as the output of the VAE's decoder
                            for position data
                sigma_z:    the scale parameter of the distribution given as the output of the VAE's encoder
        """
        super(VAE, self).__init__()

        #architecture
        self.p = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers[1:-1]  # Dimension of hidden layers
        self.pos_dof = pos_dof
        self.qua_dof = qua_dof

        # Hyper-parameters
        self.device = 'cpu'
        self.batch_size = batch_size
        self.num_clusters = 500  # Number of clusters in the RBF k_mean
        self.vmf_concentration_scale = 1e2  # the scale of vmf distribution concentration

        #  Initialize VAE
        enc = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc.append(nnj.BatchNorm1d(in_features))
            enc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        enc.append(nnj.Linear(out_features, self.d))

        enc_scale = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc_scale.append(nnj.BatchNorm1d(in_features))
            enc_scale.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features), nnj.Softplus()))

        dec = []
        for k in reversed(range(len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            dec.append(nnj.BatchNorm1d(in_features))
            dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        dec.pop(0)  # remove initial batch-norm as it serves no purpose
        dec.append(nnj.Linear(out_features, self.p))

        self.encoder_loc = nnj.Sequential(*enc)
        self.decoder_loc = nnj.Sequential(*dec)
        self.encoder_scale = nnj.Sequential(*enc_scale)

        self.encoder_scale_fixed = nn.Parameter(torch.tensor([sigma_z]), requires_grad=False)
        self.decoder_scale_pos = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.decoder_scale_qua = nn.Parameter(torch.tensor(np.ones((self.batch_size, 3)) *
                                                           self.vmf_concentration_scale), requires_grad=False)
        self.dec_std_pos = lambda z: torch.ones(20, self.p, device=self.device)
        self.dec_std_qua = lambda z: torch.ones(20, self.p, device=self.device)

        self.prior_loc = nn.Parameter(torch.zeros(self.d), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.prior = td.Independent(td.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)

    def embed(self, points, jacobian=False):
        """
        Embed the manifold into (mu, std) space.

        Input:
            points:     a Nx(d) or BxNx(d) torch Tensor representing a (batch of a)
                        set of N points in latent space that will be embedded
                        in R^2D.

        Optional input:
            jacobian:   a boolean indicating if the Jacobian matrix of the function
                        should also be returned. Default is False.

        Output:
            embedded:   a Nx(2D) of BxNx(2D) torch tensor containing the N embedded points.
                        The first Nx(d) part contain the mean part of the embedding,
                        whlie the last Nx(d) part contain the standard deviation
                        embedding.

        Optional output:
            J:          If jacobian=True then a second Nx(2D)x(d) or BxNx(2D)x(d)
                        torch tensor is returned that contain the Jacobian matrix
                        of the embedding function.
        """
        std_scale = 1.0
        metric = None
        j = None
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # BxNxD
        if jacobian:
            mu_pos, mu_qua, j_mu = self.decode(points, train_rbf=True, jacobian=True)  # BxNxD, BxNxDx(d)
            std, j_std = self.dec_std_pos(points, jacobian=True)  # BxNxD, BxNxDx(d)
            std_qua, j_std_qua = self.dec_std_qua(points, jacobian=True)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu_pos.mean, mu_qua.loc.unsqueeze(0), std_scale * std, std_scale * (1 / std_qua)),
                                 dim=2)  # BxNx(2D)
            j = torch.cat((j_mu, torch.cat((std_scale * j_std.squeeze(0), std_scale * j_std_qua.squeeze(0)), dim=1)),
                          dim=2)  # BxNx(2D)x(d)
            m = torch.einsum("bji,bjk->bik", j_mu, j_mu)
            m2 = torch.einsum("bji,bjk->bik", j_std.squeeze(0), j_std.squeeze(0))
            m3 = torch.einsum("bji,bjk->bik", j_std_qua.squeeze(0), j_std_qua.squeeze(0))
            metric = (m3 + m2 + m).detach().numpy()
        else:
            mu_pos, mu_qua = self.decode(points, train_rbf=True, jacobian=False)  # BxNxD, BxNxDx(d)
            std = self.dec_std_pos(points, jacobian=False)  # BxNxD, BxNxDx(d)
            std_qua = self.dec_std_qua(points, jacobian=False)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu_pos.mean, mu_qua.loc.unsqueeze(0), std_scale * std, std_scale * (1 / std_qua)),
                                 dim=2)  # BxNx(2D)
        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                j = j.squeeze(0)
        if jacobian:
            return embedded, j, metric
        else:
            return embedded

    def encode(self, x, train_rbf=False):
        """ Encode the input space sample
        Inputs:
            x: a torch Tensor corresponding to one input space point.
            train_rbf: True when training the RBFs
        Outputs:
            z_distribution: latent space encoded distribution given x
        """
        z_loc = self.encoder_loc(x)
        if train_rbf:
            z_scale = self.encoder_scale(x)
        else:
            z_scale = self.encoder_scale_fixed
        z_distribution = td.Independent(td.Normal(loc=z_loc, scale=z_scale, validate_args=False), 1), z_loc
        return z_distribution

    def decode(self, z, train_rbf=False, jacobian=False, negative=False):
        """ compute the input space estimation given the latent variable
        Inputs:
            z: sample from latent space
            train_rbf: True when training the RBFs
            jacobian: True to calculate Jacobian
            negative: True to flip the quaternion
        Outputs:
            position_distribution: gaussian distribution given z
            quaternion_distribution: vMF distribution given z
        """
        # Since batch normalization is a bit of a mess we have to apply
        # a series of reshape's to get the correct behavior
        quaternion_distribution_negative = None  # used when p(x|z) = ½vMF(x|mu(z), k(z)) + ½vMF(x|-mu(z), k(z))
        ja = None
        if jacobian:
            x_loc, ja = self.decoder_loc(z.view(-1, self.d), jacobian=jacobian)
        else:
            x_loc = self.decoder_loc(z.view(-1, self.d))
        position_scale = self.decoder_scale_pos + 1e-10
        quaternion_scale = self.decoder_scale_qua + 1e-10

        x_var_pos = self.dec_std_pos(z.view(-1, self.d))
        x_var_qua = self.dec_std_qua(z.view(-1, self.d))

        position_loc = x_loc[:, :self.pos_dof]
        quaternion_loc = x_loc[:,self.pos_dof:]
        qua_mean = quaternion_loc / quaternion_loc.norm(dim=-1, keepdim=True)

        x_shape = list(z.shape)
        x_shape[-1] = position_loc.shape[-1]

        quaternion_distribution = vmf.VonMisesFisher(qua_mean, quaternion_scale)
        quaternion_distribution_negative = vmf.VonMisesFisher(-qua_mean, quaternion_scale)
        position_distribution = td.Independent(
            td.Normal(loc=position_loc.view(torch.Size(x_shape)), scale=position_scale), 1)

        if train_rbf:
            if negative:
                quaternion_distribution_negative = vmf.VonMisesFisher(-qua_mean, x_var_qua)
            quaternion_distribution = vmf.VonMisesFisher(qua_mean, x_var_qua)
            position_distribution = td.Independent(
                td.Normal(loc=position_loc.view(torch.Size(x_shape)), scale=x_var_pos), 1)

        if jacobian:
            return position_distribution, quaternion_distribution, ja
        if negative:
            return position_distribution, quaternion_distribution, quaternion_distribution_negative
        return position_distribution, quaternion_distribution

    def disable_training(self):
        """ Disabling the training for all the networks
        Inputs:

        Outputs:

        """
        for module in self.encoder_loc._modules.values():
            module.training = False
        for module in self.decoder_loc._modules.values():
            module.training = False

    def init_std(self, x, load_clusters=False, cluster_path="./data/clusters.p", beta_scale=1.0):
        """ initializing the RBF networks
        Inputs:
            x: a torch Tensor corresponding to one input space point.
            load_clusters: loading the clusters from a file
        Outputs:
            cluster_centers: center of clusters computed by kmeans
        """
        self.train_var = True
        with torch.no_grad():
            _, z = self.encode(x, train_rbf=True)
        d = z.shape[1]
        inv_max_std = np.sqrt(1e-12)  # 1.0 / x.std()
        beta = beta_scale / z.std(dim=0).mean()
        rbf_beta = beta * torch.ones(1, self.num_clusters)

        if load_clusters:
            k_means = pickle.load(open(cluster_path, "rb"))
        else:
            k_means = KMeans(n_clusters=self.num_clusters).fit(z.numpy())
            pickle.dump(k_means, open(cluster_path, "wb"))

        centers = torch.tensor(k_means.cluster_centers_)

        self.dec_std_pos = nnj.Sequential(nnj.RBF(d, self.num_clusters, points=centers, beta=rbf_beta),
                                          nnj.PosLinear(self.num_clusters, 1, bias=False),
                                          nnj.Reciprocal(inv_max_std),
                                          nnj.PosLinear(1, self.pos_dof))
        self.dec_std_qua = nnj.Sequential(nnj.RBF(d, self.num_clusters, points=centers, beta=rbf_beta),
                                          nnj.PosLinear(self.num_clusters, self.qua_dof))

        self.dec_std_pos.to(self.device)
        self.dec_std_qua.to(self.device)
        cluster_centers = k_means.cluster_centers_
        return cluster_centers

# Metric set of points *also paths
def get_M(model, data):
    ### Getting metric for all point in the tensor data with Nxd points
    model.eval()
    device = next(model.parameters()).device
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        e, J, M = model.embed(data.unsqueeze(0), jacobian=True)

    if not isinstance(M, torch.Tensor):
        M = torch.as_tensor(M, dtype=torch.float32, device=device)

    if M.dim() == 4 and M.shape[0] == 1:
        M = M.squeeze(0)
    return e, J, M

# Load the pretrained VAE
def load_pretrained_vae(config, dummy_tensor, device='cpu'):
    """
    Instantiates the VAE, builds empty layers using a dummy tensor,
    and loads the real pre-trained weights.
    """
    # 1. Read directly from the config dictionary
    layers = config['architecture']['layers']
    batch_size = config['training_artifacts']['batch_size']
    pos_dof = config['architecture']['pos_dof']
    qua_dof = config['architecture']['qua_dof']
    sigma_z = config['architecture']['sigma_z']
    beta_scale = config['architecture'].get('beta_scale', 1.0)
    model_path = config['training_artifacts']['model_path']
    cluster_path = config['training_artifacts']['cluster_path']

    # 2. Initialize the empty architecture
    vae = VAE(layers=layers, batch_size=batch_size, pos_dof=pos_dof, qua_dof=qua_dof, sigma_z=sigma_z).to(device)
    vae.obstacle_input_space = None

    # 3. Initialize clusters (Using random noise just to build the graph dimensions!)
    vae.init_std(dummy_tensor.to(device), load_clusters=True, cluster_path=cluster_path, beta_scale=beta_scale)

    # 4. OVERWRITE the fake initialization with your real, trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])

    # 5. Freeze for inference
    vae.disable_training()
    vae.eval()

    print(f"✅ Successfully loaded VAE from {model_path}")
    return vae