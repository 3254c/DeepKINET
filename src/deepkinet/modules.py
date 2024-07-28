import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import functorch
from einops import rearrange, reduce, repeat

class SeqNN_LinearGELU(nn.Module):
    def __init__(self, num_steps, dim):
        super(SeqNN_LinearGELU, self).__init__()
        modules = [
            LinearGELU(dim, dim)
            for _ in range(num_steps)
        ]
        self.f = nn.Sequential(*modules)
    def forward(self, pre_h):
        post_h = self.f(pre_h)
        return(post_h)

class LinearGELU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearGELU, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.GELU())
    def forward(self, x):
        h = self.f(x)
        return(h)

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()
        self.x2h = LinearGELU(x_dim, h_dim)
        self.h2mu = nn.Linear(h_dim, z_dim)
        self.h2logvar = nn.Linear(h_dim, z_dim)
        self.softplus = nn.Softplus()
        self.dist = dist.Normal

    def forward(self, x):
        h = self.x2h(x)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        qz = self.dist(mu, self.softplus(logvar))
        z = qz.rsample()
        return(z, qz)


class Encoder_of_s_u(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim):
        super(Encoder_of_s_u, self).__init__()
        self.x2h = LinearGELU(x_dim*2, h_dim)
        self.seq_nn = SeqNN_LinearGELU(num_h_layers - 1, h_dim)
        self.h2mu = nn.Linear(h_dim, z_dim)
        self.h2logvar = nn.Linear(h_dim, z_dim)
        self.softplus = nn.Softplus()
        self.dist = dist.Normal

    def forward(self, s, u):
        s_u = torch.cat([s, u], dim=1)
        pre_h = self.x2h(s_u)
        post_h = self.seq_nn(pre_h)
        mu = self.h2mu(post_h)
        logvar = self.h2logvar(post_h)
        qz = self.dist(mu, self.softplus(logvar))
        z = qz.rsample()
        return(z, qz)

class Decoder(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super(Decoder, self).__init__()
        self.z2h = LinearGELU(z_dim, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        h = self.z2h(z)
        ld = self.h2ld(h)
        correct_ld = self.softplus(ld)
        normalize_ld = correct_ld  / correct_ld.mean(dim=-1, keepdim=True)
        return(normalize_ld)

class Decoder_onehot(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super(Decoder_onehot, self).__init__()
        self.z2h = LinearGELU(z_dim, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z, onehot):
        zb = torch.cat([z, onehot], dim=-1)
        h = self.z2h(zb)
        ld = self.h2ld(h)
        correct_ld = self.softplus(ld)
        normalize_ld = correct_ld  / correct_ld.mean(dim=-1, keepdim=True)
        return(normalize_ld)

class DeepKINET(nn.Module):
    def __init__(self, x_dim, loss_mode, z_dim = 20, h_dim = 100, enc_z_layers = 2, batch_key=None, batch_onehot=None):
        super(DeepKINET, self).__init__()
        self.batch_key = batch_key
        self.batch_onehot = batch_onehot
        if self.batch_key is None:
            print('batch_key is None')
            self.enc_z = Encoder_of_s_u(enc_z_layers, x_dim, h_dim, z_dim)
            self.enc_d = Encoder(z_dim, h_dim, z_dim)
            self.dec_z = Decoder(z_dim, h_dim, x_dim)
            self.dec_b = Decoder(z_dim, h_dim, x_dim)
            self.dec_g = Decoder(z_dim, h_dim, x_dim)
        else:
            print('batch_key is not None')
            self.enc_z = Encoder_of_s_u(enc_z_layers, x_dim * 2 + batch_onehot.shape[1], h_dim, z_dim)
            self.enc_d = Encoder(z_dim, h_dim, z_dim)
            self.dec_z = Decoder_onehot(z_dim + batch_onehot.shape[1], h_dim, x_dim)
            self.dec_b = Decoder_onehot(z_dim + batch_onehot.shape[1], h_dim, x_dim)
            self.dec_g = Decoder_onehot(z_dim + batch_onehot.shape[1], h_dim, x_dim)
        self.dt = 1
        self.d_coeff = 0.01
        self.loggamma = Parameter(torch.Tensor(x_dim))
        self.logbeta = Parameter(torch.Tensor(x_dim))
        self.logtheta = Parameter(torch.Tensor(x_dim))
        self.softplus = nn.Softplus()
        self.dynamics = False
        self.kinetics = False
        self.relu = nn.ReLU()
        self.loss_mode = loss_mode
        print('loss_mode',self.loss_mode)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.loggamma)
        init.normal_(self.logbeta)
        init.normal_(self.logtheta)

    def calc_z_d_kld(self, qz):
        kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
        return(kld.sum(dim=-1))

    def calculate_diff_x_grad(self, z, d):
        dec_f = lambda vz: self.dec_z(vz)
        dec_jvp = lambda vz, vd: functorch.jvp(dec_f, (vz, ), (vd, ))[1]
        diff_px_zd_ld = self.d_coeff * functorch.vmap(dec_jvp, in_dims=(0, 0))(z, d)
        return diff_px_zd_ld

    def calculate_diff_x_grad_onehot(self, z, onehot, d):
        dec_f = lambda vz, vonehot: self.dec_z(vz, vonehot)
        dec_jvp = lambda vz, vonehot, vd: functorch.jvp(lambda z: dec_f(z, vonehot), (vz,), (vd,))[1]
        diff_px_zd_ld = self.d_coeff * functorch.vmap(dec_jvp, in_dims=(0, 0, 0))(z, onehot, d)
        return diff_px_zd_ld

    def calc_poisson_loss(self, ld, norm_mat, obs):
        p_z = dist.Poisson(ld * norm_mat+ 1.0e-16)
        l = - p_z.log_prob(obs)
        return(l.sum(dim=-1))

    def calc_nb_loss(self, ld, norm_mat, theta, obs):
        ld = norm_mat * ld + 1.0e-16
        theta = theta + 1.0e-16
        lp =  ld.log() - (theta).log()
        p_z = dist.NegativeBinomial(theta, logits=lp)
        l = - p_z.log_prob(obs)
        return(l.sum(dim=-1))

    def forward(self, s, u, batch_onehot):
        if self.batch_key is None:
            z, qz = self.enc_z(s, u)
            s_hat = self.dec_z(z)
            d, qd = self.enc_d(z)
            diff_px_zd_ld = self.calculate_diff_x_grad(z, d)
            each_beta = self.dec_b(z) * self.dt
            each_gamma = self.dec_g(z) * self.dt
        else:
            ub = torch.cat([u, batch_onehot], dim=-1)
            z, qz = self.enc_z(s, ub)
            s_hat = self.dec_z(z, batch_onehot)
            d, qd = self.enc_d(z)
            diff_px_zd_ld = self.calculate_diff_x_grad_onehot(z, batch_onehot, d)
            each_beta = self.dec_b(z, batch_onehot) * self.dt
            each_gamma = self.dec_g(z, batch_onehot) * self.dt
        beta = self.softplus(self.logbeta) * self.dt
        gamma = self.softplus(self.loggamma) * self.dt
        raw_u_ld = (diff_px_zd_ld + s_hat * gamma) / beta
        pu_zd_ld = raw_u_ld + self.relu(- raw_u_ld).detach()
        if self.kinetics:
            raw_u_ld = (diff_px_zd_ld + s_hat * each_gamma) / each_beta
            pu_zd_ld = raw_u_ld + self.relu(- raw_u_ld).detach()
        return(z, d, qz, qd, s_hat, diff_px_zd_ld, pu_zd_ld)

    def elbo_loss(self, s, u, norm_mat, norm_mat_u, batch_onehot):
        z, d, qz, qd, s_hat, diff_px_zd_ld, pu_zd_ld =  self(s, u, batch_onehot)
        if self.loss_mode == 'poisson':
            loss_func = lambda ld: self.calc_poisson_loss(ld, norm_mat, s)
            loss_func_u = lambda ld: self.calc_poisson_loss(ld, norm_mat_u, u)
        elif self.loss_mode == 'nb':
            theta = self.softplus(self.logtheta)
            loss_func = lambda ld: self.calc_nb_loss(ld, norm_mat, theta, s)
            loss_func_u = lambda ld: self.calc_nb_loss(ld, norm_mat_u, theta, u)
        z_kld = self.calc_z_d_kld(qz)
        d_kld = self.calc_z_d_kld(qd)
        lx = loss_func(s_hat)
        lu = loss_func_u(pu_zd_ld)
        elbo_loss = torch.mean(z_kld + d_kld + lx + lu)
        return(elbo_loss)