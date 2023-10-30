import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions.kl import kl_divergence
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
        mu = self.h2mu(h)#意外とこの隠れ層→隠れ層って構造なくてもいい気がする。
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
        mu = self.h2mu(post_h)#意外とこの隠れ層→隠れ層って構造なくてもいい気がする。
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

class VicDyf(nn.Module):
    def __init__(self, x_dim, loss_mode, z_dim = 20, h_dim = 100, enc_z_layers = 2):
        super(VicDyf, self).__init__()
        self.enc_z = Encoder_of_s_u(enc_z_layers, x_dim, h_dim, z_dim)
        self.enc_d = Encoder(z_dim, h_dim, z_dim)
        self.dec_z = Decoder(z_dim, h_dim, x_dim)
        self.dec_b = Decoder(z_dim, h_dim, x_dim)
        self.dec_g = Decoder(z_dim, h_dim, x_dim)
        self.dt = 1
        self.d_coeff = 0.01
        self.loggamma = Parameter(torch.Tensor(x_dim))
        self.logbeta = Parameter(torch.Tensor(x_dim))
        self.logtheta = Parameter(torch.Tensor(x_dim))
        self.softplus = nn.Softplus()
        self.dynamics = False
        self.kinetics_rates = False
        self.relu = nn.ReLU()
        self.loss_mode = loss_mode
        print('loss_mode',self.loss_mode)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.loggamma)
        init.normal_(self.logbeta)
        init.normal_(self.logtheta)

    def calc_z_d_kld(self, qz):
      #kl_divergence(qz, dist.Normal(*self.pz_params)).sum(-1)
        kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
        return(kld.sum(dim=-1))

    def calculate_diff_x_grad(self, z, d):
        dec_f = lambda vz: self.dec_z(vz)
        #print(functorch.jvp(dec_f, (z, ), (d, ))[0].shape,functorch.jvp(dec_f, (z, ), (d, ))[1].shape) #torch.Size([100, 2000]) torch.Size([100, 2000])
        dec_jvp = lambda vz, vd: functorch.jvp(dec_f, (vz, ), (vd, ))[1]#(z, d) → (f(z), df(z)*d) #df(z)は押し出しにより 10 → 2000次元であり、空間R10でのzの接空間を空間R2000での出力点f(z)の接空間に移す線型写像 R 10*2000 dims
        diff_px_zd_ld = self.d_coeff * functorch.vmap(dec_jvp, in_dims=(0, 0))(z, d) #(100, 2000)
        return diff_px_zd_ld

    def calculate_diff_x_std(self, z, dscale): #zのsizeは100 * 10とかcells *10のはず
        d_id = torch.eye(z.size()[1], z.size()[1]).to(z.device) #10*10の単位行列作る(対角成分が1で他は0)
        gene_vel_std = sum([
            (self.calculate_diff_x_grad(z, repeat(delta_d, 'd -> b d', b=z.size()[0])) * dscale[:, i].unsqueeze(1))**2
            for delta_d, i in zip(d_id, range(dscale.size()[1]))]).sqrt()
        return gene_vel_std

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

    def forward(self, s, u):
        z, qz = self.enc_z(s, u)
        dz, qd = self.enc_d(z)
        s_hat = self.dec_z(z)
        diff_px_zd_ld = self.calculate_diff_x_grad(z, dz)
        beta = self.softplus(self.logbeta) * self.dt
        gamma = self.softplus(self.loggamma) * self.dt
        raw_u_ld = (diff_px_zd_ld + s_hat * gamma) / beta
        pu_zd_ld = raw_u_ld + self.relu(- raw_u_ld).detach()

        if self.kinetics_rates:
            each_beta = self.dec_b(z) * self.dt
            each_gamma = self.dec_g(z) * self.dt
            raw_u_ld = (diff_px_zd_ld + s_hat * each_gamma) / each_beta
            pu_zd_ld = raw_u_ld + self.relu(- raw_u_ld).detach()

        return(z, dz, qz, qd, s_hat, diff_px_zd_ld, pu_zd_ld)

    def elbo_loss(self, s, u, norm_mat, norm_mat_u):
        z, dz, qz, qd, s_hat, diff_px_zd_ld, pu_zd_ld =  self(s, u)

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