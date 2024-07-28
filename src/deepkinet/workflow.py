import torch
import utils
import numpy as np

def estimate_kinetics(adata, epoch = 2000, learned_checkpoint = None, loss_mode = 'poisson', checkpoint='.deepkinet_opt.pt', lr = 0.001, weight_decay=0.01, batch_key=None):

    model_params = {}
    model_params['x_dim'] = adata.n_vars
    model_params['loss_mode'] = loss_mode
    model_params['batch_key'] = batch_key
    batch_onehot = make_sample_one_hot_mat(adata, batch_key)
    model_params['batch_onehot'] = batch_onehot
    
    deepkinet_exp = utils.define_exp(adata, model_params = model_params, lr = lr, weight_decay = weight_decay, val_ratio=0.10, test_ratio = 0.05, batch_size = 100, num_workers = 2, checkpoint = checkpoint)

    adata.uns['Dynamics_last_val_loss'] = np.array([0])
    adata.uns['Dynamics_last_test_loss'] = np.array([0])
    adata.uns['Kinetics_last_val_loss'] = np.array([0])
    adata.uns['Kinetics_last_test_loss'] = np.array([0])
    patience = 10

    if learned_checkpoint==None:

        print('Start Dynamics opt')
        print('Dynamics opt patience',patience)
        deepkinet_exp.model.dynamics = True
        deepkinet_exp.model.kinetics = False
        deepkinet_exp.init_optimizer(lr, weight_decay)
        deepkinet_exp.train_total(epoch, patience)
        deepkinet_exp.model.load_state_dict(torch.load(checkpoint))
        print('Done Dynamics opt')
        dynamics_last_val_loss=deepkinet_exp.evaluate().cpu().detach().numpy()
        dynamics_last_test_loss=deepkinet_exp.test().cpu().detach().numpy()
        print(f'Dynamics_last_val_loss:{dynamics_last_val_loss}')
        print(f'Dynamics_last_test_loss:{dynamics_last_test_loss}')
        adata.uns['Dynamics_last_val_loss']=dynamics_last_val_loss
        adata.uns['Dynamics_last_test_loss']=dynamics_last_test_loss


        print('Start Kinetics opt')
        print('Kinetics opt patience',patience)
        deepkinet_exp.model.dynamics = False
        deepkinet_exp.model.kinetics = True
        for param in deepkinet_exp.model.parameters():
            param.requires_grad = False
        for param in deepkinet_exp.model.dec_b.parameters():
            param.requires_grad = True
        for param in deepkinet_exp.model.dec_g.parameters():
            param.requires_grad = True
        deepkinet_exp.init_optimizer(lr, weight_decay)
        deepkinet_exp.train_total(epoch, patience)
        deepkinet_exp.model.load_state_dict(torch.load(checkpoint))
        print('Done Kinetics opt')
        kinetics_last_val_loss=deepkinet_exp.evaluate().cpu().detach().numpy()
        kinetics_last_test_loss=deepkinet_exp.test().cpu().detach().numpy()
        print(f'Kinetics_last_val_loss:{kinetics_last_val_loss}')
        print(f'Kinetics_last_test_loss:{kinetics_last_test_loss}')
        adata.uns['Kinetics_last_val_loss']=kinetics_last_val_loss
        adata.uns['Kinetics_last_test_loss']=kinetics_last_test_loss

        adata.uns['param_path'] = checkpoint
    else:
        deepkinet_exp.model.load_state_dict(torch.load(learned_checkpoint))
        adata.uns['param_path'] = learned_checkpoint


    results_dict = utils.post_process(adata, deepkinet_exp)
    return(adata, deepkinet_exp)


def make_sample_one_hot_mat(adata, sample_key):
    print('make_sample_one_hot_mat')
    if sample_key is not None:
        sidxs = np.sort(adata.obs[sample_key].unique())
        b = np.array([
            (sidxs == sidx).astype(int)
            for sidx in adata.obs[sample_key]]).astype(float)
        b = torch.tensor(b).float()
    else:
        b = np.zeros((len(adata.obs_names), 1))
        b = torch.tensor(b).float()
    return b