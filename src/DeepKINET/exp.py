import torch
from modules import DeepKINET
from dataset import DeepKINETDataManager
import numpy as np
from statistics import mean

class EarlyStopping:
    def __init__(self, patience, path):
        self.patience = patience    #設定ストップカウンタ
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        score =  - val_loss
        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット
    def checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

class DeepKINETExperiment:
    def __init__(self, model_params, lr, s, u,  test_ratio, batch_size, num_workers, checkpoint, validation_ratio):
        self.edm = DeepKINETDataManager(s, u, test_ratio, batch_size, num_workers, validation_ratio)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepKINET(**model_params)
        self.model_params = model_params
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.checkpoint=checkpoint
        self.lr = lr

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        entry_num = 0
        for s, u, norm_mat, norm_mat_u in self.edm.train_loader:
            s = s.to(self.device)
            u = u.to(self.device)
            norm_mat = norm_mat.to(self.device)
            norm_mat_u = norm_mat_u.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.elbo_loss(s, u, norm_mat, norm_mat_u)
            loss.backward()
            self.optimizer.step()
            total_loss = total_loss + loss.item()
            entry_num += s.shape[0]
        return(total_loss / entry_num)

    def evaluate(self):
        self.model.eval()
        s = self.edm.validation_s.to(self.device)
        u = self.edm.validation_u.to(self.device)
        norm_mat = self.edm.validation_norm_mat.to(self.device)
        norm_mat_u = self.edm.validation_norm_mat_u.to(self.device)
        loss = self.model.elbo_loss(s, u, norm_mat, norm_mat_u)
        entry_num = s.shape[0]
        return(loss / entry_num)

    def test(self):
        self.model.eval()
        s = self.edm.test_s.to(self.device)
        u = self.edm.test_u.to(self.device)
        norm_mat = self.edm.test_norm_mat.to(self.device)
        norm_mat_u = self.edm.test_norm_mat_u.to(self.device)
        loss = self.model.elbo_loss(s, u, norm_mat, norm_mat_u)
        entry_num = s.shape[0]
        return(loss / entry_num)

    def train_total(self, epoch_num, patience):
        earlystopping = EarlyStopping(patience=patience, path=self.checkpoint)
        val_loss_list = []
        for epoch in range(epoch_num):
            loss = self.train_epoch()
            val_loss = self.evaluate()
            val_loss_list.append(val_loss.item())
            val_loss_mean_num = 10
            val_loss_post_mean = mean(val_loss_list[-val_loss_mean_num:])
            if epoch % 100 == 0:
              print(f'val loss mean (post {val_loss_mean_num} epochs) at epoch {epoch} is {val_loss_post_mean}')
            earlystopping(val_loss_post_mean, self.model) #callメソッド呼び出し
            if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
              print(f"Early Stopping! at {epoch} epoch")
              break

    def init_optimizer(self, lr):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)