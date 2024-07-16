import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from antigen_antibody_emb import * 
from AntiBinder import *
import torch
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader 
from copy import deepcopy 
from tqdm import tqdm
import sys 
import argparse
from utils.utils import CSVLogger_my
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score
sys.path.append ('../') 
import warnings 
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self,model,train_dataloader,args,logger,load=False) -> None:
        self.model = model
        self. train_dataloader = train_dataloader
        # self. vaLid_dataloader = valid_dataloader
        # self. test_dataloader = test_dataloader
        self.args = args
        self.logger = logger
        # self.grad_clip = args.grad_clip # cLip gradients at this value, or disable if == 0.0
        self.best_loss = None
        self.load = load

        if self.load==False:
            self.init()
        else:
            print("no init model")

    def init(self):
        init = AntiModelIinitial()
        self.model.apply(init._init_weights)
        print("init successfully!")


    def matrix(self,yhat,y):
        return sum (y==yhat)
    

    def matrix_val(self,yhat,y) :
        # print(sum(yhat))
        return accuracy_score(y,yhat), precision_score(y, yhat), f1_score(y,yhat), recall_score(y, yhat)
    

    def train(self, criterion, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
        for epoch in range(epochs) :
            self.model.train(True)
            train_acc = 0
            train_loss = 0
            num_train = 0
            Y_hat = []
            Y = []
            for antibody_set, antigen_set, label in tqdm(self.train_dataloader):
                probs = self.model(antibody_set, antigen_set)

                yhat = (probs>0.5).long()
                y = label.float().cuda()
                loss = criterion(probs.view(-1),y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_train += antibody_set[0].shape[0]
                Y_hat.extend(yhat)
                Y.extend(y)

            train_acc, train_precision, train_f1, recall = self.matrix_val((torch.cat([temp.view(1, -1) for temp in Y_hat], dim=0)).long().cpu().numpy(),
                                                                            torch.tensor(Y))
            train_loss = train_loss / num_train
            train_loss = np.exp(train_loss)

            self.logger.log([epoch+1, train_loss, train_acc, train_precision,train_f1,recall])

            if self.best_loss==None or train_loss < self.best_loss:
                print('epoch: ',epoch, 'saving...')
                self.best_loss = train_loss
                self.save_model()


    def save_model(self):
        torch.save(self.model.state_dict(),f"/AntiBinder/ckpts/{self.args.model_name}_{self.args.data}_{self.args.batch_size}_{self.args.epochs}_{self.args.latent_dim}_{self.args.lr}.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--latent_dim', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay used in optimizer') # 1e-5
    parser.add_argument('--lr', type=float, default=6e-5, help='learning rate') # 6e-7
    parser.add_argument('--model_name', type=str, default= 'AntiBinder')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--data', type=str, default='train')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings',1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings',149)

    model = antibinder(antibody_hidden_dim=1024,antigen_hidden_dim=1024,latent_dim=args.latent_dim).cuda()
    print(model)

    # # muti-gpus
    # model.combined_embedding = torch.nn.DataParallel(model.combined_embedding).cuda()
    # model.bicrossatt = torch.nn.DataParallel(model.bicrossatt).cuda()
    # model.cls = torch.nn.DataParallel(model.cls).cuda()

    # here choose dataset
    if args.data == 'train':
        data_path = '/AntiBinder/datasets/**'
    # elif args.data == 'train_2':
    #     data_path = ''


    # print (data_path)
    train_dataset = antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config,data_path=data_path, train=True, test=False, rate1=1) # (antibody, type, antibody_structure), antigen, Label
    # vaL_dataset =antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config,data_path=data path, train=False, test=True, rate1=0.7) # (antibody, type, antibody_structure), antigen, Label

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    # vaL_dataloader = DataLoader(val_dataset, shuffLe=False, batch_size=args.batch_size)
  

    logger = CSVLogger_my(['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall'],f"/home/zhangkaiwen/AEC/logs/ZKW/{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv")
    scheduler = None

    # load model if needs
    load = False
    if load:
        weight = torch.load('')
        model.load_state_dict(weight)
        print("load model success")


    trainer = Trainer(model=model,
        train_dataloader=train_dataloader,
        # valid_dataloader=val_dataLoader,
        # test_dataLoader=test._dataloader,
        logger = logger,
        args= args,
        load=load
        )

    criterion = nn.BCELoss()
    trainer.train(criterion=criterion, epochs=args.epochs)