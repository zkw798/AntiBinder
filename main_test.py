import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from antigen_antibody_emb import * 
from AntiBinder import *
import torch
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader 
from copy import deepcopy 
from tqdm import tqdm
import os
import sys
import argparse
from utils.utils import CSVLogger_my
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score,roc_auc_score,confusion_matrix
sys.path.append ('../') 
import warnings 
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self,model,valid_dataloader,args,logger) -> None:
        self.model = model
        self.valid_dataloader = valid_dataloader
        self.args = args
        self.logger = logger
        # self.grad_clip = args.grad_clip # cLip gradients at this value, or disable if == 0.0
        self.best_loss = None

    def matrix(self,yhat,y):
        return sum (y==yhat)
    

    def matrix_val(self,yhat,y,yscores) :
        print(sum(yhat))

        TN,FP,FN,TP =0,0,0,0
        cm = confusion_matrix(y, yhat).ravel()
        # print (cm)
        if len(cm)== 1:
            # print (y[0].item(), yhat[0].item())
            if y[0].item() == yhat[0].item()== 0:
                TN= cm[0]
            elif y[0].item() == yhat[0].item() == 1:
                TP = cm[0]
        else:
            TN,FP,FN,TP = confusion_matrix(y,yhat).ravel()
        if len(np.unique(y))>1:
            roc_auc = roc_auc_score(y, yscores)
        else:
            roc_auc = None
        
        return roc_auc, precision_score(y, yhat),accuracy_score(y,yhat), recall_score(y, yhat),f1_score(y,yhat),TN,FP,FN,TP
    

    def valid(self):
        self.model.eval()
        val_acc = 0
        Y_hat = []
        Y = []
        Y_scores = []
        with torch.no_grad():
            for antibody_set, antigen_set, label in tqdm(self.valid_dataloader):
                probs = self.model(antibody_set, antigen_set)
                # print(probs)
                #10*2
                y = label.float()
                yhat = (probs>0.5).long().cuda()
                y_scores = probs

                Y_hat.extend(yhat)
                Y.extend(y)
                Y_scores.extend(y_scores)

        auc, val_prescision, val_acc, recall, val_f1, TN, FP, FN, TP= self.matrix_val((torch.cat([temp.view(1, -1) for temp in Y_hat], dim=0)).long().cpu().numpy(),
                                                                                      torch.tensor(Y),
                                                                                      (torch.cat([temp2.view(1, -1) for temp2 in Y_scores], dim=0)).cpu().numpy())
        return auc, val_prescision, val_acc, recall, val_f1, TN, FP, FN, TP
    

    def train(self):
        val_auc, val_prescision, val_acc, val_recall, val_f1, TN, FP, FN, TP =self.valid()
        self.logger.log([val_auc, val_prescision, val_acc, val_recall, val_f1, TN, FP, FN, TP])

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=36)
    parser.add_argument('--model_name', type=str, default= 'AntiBinder')
    # parser.add_argument('--device', type=str, default='0,1')
    parser.add_argument('--data', type=str, default= 'test')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings', 1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings',149)
 


    model = antibinder(antibody_hidden_dim=1024,antigen_hidden_dim=1024,latent_dim=args.latent_dim).cuda()
    print(model)
   
    
    # load model
    weight = torch.load('')
    model.load_state_dict(weight)
    print("load success")


    # choose test dataset
    if args.data == 'test':
        data_path = ''
  

    print (data_path)
    val_dataset =antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config,data_path=data_path, train=False, test=True, rate1=0) # (antibody, type, antibody_structure), antigen, Label
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
  
    logger = CSVLogger_my(['val_auc', 'val_prescision', 'val_acc', 'val_recall', 'val_f1', 'TN', 'FP', 'FN', 'TP'],f"/AntiBinder/logs/{args.model_name}_{args.latent_dim}_{args.data}.csv")
    
    scheduler = None
    trainer = Trainer(
        model = model,
        valid_dataloader = val_dataloader,
        logger = logger,
        args= args,
        )
    
    trainer.train()
