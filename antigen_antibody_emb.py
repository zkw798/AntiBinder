import torch
import torch.nn as nn
import torch.nn. functional as F
import pandas as pd
import sys
sys.path.append("/AntiBinder")
from configuration_antibody_vat import AA_VOCAB
from configuration_antibody_vat import configuration
import pdb
from math import ceil
import esm
import os
import lmdb
import pickle
from igfold import IgFoldRunner


class antibody_antigen_dataset(nn.Module):
    def __init__(self,
                antigen_config: configuration, 
                antibody_config: configuration, 
                data_path=None,
                train = True,
                test = False, 
                rate1 = 0.8,
                data = None) -> None:
        super().__init__()
        self.antigen_config = antigen_config
        self.antibody_config = antibody_config
        print (data_path)
        if isinstance(data,pd.DataFrame):
            df = data
            df.dropna()
        else:
            df = pd.read_csv(data_path)## samples of data, attention to your file type
            
            # df = df.dropna()
            df= df[df['antigen_name']=='Cov']
            df = df.dropna(subset=['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4'])
            # df = df[df['Target'].isin(['SARS-CoV2_WT', 'SARS-CoV2_Beta','SARS-CoV2_Alpha',
            #                             'SARS-CoV2_Delta',
            #                             'SARS-CoV2_Kappa','SARS-CoV2_Mu','SARS-CoV2_Gamma'])]
            # df = df[df['Target'].isin(['SARS-CoV2_Omicron'])]
            # df = df.sample(n=5000,random_state=42)

            # df = df.dropna(subset=['Species'])
            # # df = df[df['Species'].apply(lambda x:"Homo sapiens" not in x and "Mus musculus" in x)]
            # df = df[df['Species']== "Homo sapiens"]
            # print(df['Target'].unique())
        self.antigen_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()

        if train==True and test==False: # train
            print("This part of dataset is for train.")
            self.data = df.iloc[:int(df.shape[0]*(rate1))]
            print("len train data", len(self.data)) 

        elif train==False and test == True: # val
            print("This part of dataset is for test.")
            self.data = df.iloc[int(df.shape[0]*(rate1)):]
            print("len test data",len(self.data))

        self.env = None
        self.structure_embedding = None
        self.igfold = IgFoldRunner()


    def padding(self, seq, maxlen):
        if len(seq) > maxlen:
            return seq[:maxlen]
        else:
            return torch.cat([seq,torch.zeros(maxlen-len(seq))]).long()
    

    def padding_esm(self, seq, maxlen):
        if len(seq) > maxlen:
            return seq[:maxlen]
        else:
            return seq+'<pad>'*(maxlen-len(seq)-2)
        

    def type_position(self,index):
        data = self.data.iloc[index]
        HF1 = [1 for _ in range(len(data['H-FR1']))]
        HCDR1 = [3 for _ in range(len (data['H-CDR1']))]
        HF2 = [1 for _ in range(len(data['H-FR2']))]
        HCDR2 = [4 for _ in range(len(data['H-CDR2']))]
        HF3 = [1 for _ in range(len(data['H-FR3']))]
        HCDR3 = [5 for _ in range(len(data['H-CDR3']))]
        HF4 = [1 for _ in range(len(data['H-FR4']))]
        vh = torch.tensor(list(HF1+HCDR1+HF2+HCDR2+HF3+HCDR3+HF4))
        vh = self.padding(vh,self.antibody_config.max_position_embeddings)

        return vh
    

    def __getitem__(self, index):
        data = self.data.iloc[index]
        label = torch.tensor(data['ANT_Binding'])
        if not os.path.exists('/AntiBinder/antigen_esm/train/'+str(self.data.iloc[index]['Antigen'])+'.pt'):
            antigen = self.padding_esm(self.data['Antigen Sequence'].iloc[index], self.antigen_config.max_position_embeddings)
            antigen = [('antigen', antigen)]
            batch_labels, batch_strs, antigen = self.batch_converter(antigen)
            with torch.no_grad():
                self.antigen_model = self.antigen_model.eval()
                antigen = self.antigen_model(antigen.squeeze(1), repr_layers=[33], return_contacts=True)
                # antigen = antigen[ 'representations'][33][:,:len(self.data['Antigen Sequence'].iLoc[index])].norm(dim=1).view(-1)
                antigen = antigen['representations'][33].squeeze(0) # shape: [1024, 1280]

            torch.save(antigen,'/AntiBinder/antigen_esm/train/'+str(self.data.iloc[index]['Antigen'])+'.pt')
        
        antigen_structure = torch.load("/AntiBinder/antigen_esm/train/"+str(self.data.iloc[index]['Antigen'])+'.pt')

        # print("antigen_structure："，antigen_structure)#含离散值Btenson
        # print("antigen_structure shape: ", antigen_structure.shape) # [1024, 1280]
        

        emb_seq = data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3'] + data['H-CDR3']+data['H-FR4']
        #if not emb_seq in self.structure_embedding.keys():
        self.env = lmdb.open('/AntiBinder/datasets/fold_emb/fold_emb_for_train',map_size=1024*1024*1024*50,lock=False)
        self.structure_embedding = self.env.begin(write=True)
        if self.structure_embedding.get(emb_seq.encode()) == None:
            sequences = {
                "H": emb_seq
                }
            emb = self.igfold.embed(
                sequences=sequences, # Antibody sequences
                )
            structure = emb.structure_embs.detach().cpu()
            #self.structure_embedding[emb_seq] = structure
            self.structure_embedding.put(key=emb_seq.encode(), value=pickle.dumps(structure)) 
            self.structure_embedding.commit()
        else:
            # print("Structure is not none.")
            structure = pickle.loads(self.structure_embedding.get(emb_seq.encode()))
            # print("stru"，structure) #含离散值的tensor
            # print("stru.shape"，structure.shape)#［1,118,64］ 第1维长度不一定
        self.env.close()

        st_l = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3'])
        st_r = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3']+ data['H-CDR3'])
        st_e = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3']+ data['H-CDR3'] + data['H-FR4'])


        # print("pads_zero", pads_zero)
        # print("pads_zero.shape"，pads_zero.shape)
        structure = torch.cat((structure[:,:st_l,:],structure[:,st_l:st_r,:].mean(1).unsqueeze(1),structure[:,st_r:st_e,:]),dim=1)
        # print("stru.shape", structure.shape) # [150, 64]
        pads_zero = torch.zeros(1,self.antibody_config.max_position_embeddings-structure.shape[1], structure.shape[-1]).float()
        structure = torch.cat((structure, pads_zero) ,dim=1).squeeze(0)


        antibody = data['vh']
        # print("antibody："，antibody)
        antibody = torch.tensor([AA_VOCAB[aa] for aa in antibody])
        # print("antibody："，antibody)
        antibody = self.padding(seq=antibody, maxlen=self.antibody_config.max_position_embeddings)
        # print("antibody: ", antibody)
        # print("antibody_shape:", antibody. shape) #［150]

        at_type = self.type_position(index)
        # print(type)
        # print(type.shape)  # [150]
        antibody_structure = structure

        antigen = data['Antigen Sequence']
        antigen = torch.tensor([AA_VOCAB[aa] for aa in antigen])
        antigen = self.padding(seq=antigen, maxlen=self.antigen_config.max_position_embeddings)

        antigen_structure = antigen_structure[:1024, :]
        # print(antigen_structure.shape)
        # ([150] [150] ,[150,64]),([1024],[1024,1280])
        return [antibody,at_type,antibody_structure],[antigen,antigen_structure],label

    def __len__(self):
        return self.data.shape[0]

if __name__ == "__main__":
    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings',1024)
    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings', 150)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    data_path = '/AntiBinder/datasets/xx'
    dataset = antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config, data_path=data_path, train=True, test=False, rate1=0.0001)
    # pdb. set_trace()
    x1 = dataset[0]
 
    pdb.set_trace()

