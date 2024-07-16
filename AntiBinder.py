from torch.distributions import Normal
import torch
import torch.nn as nn 
from antigen_antibody_emb import *
from configuration_antibody_vat import *
from torch.utils.data import DataLoader
os.chdir('/AntiBinder')


class AntiModelIinitial():
    def __init__(self, initializer_range=0.02) -> None: ## 0.02
        self.initializer_range = initializer_range

    def _init_weights(self, module):
        """Initialize the weights""" 
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None: 
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1) :
        super().__init__()
        # Define multi-head attention Layers for both directions
        self.antibody_to_antigen_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout,bias=True, add_bias_kv=False)
        self.antigen_to_antibody_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout,bias=True, add_bias_kv=False)

    def forward (self, antibody_embed, antigen_embed, antibody_mask=None, antigen_mask=None) :
        # Antibody to Antigen Attention
        antibody_as_query = antibody_embed.permute(1, 0, 2)
        antigen_as_kv = antigen_embed.permute(1, 0, 2)
        attn_output_antibody, attn_weights_antibody = self.antibody_to_antigen_attention(
            query=antibody_as_query, 
            key=antigen_as_kv,
            value=antigen_as_kv,
            key_padding_mask=antigen_mask
            )
        
        # Antigen to Antibody Attention
        antigen_as_query = antigen_embed.permute(1, 0, 2)
        antibody_as_kv = antibody_embed.permute(1, 0, 2)
        attn_output_antigen, attn_weights_antigen = self.antigen_to_antibody_attention(
            query=antigen_as_query,
            key=antibody_as_kv,
            value=antibody_as_kv, 
            key_padding_mask=antibody_mask
        )
        return attn_output_antibody.permute(1, 0, 2), attn_output_antigen.permute(1, 0, 2)


class AntiEmbeddings(nn.Module) :
    def __init__(self, hidden_size=1024, token_size=22,type_residue_size=22,layer_norm_eps=1e-12) -> None:
        super().__init__()
        self.residue_embedding = nn.Embedding(token_size,hidden_size)
        self.token_type_embeddings = nn.Embedding(type_residue_size,hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(0.1)


    def forward(self,seq=None,type=None):
        # seq type embedding
        seq_embeddings = self.residue_embedding(seq)

        if type is not None:
            # token type embedding
            token_type_embeddings = self.token_type_embeddings(type)
            embeddings = seq_embeddings + token_type_embeddings

        else:
            embeddings = seq_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class pool(nn.Module):
    def __init__(self,latent_dim) -> None:
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(149*latent_dim, latent_dim*latent_dim),nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(1024*latent_dim, latent_dim*latent_dim),nn.ReLU(inplace=True))

    def forward(self,input,latent_dim,is_antibody=True):
        batch_size = input.size(0)

        input = input.view(batch_size, -1)
        if is_antibody:
            output = self.linear1(input)
    
        else:
            output = self.linear2(input)

        output_tensor = output.view(batch_size, latent_dim, latent_dim)
        return output_tensor
    

class antibody_sturcture_change_dim(nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = nn.Linear(64,1024)
        self.relu = nn.ReLU()

    def forward(self,antibody):
        return self.relu(self.linear(antibody))


class Combine_Embedding(nn.Module):
    def __init__(self,antibody_hidden_dim,antigen_hidden_dim):
        super().__init__()
        self.seq_emb = AntiEmbeddings()
        self.antibody_sturcture_change_dim = antibody_sturcture_change_dim()
        self.antigen_sturcture_change_dim = nn.Sequential(nn.Linear(1280,antigen_hidden_dim),nn.ReLU())

        
    def forward(self, antibody, antigen):  
        # antibody: [antibody,at_type,antibody_structure]
        # antigen : [antigen,antigen_structure]
        antibody[0], antibody[1], antibody[2] = antibody[0].cuda(), antibody[1].cuda(), antibody[2].cuda()
        antigen[0],antigen[1] = antigen[0].cuda(),antigen[1].cuda()



        antibody_seq_emb = self.seq_emb(seq = antibody[0], type = antibody[1])  # shape: [batch_size, 141, antigen_hidden_dim (1024)]
        # print(antibody[2].shape)
        antibody_structure = self.antibody_sturcture_change_dim(antibody[2])  # shape: [batch_size, 141, antibody_hidden_dim(1024)]
        antigen_seq_emb = self.seq_emb(seq = antigen[0])  # shape: [batch_size, 2000, antigen_hidden_dim(1024)]
        antigen_structure = self.antigen_sturcture_change_dim(antigen[1])  # shape: [batch_size, 2000, antigen_hidden_dim(1024)]

        antibody_seq_plus_stru = antibody_seq_emb + antibody_structure
        antigen_seq_plus_stru = antigen_seq_emb + antigen_structure

        return antibody_seq_plus_stru,antigen_seq_plus_stru
    

class bicrossatt(nn.Module):
    def __init__(self, antibody_hidden_dim,latent_dim = 64) -> None:
        super().__init__()
        self.bidirectional_crossatt = BidirectionalCrossAttention(embed_dim=antibody_hidden_dim, num_heads=1)
        self.LayerNorm = nn.LayerNorm(antibody_hidden_dim, eps=1e-12)
        self.linear = nn.Sequential(nn.Linear(1024,1024),nn.ReLU(inplace=True))
        self.change_dim = nn.Sequential(nn.Linear(1024,latent_dim),nn.ReLU(inplace=True))
        self.pool = pool(latent_dim)
        self.flatten = nn.Flatten()
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.latent_dim = latent_dim

    def forward(self, antibody_seq_stru, antigen_seq_stru): # [batch,150,1024],[batch,2000,1024]
        antibody_seq_stru,antigen_seq_stru = self.bidirectional_crossatt(antibody_seq_stru,antigen_seq_stru)

        antibody_seq_stru = self.change_dim(self.linear(self.LayerNorm(antibody_seq_stru)))  # [batch,150,64]
        antigen_seq_stru = self.change_dim(self.linear(self.LayerNorm(antigen_seq_stru)))   # [batch,2000,64]

        antibody_seq_stru = self.pool(antibody_seq_stru,self.latent_dim,is_antibody=True)  # [batch,64,64]
        antigen_seq_stru = self.pool(antigen_seq_stru,self.latent_dim,is_antibody=False)  # [batch,64,64]

        antibody_seq_stru = self.flatten(antibody_seq_stru)  # [batch,4096]
        antigen_seq_stru = self.flatten(antigen_seq_stru)  # [batch,4096]

        concatenated_tensor = torch.cat((antibody_seq_stru, self.alpha*antigen_seq_stru), dim=-1)
        return concatenated_tensor


class antibinder(nn.Module):
    def __init__(self,antibody_hidden_dim,antigen_hidden_dim,latent_dim) -> None:
        super().__init__()
        self.combined_embedding = Combine_Embedding(antibody_hidden_dim,antigen_hidden_dim)
        self.bicrossatt = bicrossatt(antibody_hidden_dim,latent_dim)
        self.cls = nn.Sequential(
            nn.Linear(latent_dim*latent_dim*2,latent_dim*latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim*latent_dim,1),
            nn.Sigmoid()
            )


    def forward(self, antibody, antigen): 
        # antibody: [antibody,at_type,antibody_structure]
        # antigen : [antigen,antigen_structure]
        antibody_seq_stru,antigen_seq_stru = self.combined_embedding(antibody, antigen) # [batch,150,1024],[batch,2000,1024]
        concat_tensor = self.bicrossatt(antibody_seq_stru,antigen_seq_stru)

        return self.cls(concat_tensor)


if __name__ == "__main__":
    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings',1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings', 149)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    # data_path = '/AntiBinder/datasets/combined_all_for_train/combined_for_train.csv'
    # dataset = antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config, data_path=data_path, train=True, test=False, rate1=0.0001)
    # # pdb. set_trace()
    # x1 = dataset[0]
    model = AntiBinder(antibody_hidden_dim=1024,antigen_hidden_dim=1024,latent_dim=32)
    print(model)
    model.combined_embedding = torch.nn.DataParallel(model.combined_embedding).cuda()
    model.bicrossatt = torch.nn.DataParallel(model.bicrossatt).cuda()
    model.cls = torch.nn.DataParallel(model.cls).cuda()

    x1 = torch.randint(low=0, high=10,size=(3,149))
    x2 = torch.randint(low=0, high=10,size=(3,149))
    x3 = torch.rand(3,149,64)

    x4 = torch.randint(low=0, high=10,size=(3,1024))
    x5 = torch.rand(3,1024,1280)

    lst1 = [x1,x2,x3]
    lst2 = [x4,x5]

    print(model(lst1,lst2))



