from math import fabs
from turtle import pd

AA_VOCAB = {
    "A":1,
    "R":2,
    "N":3, 
    "D":4,
    "C":5,
    "Q":6,
    "E":7,
    "G":8,
    "H":9,
    "I":10,
    "L":11,
    "K":12,
    "M":13,
    "F":14,
    "P":15,
    "S":16,
    "T":17,
    "W":18,
    "Y":19,
    "V":20,

    "X":21,  # <END>

}

R_AA_VOCAB = {v:k for k,v in AA_VOCAB.items()}

class configuration():
    def __init__(self,
                    hidden_size: int = 768,
                    max_position_embeddings: int = 263,
                    type_residue_size: int = 9,
                    layer_norm_eps: float = 1e-12,
                    hidden_dropout_prob = 0.1,
                    position_embedding_type = "absolute",
                    num_attention_heads: int = 12,
                    use_bias = True,
                    attention_probs_dropout_prob = 0.1,
                    intermediate_size=3072, 
                    hidden_act="gelu_new",
                    add_cross_attention = False,
                    initializer_range=0.02,
                    num_hidden_layers = 4,
                    maxlen_HCDR3 = 30,
                    one_hot = False,
                    C_num_class=1,
                    type_embedding=False,
                    region_info = False,
                    structure = False,
                    cross = False,
                    Esm_length = 1000,
                    use_aa_feature = False,
                    ) -> None:
        
        self.AA_VOCAB = AA_VOCAB
        self.token_size = len(self.AA_VOCAB)
        self.residue_size = 21
        self.hidden_size = hidden_size
        self.pad_token_id = 0
        self.max_position_embeddings = max_position_embeddings
        self.type_residue_size = type_residue_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.position_embedding_type= position_embedding_type
        self.num_attention_heads = num_attention_heads
        self.use__bias = use_bias
        self.attention_probs_drogout_prob = attention_probs_dropout_prob
        self.intermediate_size=intermediate_size
        self.hidden_act=hidden_act
        self.add_cross_attention = add_cross_attention
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.maxlen_HCDR3 = maxlen_HCDR3
        self.one_hot = one_hot
        self.C_num_class = C_num_class
        self.type_embedding = type_embedding
        self.region_info = region_info
        self.structure = structure
        self.cross = cross
        self.Esm_length=Esm_length
        self.use_aa_feature = use_aa_feature