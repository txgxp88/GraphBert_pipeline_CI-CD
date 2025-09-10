import torch 
from transformers.configuration_utils import PretrainedConfig

class GraphBertConfig(PretrainedConfig):
    # default values
    def __init__(
        self,
        residual_type = 'none',
        x_size=3000,
        y_size=7,
        k=5,
        max_wl_role_index = 100,
        max_hop_dis_index = 100,
        max_inti_pos_index = 100,
        # max_attr_dis_index = 100,
        hidden_size=32,#32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,#32,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,#0.5,
        attention_probs_dropout_prob=0.5,#0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,
        input_similarity = None,
        input_feature = None,
        **kwargs
    ):
        super(GraphBertConfig, self).__init__(**kwargs)
        self.max_wl_role_index = max_wl_role_index
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        # self.max_attr_dis_index = max_attr_dis_index
        self.residual_type = residual_type
        self.x_size = x_size
        self.y_size = y_size
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder
        self.input_similarity = input_similarity
        self.input_feature = input_feature



def step_3(data, args):
    # Graph model settings

    # Cora config
    nclass = args.nclass
    # len(data.y.unique()) # for cora dataset
    nfeature = args.nfeature
    # data.x.shape[1]
    ngraph = args.ngraph
    # data.x.shape[0]

    #
    x_size = nfeature
    y_size = nclass
    graph_size = ngraph
    residual_type = args.residual_type
    # residual_type = 'graph_raw'
    # residual_type = 'raw'
    # residual_type = 'none'

    #Bert Config
    max_wl_role_index = args.max_wl_role_index
    # 100
    max_hop_dis_index = args.max_hop_dis_index
    # 100
    # max_attr_dis_index = embedding_dimension+1
    max_inti_pos_index = args.max_inti_pos_index
    # 100
    residual_type = residual_type
    x_size = x_size
    y_size = y_size
    k = nclass#Embedding dimension


    # Network setting
    hidden_size = args.hidden_size
    # 32 #32
    num_hidden_layers = args.num_hidden_layers
    # 1 #2
    num_attention_heads = args.num_attention_heads
    # 4 #2
    hidden_act = args.hidden_act
    # 'gelu'
    intermediate_size = args.intermediate_size
    # 128 #32: 2*hidden_size
    hidden_dropout_prob = args.hidden_dropout_prob
    # 0.2#0.5
    attention_probs_dropout_prob = args.attention_probs_dropout_prob
    # 0.2#0.3
    initializer_range = args.initializer_range
    # 0.02
    layer_norm_eps = args.layer_norm_eps
    # 1e-12


    class Config:
        pass  # simple container

    config = Config()

    # Input embedding
    config.x_size = x_size
    config.hidden_size = hidden_size
    config.max_wl_role_index = max_wl_role_index
    config.max_inti_pos_index = max_inti_pos_index
    config.max_hop_dis_index = max_hop_dis_index
    # config.max_attr_dis_index = max_attr_dis_index
    config.layer_norm_eps = layer_norm_eps
    config.hidden_dropout_prob = hidden_dropout_prob

    # Encoder
    config.output_attentions = False
    config.output_hidden_states = False
    config.num_hidden_layers = num_hidden_layers
    


    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, \
                                hidden_size=hidden_size, intermediate_size=intermediate_size, \
                                num_attention_heads=num_attention_heads, \
                                num_hidden_layers=num_hidden_layers, \
                                input_similarity = data.adj, \
                                input_feature = data.x 
                                )

    return bert_config



