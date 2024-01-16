import logging
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from .modeling_utils import ProteinConfig
from .modeling_utils import ProteinModel
from .modeling_utils import ValuePredictionHead
from .modeling_utils import SequenceClassificationHead
from .modeling_utils import SequenceToSequenceClassificationHead
from .modeling_utils import PairwiseContactPredictionHead
from ..registry import registry

logger = logging.getLogger(__name__)


URL_PREFIX = "https://s3.amazonaws.com/songlabdata/proteindata/pytorch-models/"
LSTM_PRETRAINED_CONFIG_ARCHIVE_MAP: typing.Dict[str, str] = {}
LSTM_PRETRAINED_MODEL_ARCHIVE_MAP: typing.Dict[str, str] = {}


class ProteinGNNConfig(ProteinConfig):
    pretrained_config_archive_map = LSTM_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 30,
                 input_size: int = 128,
                 hidden_size: int = 1024,
                 hidden_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 network_type: str = "GraphSAGE",
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.network_type = network_type


class ProteinGNNAbstractModel(ProteinModel):

    config_class = ProteinGNNConfig
    pretrained_model_archive_map = LSTM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "gnn"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ProteinGCNModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv1 = GCNConv(config.input_size,config.hidden_size)
        self.conv2 = GCNConv(config.hidden_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x,edge_index,edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x,edge_index,edge_attr)
        # x:[batch_size x node_num x hidden_size]
        return x


class ProteinGraphSAGEModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.sage1 = SAGEConv(config.input_size,config.hidden_size)
        self.sage2 = SAGEConv(config.hidden_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x, edge_index, edge_attr):
        # GraphSAGE原论文未使用边信息
        x = self.sage1(x,edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.sage2(x,edge_index)
        return x
    

class ProteinGATModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gat1 = GATConv(config.input_size,config.hidden_size)
        self.gat2 = GATConv(config.hidden_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.gat1(x,edge_index,edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x,edge_index,edge_attr)
        return x


class ProteinGNNOutput(nn.Module):

    def __init__(self, config):
        super().__init__()

    def forward(self, input_ids, node_index, x, ptr):

        graph_outputs = []
        sequence_outputs = []
        for i in range(len(ptr)-1):
            node_outputs = x[ptr[i]:ptr[i+1]]
            graph_output = torch.sum(node_outputs,dim=0)
            graph_outputs.append(graph_output)

            x_index = list(node_index[ptr[i]:ptr[i+1]].squeeze())
            sequence_id = input_ids[i]
            sequence_output = []
            for node in sequence_id:
                node = node.item()
                if node in x_index:
                    sequence_output.append(x[x_index.index(node)])
                else:
                    sequence_output.append(torch.zeros_like(graph_output))
            
            sequence_output = torch.vstack(sequence_output)
            sequence_outputs.append(sequence_output)

        graph_outputs = torch.vstack(graph_outputs)
        sequence_outputs = torch.stack(sequence_outputs,dim=0)
            
        return sequence_outputs,graph_outputs


@registry.register_task_model('embed', 'gnn')
class ProteinGNNModel(ProteinGNNAbstractModel):

    def __init__(self, config: ProteinGNNConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size,config.input_size)
        if config.network_type == "GCN":
            self.gnn_model = ProteinGCNModel(config)
        elif config.network_type == "GraphSAGE":
            self.gnn_model = ProteinGraphSAGEModel(config)
        elif config.network_type == "GAT":
            self.gnn_model = ProteinGATModel(config)
        
        self.output = ProteinGNNOutput(config)
        self.init_weights()

    def forward(self, input_ids, x, edge_index, edge_attr, ptr):
        """ Runs the forward model pass

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [batch_size x sequence_length]
            x (Tensor[float]):
                The node feature of Graph, with the shape of [node_num x node_feature(1)]
            edge_index (Tensor[float]):
                The edge index of Graph, with the shape of [2 x edge_num]
            edge_attr (Tensor[float]):
                The edge attribute(weight) of Graph, 
                with the shape of [edge_num x edge_attribute(1)]
         

        Returns:
            sequence_embedding (Tensor[float]):  TOKEN_LEVEL
                Embedded sequence of shape [batch_size x protein_length x hidden_size]
            pooled_embedding (Tensor[float]):    SEQ_LEVEL
                Pooled representation of the entire sequence of size [batch_size x hidden_size]
        """

        node_index = x
        x = self.embedding(x).squeeze()

        x = self.gnn_model(x, edge_index, edge_attr) # [node_num x hidden_size]
        sequence_output,pooled_outputs = self.output(input_ids, node_index, x, ptr)
        
        outputs = (sequence_output, pooled_outputs)
        return outputs  # sequence_output, pooled_output


@registry.register_task_model('fluorescence', 'gnn')
@registry.register_task_model('stability', 'gnn')
class ProteinGNNForValuePrediction(ProteinGNNAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.gnn = ProteinGNNModel(config)
        self.predict = ValuePredictionHead(config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, graph_data, targets):
        # print(graph_data)
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        ptr = graph_data.ptr
        
        outputs = self.gnn(input_ids, x, edge_index, edge_attr, ptr)

        sequence_output, pooled_output = outputs
        outputs = self.predict(pooled_output, targets)
        # (loss), prediction_scores
        return outputs


@registry.register_task_model('remote_homology', 'gnn')
class ProteinGNNForSequenceClassification(ProteinGNNAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.gnn = ProteinGNNModel(config)
        self.classify = SequenceClassificationHead(
            config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, graph_data, targets):
        # print(graph_data)
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        ptr = graph_data.ptr
        
        outputs = self.gnn(input_ids, x, edge_index, edge_attr, ptr)

        sequence_output, pooled_output = outputs
        outputs = self.classify(pooled_output, targets)
        # (loss), prediction_scores
        return outputs


@registry.register_task_model('secondary_structure', 'gnn')
class ProteinGNNForSequenceToSequenceClassification(ProteinGNNAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.gnn = ProteinGNNModel(config)
        self.classify = SequenceToSequenceClassificationHead(
            config.hidden_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, graph_data, targets):

        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        ptr = graph_data.ptr
        
        outputs = self.gnn(input_ids, x, edge_index, edge_attr, ptr)

        sequence_output, pooled_output = outputs[:2]

        outputs = self.classify(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('contact_prediction', 'gnn')
class ProteinGNNForContactPrediction(ProteinGNNAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.gnn = ProteinGNNModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, graph_data, targets):

        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        ptr = graph_data.ptr
        
        outputs = self.gnn(input_ids, x, edge_index, edge_attr, ptr)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
