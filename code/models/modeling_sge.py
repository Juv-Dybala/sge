"""Example of how to add a model in tape.

This file shows an example of how to add a new model to the tape training
pipeline. tape models follow the huggingface API and so require:

    - A config class
    - An abstract model class
    - A model class to output sequence and pooled embeddings
    - Task-specific classes for each individual task

This will walkthrough how to create each of these, with a task-specific class for
secondary structure prediction. You can look at the other task-specific classes
defined in e.g. tape/models/modeling_bert.py for examples on how to
define these other task-specific models for e.g. contact prediction or fluorescence
prediction.

In addition to defining these models, this shows how to register the model to
tape so that you can use the same training machinery to run your tasks.
"""


import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import logging
import math
import typing

from tape import ProteinModel, ProteinConfig

from .modeling_utils import prune_linear_layer
from .modeling_utils import get_activation_fn
from .modeling_utils import LayerNorm
from .modeling_utils import SequenceToSequenceClassificationHead
from .modeling_utils import ValuePredictionHead
from tape.registry import registry


logger = logging.getLogger(__name__)

URL_PREFIX = "https://s3.amazonaws.com/songlabdata/proteindata/pytorch-models/"
SGE_PRETRAINED_CONFIG_ARCHIVE_MAP: typing.Dict[str, str] = {}
SGE_PRETRAINED_MODEL_ARCHIVE_MAP: typing.Dict[str, str] = {}


class ProteinSGEConfig(ProteinConfig):
    """ The config class for our new model. This should be a subclass of
        ProteinConfig. It's a very straightforward definition, which just
        accepts the arguments that you would like the model to take in
        and assigns them to the class.

        Note - if you do not initialize using a model config file, you
        must provide defaults for all arguments.
    """

    pretrained_config_archive_map = SGE_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 30,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 8096,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 max_sequence_length: int = 32, # include <cls> etc.
                 num_path: int = 3,
                 path_length: int = 5,
                 beta: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_sequence_length = max_sequence_length
        self.num_path = num_path
        self.path_length = path_length
        self.beta = beta


class ProteinSGEAbstractModel(ProteinModel):
    """ All your models will inherit from this one - it's used to define the
        config_class of the model set and also to define the base_model_prefix.
        This is used to allow easy loading/saving into different models.
    """
    config_class = ProteinSGEConfig
    base_model_prefix = 'sge'
    pretrained_model_archive_map = SGE_PRETRAINED_MODEL_ARCHIVE_MAP

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

# The batch_size in sub_module means sub_batch_size.

class ProteinSGEEmbeddingBias(nn.Module):
    # APE
    def __init__(self, config) -> None:
        super().__init__()
        self.node_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.linear = nn.Linear(
            config.hidden_size,config.hidden_size,bias=False)

    def forward(self, walk_paths):
        # RW\ARW
        # walk_paths: [batch_size x seq_length x num_path x path_length]
        walk_paths = self.node_embeddings(walk_paths) 
        # path_embedding = torch.mean(walk_paths,dim=(2,3)) # [batch_size x hidden_size]
        path_embedding = torch.sum(walk_paths,dim=(2,3))
        # path_embedding = self.linear(path_embedding)

        return path_embedding


class ProteinSGESelfAttentionBias(nn.Module):
    # RPE
    def __init__(self, config) -> None:
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.path_embedding = ProteinSGEEmbeddingBias(config)
        self.distance_embedding = nn.Embedding(
            config.max_position_embeddings, config.num_attention_heads,padding_idx=0)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, distance, random_walk=None, anonymous_random_walk=None):
        distance = torch.add(distance,1)
        distance_rpe = self.distance_embedding(distance).permute(0,3,1,2)
        attn_bias = distance_rpe
        
        if random_walk is not None:
            rw_embedding = self.path_embedding(random_walk)
            rw_embedding = self.transpose_for_scores(rw_embedding)
            rw_rpe = torch.matmul(rw_embedding, rw_embedding.transpose(-1, -2))            
            attn_bias = attn_bias + rw_rpe
        
        if anonymous_random_walk is not None:
            arw_embedding = self.path_embedding(anonymous_random_walk)
            arw_embedding = self.transpose_for_scores(arw_embedding)
            arw_rpe = torch.matmul(arw_embedding, arw_embedding.transpose(-1, -2))
            attn_bias = attn_bias + arw_rpe
        return attn_bias


class ProteinSGEEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.embedding_bias = ProteinSGEEmbeddingBias(config)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, 
                random_walk=None, anonymous_random_walk=None):
        seq_length = input_ids.size(-1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Random Walk
        rw_embedding = self.embedding_bias(random_walk)
        # Anonymous Random Walk
        arw_embedding = self.embedding_bias(anonymous_random_walk)
        embedding_bias = rw_embedding + arw_embedding

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = embeddings + embedding_bias
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ProteinSGESelfAttention(nn.Module):
    #TODO: 加RPE
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn_bias = ProteinSGESelfAttentionBias(config)
        self.beta = config.beta

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, distance,
                random_walk=None, anonymous_random_walk=None):
        # hidden_states: [batch_size x seq_length x hidden_size]
        # distance: [batch_size x seq_length x seq_length]
        # walk_paths: [batch_size x seq_length x num_path x path_length]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # qkv_layer: [batch_size x num_head x seq_length x head_size]
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores: [batch_size x num_head x seq_length x seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attn_bias = self.attn_bias(distance,random_walk,anonymous_random_walk)
        attention_scores = attention_scores + attn_bias * self.beta
        
        # Apply the attention mask is (precomputed for all layers in
        # ProteinBertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original ProteinBert paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) \
            if self.output_attentions else (context_layer,)
        return outputs


class ProteinSGESelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ProteinSGEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ProteinSGESelfAttention(config)
        self.output = ProteinSGESelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask,
                random_walk, anonymous_random_walk, distance):
        self_outputs = self.self(input_tensor, attention_mask, distance,
                                 random_walk, anonymous_random_walk)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ProteinSGEIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation_fn(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ProteinSGEOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ProteinSGELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ProteinSGEAttention(config)
        self.intermediate = ProteinSGEIntermediate(config)
        self.output = ProteinSGEOutput(config)

    def forward(self, hidden_states, attention_mask,
                random_walk, anonymous_random_walk, distance):
        attention_outputs = self.attention(hidden_states, attention_mask,
                                           random_walk, anonymous_random_walk, distance)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class ProteinSGEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [ProteinSGELayer(config) for _ in range(config.num_hidden_layers)])

    def run_function(self, start, chunk_size):
        def custom_forward(hidden_states, attention_mask,
                           random_walk, anonymous_random_walk, distance):
            all_hidden_states = ()
            all_attentions = ()
            chunk_slice = slice(start, start + chunk_size)
            for layer in self.layer[chunk_slice]:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer(hidden_states, attention_mask,
                                      random_walk, anonymous_random_walk, distance)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = (hidden_states,)
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs

        return custom_forward

    def forward(self, hidden_states, attention_mask, chunks=None,
                random_walk=None, anonymous_random_walk=None, distance=None):
        all_hidden_states = ()
        all_attentions = ()

        if chunks is not None:
            assert isinstance(chunks, int)
            chunk_size = (len(self.layer) + chunks - 1) // chunks
            for start in range(0, len(self.layer), chunk_size):
                outputs = checkpoint(self.run_function(start, chunk_size),
                                     hidden_states, attention_mask, 
                                     random_walk, anonymous_random_walk, distance)
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + outputs[1]
                if self.output_attentions:
                    all_attentions = all_attentions + outputs[-1]
                hidden_states = outputs[0]
        else:
            for i, layer_module in enumerate(self.layer):
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, attention_mask,
                                             random_walk, anonymous_random_walk, distance)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            # Add last layer
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = (hidden_states,)
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class ProteinSGEPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinSGEAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gru = nn.GRU(config.hidden_size,config.hidden_size)
    
    def forward(self, sequence_output, pooled_output, session_index):
        session_index = session_index.tolist()
        
        # aggregate sequence_output (token level)
        # TODO: 重组起止符，加入相同token的聚合，目前仅简单concatenate
        sequence_output = list(torch.split(sequence_output,session_index,dim=0))
        for i in range(len(sequence_output)):         
            sequence_output[i] = torch.cat(tuple(sequence_output[i]),dim=0)
        sequence_output = tuple(sequence_output)

        # aggregate pooled_output (sequence level)
        # 使用GRU聚合
        pooled_output = list(torch.split(pooled_output,session_index,dim=0))
        for i in range(len(pooled_output)):
            _,pooled_output[i] = self.gru(pooled_output[i])
        pooled_output = torch.cat(pooled_output,dim=0)

        return sequence_output,pooled_output


@registry.register_task_model('embed', 'sge')
class ProteinSGEModel(ProteinSGEAbstractModel):
    """ The base model class. This will return embeddings of the input amino
        acid sequence. It is not used for any specific task - you'll have to
        define task-specific models further on. Note that there is a little
        more machinery in the models we define, but this is a stripped down
        version that should give you what you need
    """
    # init expects only a single argument - the config
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ProteinSGEEmbeddings(config)
        self.encoder = ProteinSGEEncoder(config)
        self.pooler = ProteinSGEPooler(config)
        self.aggregator = ProteinSGEAggregator(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class ProteinModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, input_mask=None,session_index=None,
                random_walk=None, anonymous_random_walk=None, distance=None):
        """ Runs the forward model pass

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [sub_batch_size x sequence_length]
            input_mask (Tensor[bool]):
                Tensor of booleans w/ same shape as input_ids, indicating whether
                a given sequence position is valid
            session_index (Tensor[int]):
                Tensor of the sub_sequence of each bacth,
                with the shape of [batch_size] 
                
            random_walk/anonymous_random_walk (Tensor[Long]):
                Tensor of random walk path, 
                with the shape of [sub_batch_size x sequence_length x path_num x path_length]
            ditance (Tensor[Long]):
                Matrix of the shortest distance between nodes,
                with the shape of [sub_batch_size x sequence_length x sequence_length]

        Returns:
            sequence_embedding (Tensor[float]):  TOKEN_LEVEL
                Embedded sequence of shape [batch_size x protein_length x hidden_size]
            pooled_embedding (Tensor[float]):    SEQ_LEVEL
                Pooled representation of the entire sequence of size [batch_size x hidden_size]
        """

        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        # Since input_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids,
                                           random_walk=random_walk,
                                           anonymous_random_walk=anonymous_random_walk)
        
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       chunks=None,
                                       random_walk=random_walk,
                                       anonymous_random_walk = anonymous_random_walk,
                                       distance=distance)
        
        sequence_output = encoder_outputs[0] # [sub_batch_size x seq_len x hidden_size]
        pooled_output = self.pooler(sequence_output) # [sub_batch_size x hidden_size]
        
        # 聚合子序列
        sequence_output,pooled_output = self.aggregator(sequence_output,pooled_output,session_index)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs


# This registers the model to a specific task, allowing you to use all of TAPE's
# machinery to train it.
@registry.register_task_model('secondary_structure', 'sge')
class ProteinSGEForSequenceToSequenceClassification(ProteinSGEAbstractModel):

    def __init__(self, config: ProteinSGEConfig):
        super().__init__(config)
        # the name of this variable *must* match the base_model_prefix
        self.sge = ProteinSGEModel(config)
        # The seq2seq classification head. First argument must match the
        # output embedding size of the SimpleConvModel. The second argument
        # is present in every config (it's an argument of ProteinConfig)
        # and is used for classification tasks.
        self.classify = SequenceToSequenceClassificationHead(
            config.hidden_size, config.num_labels)

    def forward(self, input_ids, input_mask=None, targets=None):
        """ Runs the forward model pass and may compute the loss if targets
            is present. Note that this does expect the third argument to be named
            `targets`. You can look at the different defined models to see
            what different tasks expect the label name to be.

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [batch_size x protein_length]
            input_mask (Tensor[bool]):
                Tensor of booleans w/ same shape as input_ids, indicating whether
                a given sequence position is valid
            targets (Tensor[long], optional):
                Tensor of output target labels of shape [batch_size x protein_length]
        """
        outputs = self.sge(input_ids, input_mask)
        sequence_embedding = outputs[0]
        
        print("breakpoint 1")
        exit()
        prediction = self.classify(sequence_embedding)

        outputs = (prediction,)

        return outputs  # ((loss, metrics)), prediction


@registry.register_task_model('fluorescence', 'sge')
@registry.register_task_model('stability', 'sge')
class ProteinSGEForValuePrediction(ProteinSGEAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.sge = ProteinSGEModel(config)
        self.predict = ValuePredictionHead(config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None, session_index=None,
                random_walk=None, anonymous_random_walk=None, distance=None):

        # targets、session_index: [batch_size]
        # random_walk、anonymous_random_walk: [sub_batch_size x sequence_length x path_num x path_length]
        # distance: [sub_batch_size x sequence_length x sequence_length]

        outputs = self.sge(input_ids, input_mask, session_index, 
                           random_walk, anonymous_random_walk,distance)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)

        return outputs