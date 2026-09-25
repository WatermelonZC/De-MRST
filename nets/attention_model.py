import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import math
from typing import NamedTuple

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.functions import sample_many_hc



def set_decode_type(model, decode_type, temp=None):
    # 如果模型是DataParallel或DistributedDataParallel类型，则将其转换为module类型
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    # 调用模型的set_decode_type函数，设置解码类型
    model.set_decode_type(decode_type, temp=temp)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 mom_vehicle_size=None,
                 sub_vehicle_size=None,
                 designated_driver=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_hca = True
        # HC补充
        self.sub_vehicle_size = sub_vehicle_size
        self.mom_vehicle_size = mom_vehicle_size

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        assert designated_driver is not None, "designated_driver must be set as parameter"
        self.designated_driver = designated_driver
        self.node_dim = 2  # x, y
        self.prompt_dim = 6
        if designated_driver:
            self.node_dim = 7  # pickup, delivery, completion time, handling time, picking time
            self.prompt_dim = 5
        step_context_dim = embedding_dim
        # Special embedding projection for depot node
        self.init_embed_depot = nn.Linear(self.node_dim, embedding_dim)
        self.project_sub_vehicle_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_mom_vehicle_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.node_step_project = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
        self.init_sor_embed = nn.Linear(self.node_dim, embedding_dim)
        self.init_tar_embed = nn.Linear(self.node_dim, embedding_dim)
        self.initial_position_embed = nn.Linear(2, embedding_dim)
        self.init_embeddings = None
        self.embeddings_with_depot = None
        self.start_maker = nn.Parameter(torch.Tensor(embedding_dim))
        self.start_maker.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        self.glimpse_norm = nn.LayerNorm(embedding_dim)
        self.sub_key_norm = nn.LayerNorm(embedding_dim)
        self.mom_key_norm = nn.LayerNorm(embedding_dim)
        self.node_query_norm = nn.LayerNorm(embedding_dim)
        self.init_embed = nn.Linear(self.node_dim, embedding_dim)
        self.register_buffer(
            'request_feature_scales',
            torch.tensor([100.0, 100.0, 100.0, 100.0, 2500.0, 100.0, 60.0]),
        )
        self.vehicle_state_att = nn.Linear(embedding_dim + 1, embedding_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=self.embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_sub_context = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_mom_context = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)

        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def _request_features(self, input):
        tau_p = input.get('tau_p')
        if tau_p is None:
            tau_p = input['params'][:, 2].view(-1, 1, 1).expand_as(input['tau_h'])

        features = torch.cat((
            input['sor_loc'], input['tar_loc'], input['designated_time'],
            input['tau_h'], tau_p
        ), dim=-1)
        scales = self.request_feature_scales.to(
            device=features.device, dtype=features.dtype
        )
        return features / scales

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        batch, graph_size, _ = input['designated_time'].size()
        embeddings, _, self.init_embeddings = self.embedder(self._init_embed(input))
        self.embeddings_with_depot = torch.cat((embeddings, self.init_embeddings), dim=1)
        if 'mbr_initial_positions' in input and 'dor_initial_positions' in input:
            initial_positions = torch.cat(
                (input['dor_initial_positions'], input['mbr_initial_positions']), dim=1
            ) / 100.0
            self.embeddings_with_depot = torch.cat(
                (self.embeddings_with_depot, self.initial_position_embed(initial_positions)),
                dim=1,
            )
        state = self.problem.make_state(input)
        _log_p, pi, state = self._inner(input, embeddings, state)
        # cost, mask = self.problem.get_costs(input, pi)
        cost, mask = state.get_costs(input, pi)

        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):
        batch_size, seq_len, _ = a.shape
        likelihoods = torch.zeros(batch_size, seq_len, device=a.device)

        for i, log_p_tensor in enumerate(_log_p):
            indices = a[:, :, i]
            if i == 2:
                indices = indices - self.sub_vehicle_size
            selected_log_p = torch.gather(log_p_tensor, 2, indices.unsqueeze(-1)).squeeze(-1)
            likelihoods += selected_log_p

        # return likelihoods.sum(1)
        return likelihoods.mean(1)

    def _init_embed(self, input):
        if self.is_hca:
            if self.designated_driver:
                x = self.init_embed(self._request_features(input))
                return torch.cat(
                    (x, self.start_maker[None, None, :].expand(x.size(0), 1, self.embedding_dim)), 1)
            else:
                sor_embeddings = self.init_sor_embed(input['sor_loc'])
                tar_embeddings = self.init_tar_embed(input['tar_loc'])

            return torch.cat(
                (sor_embeddings, tar_embeddings, self.start_maker[None, None, :].expand(input['sor_loc'].size(0), 1, self.embedding_dim)), 1)
        return self.init_embed(input)

    def _inner(self, input, embeddings, state):

        outputs = []
        task_output = []
        mom_vehicle_output = []
        sub_vehicle_output = []

        sequences = []

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        ids = torch.arange(batch_size)
        while not (self.shrink_size is None and state.all_finished()):
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16（如果未完成的样本过少（小于 16），则不进行 batch shrinking（因为批量归一化在 batch 过小时效果不佳））
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]
            current_node = state.get_current_node()
            task_log_p, mask = self._get_log_p_hc(fixed, state, current_node=current_node, normalize=True, select_state="task",
                                                  select_embedding=None)
            task_selected = self._select_node(task_log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            selected_embedding = embeddings[ids, task_selected]  # (batch_size, embedding_size)
            # sub_vehicle select
            sub_log_p, mask = self._get_log_p_hc(fixed=fixed, state=state, current_node=current_node, select_state="sub_vehicle",
                                                 select_embedding=selected_embedding.unsqueeze(1), embeddings=embeddings)
            sub_selected = self._select_node(sub_log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            sub_selected_task = state.vehicle_pos[ids, sub_selected, 0]  # (batch,)
            sub_selected_node_embedding = self.embeddings_with_depot[ids, sub_selected_task].unsqueeze(1)  # (batch, 1, embedding_size)
            sub_selected_time_state = current_node[1][ids, sub_selected].unsqueeze(-1)  # (batch,)
            selected_embedding = self.vehicle_state_att(
                torch.cat((sub_selected_time_state, sub_selected_node_embedding), dim=-1)
            )
            mom_log_p, mask = self._get_log_p_hc(fixed=fixed, state=state, current_node=current_node, select_state="mom_vehicle",
                                                 select_embedding=selected_embedding,
                                                 embeddings=embeddings)
            mom_selected = self._select_node(mom_log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            mom_selected = mom_selected + self.sub_vehicle_size
            state = state.update(task_selected, sub_selected, mom_selected)
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            task_output.append(task_log_p[:, 0, :])
            sub_vehicle_output.append(sub_log_p[:, 0, :])
            mom_vehicle_output.append(mom_log_p[:, 0, :])
            sequences.append(torch.stack([task_selected, sub_selected, mom_selected], dim=1))
            i += 1

        return ((torch.stack(task_output, 1), torch.stack(sub_vehicle_output, 1), torch.stack(mom_vehicle_output, 1)),
                torch.stack(sequences, 1), state)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        return sample_many_hc(
            lambda input: self._inner(*input),
            lambda input, pi: self.problem.get_costs(input[0], pi),
            (input, self.embedder(self._init_embed(input))),
            batch_rep, iter_rep, self,
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_hc(self, fixed, state, current_node, normalize=True, select_state="task", select_embedding=None, embeddings=None):
        if select_state == "task":
            # step_context = checkpoint(self._get_parallel_step_context, fixed.node_embeddings, current_node, use_reentrant=False)

            step_context = self._get_parallel_step_context(fixed.node_embeddings, current_node)
            query = self.node_query_norm(fixed.context_node_projected + \
                                         self.project_step_context(step_context))
            glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
            # Compute the mask
            mask = state.get_task_mask()
        elif select_state == "sub_vehicle":
            query = self.sub_key_norm(self.project_sub_vehicle_query(self.last_glimpse + select_embedding) + self.last_glimpse + select_embedding)
            glimpse_K, glimpse_V, logit_K = self.get_attention_vehicle_data(embeddings, current_node, select_state)
            # Compute the mask
            mask = state.get_sub_vehicle_mask()
        elif select_state == "mom_vehicle":
            query = self.mom_key_norm(self.project_mom_vehicle_query(self.last_glimpse + select_embedding) + self.last_glimpse + select_embedding)
            glimpse_K, glimpse_V, logit_K = self.get_attention_vehicle_data(embeddings, current_node, select_state)
            mask = state.get_mom_vehicle_mask()
        log_p, glimpse, attention = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        self.last_query = query
        self.last_glimpse = glimpse
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, current_node):
        current_node, vehicle_time_state, last_order = current_node
        batch_size, vehicle_size, task_size = current_node.size()
        current_node = current_node.long()
        if current_node.max() >= self.embeddings_with_depot.shape[1] or current_node.min() < 0:
            raise ValueError(
                f"gather 索引超出范围: min={current_node.min().item()}, max={current_node.max().item()}, "
                f"但 embeddings_with_depot.shape[1]={self.embeddings_with_depot.shape[1]}")
        current_node_embedding = (torch.gather(
            self.embeddings_with_depot,
            1,
            current_node.clone()
            .view(batch_size, vehicle_size, 1)
            .expand(batch_size, vehicle_size, embeddings.size(-1))
        ))
        last_order_embedding = (torch.gather(
            self.embeddings_with_depot,
            1,
            last_order.clone()
            .view(batch_size, 1, 1)
            .expand(batch_size, 1, embeddings.size(-1))
        ))
        current_node_embedding = self.vehicle_state_att(
            torch.cat((vehicle_time_state, current_node_embedding), dim=-1)
        )
        current_node_embedding = current_node_embedding.mean(dim=1, keepdim=True)
        return current_node_embedding + last_order_embedding

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        attention = torch.softmax(compatibility, dim=-1)
        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(attention, glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = self.glimpse_norm(glimpse + query.unsqueeze(1))
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2), attention

    def _get_attention_node_data(self, fixed):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def get_attention_vehicle_data(self, embeddings, current_node, type="sub_vehicle"):
        if type == "sub_vehicle":
            vehicle_state_loc, vehicle_time_state, last_order = current_node
            vehicle_state_loc = vehicle_state_loc[:, :self.sub_vehicle_size, :]
            vehicle_time_state = vehicle_time_state[:, :self.sub_vehicle_size, :]
            batch_size, vehicle_size, _ = vehicle_state_loc.size()
            project_context = self.project_sub_context
            # time_context = self.sub_time_embedding
        else:
            vehicle_state_loc, vehicle_time_state, last_order = current_node
            vehicle_state_loc = vehicle_state_loc[:, self.sub_vehicle_size:, :]
            vehicle_time_state = vehicle_time_state[:, self.sub_vehicle_size:, :]
            batch_size, vehicle_size, _ = vehicle_state_loc.size()
            project_context = self.project_mom_context
            # time_context = self.mom_time_embedding
        vehicle_task_embedding = torch.gather(
            self.embeddings_with_depot,
            1,
            vehicle_state_loc.contiguous()
            .view(batch_size, vehicle_size, 1)
            .expand(batch_size, vehicle_size, embeddings.size(-1))
        ).view(batch_size, vehicle_size, embeddings.size(-1))
        msg_embedding = self.vehicle_state_att(
            torch.cat((vehicle_time_state, vehicle_task_embedding), dim=-1)
        )
        glimpse_key_step, glimpse_val_step, logit_key_step = \
            project_context(msg_embedding[:, None, :, :]).chunk(3, dim=-1)
        return (
            self._make_heads(glimpse_key_step),
            self._make_heads(glimpse_val_step),
            logit_key_step,
        )

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
