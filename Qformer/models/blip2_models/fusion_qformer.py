"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from Qformer.models.base_model import all_gather_with_grad, concat_all_gather
from Qformer.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
)

from torch.nn import functional as F


class EntityQformer(Blip2Base):

    def __init__(
            self,
            entity_size=1024,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=1024,
            max_txt_len=256,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, entity_size, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.entity_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def forward(self, entity, context):
        text = context
        entity_embeds = entity

        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(
            entity.device
        )

        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            use_cache=True,
            return_dict=True,
        )

        entity_feats = F.normalize(
            self.entity_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(entity.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )


        ##============== entity-text Contrastive ===================###
        entity_feats_all = concat_all_gather(
            entity_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            entity_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # entity-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), entity_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-entity similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = entity.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            entity.device
        )

        loss_etc = (
                           F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                           + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                   ) / 2

        ###============== Entity-text Matching ===================###
        # text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        # text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        # entity_embeds_world = all_gather_with_grad(entity_embeds)
        #
        # with torch.no_grad():
        #     sim_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
        #     sim_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
        #
        #     weights_t2i = F.softmax(sim_t2i, dim=1)
        #     weights_i2t = F.softmax(sim_i2t, dim=1)
        #
        #     # 修正 weights_t2i 和 weights_i2t 的 inf 和 nan
        #     weights_t2i = torch.where(torch.isnan(weights_t2i) | torch.isinf(weights_t2i),
        #                               torch.tensor(0.0, device=weights_t2i.device), weights_t2i)
        #     weights_i2t = torch.where(torch.isnan(weights_i2t) | torch.isinf(weights_i2t),
        #                               torch.tensor(0.0, device=weights_i2t.device), weights_i2t)
        #
        #     # 将负值裁剪为 0
        #     weights_t2i = torch.clamp(weights_t2i, min=0.0)
        #     weights_i2t = torch.clamp(weights_i2t, min=0.0)
        #
        #     # 确保概率分布的和为正值
        #     if (weights_t2i.sum(dim=1) == 0).any():
        #         weights_t2i += 1e-10
        #     if (weights_i2t.sum(dim=1) == 0).any():
        #         weights_i2t += 1e-10
        #
        # # select a negative entity for each text
        # entity_embeds_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     entity_embeds_neg.append(entity_embeds_world[neg_idx])
        # entity_embeds_neg = torch.stack(entity_embeds_neg, dim=0)
        #
        # # select a negative text for each entity
        # text_ids_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_ids_neg.append(text_input_ids_world[neg_idx])
        #     text_atts_neg.append(text_attention_mask_world[neg_idx])
        #
        # text_ids_neg = torch.stack(text_ids_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)
        #
        # text_ids_all = torch.cat(
        #     [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        # )  # pos, pos, neg
        # text_atts_all = torch.cat(
        #     [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
        #     dim=0,
        # )
        #
        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     entity_embeds.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)
        #
        # entity_embeds_all = torch.cat(
        #     [entity_embeds, entity_embeds_neg, entity_embeds], dim=0
        # )  # pos, neg, pos
        # entity_atts_all = torch.ones(entity_embeds_all.size()[:-1], dtype=torch.long).to(
        #     entity_embeds.device
        # )
        #
        # output_itm = self.Qformer.bert(
        #     text_ids_all,
        #     query_embeds=query_tokens_itm,
        #     attention_mask=attention_mask_all,
        #     encoder_hidden_states=entity_embeds_all,
        #     encoder_attention_mask=entity_atts_all,
        #     return_dict=True,
        # )
        #
        # vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # vl_output = self.itm_head(vl_embeddings)
        # logits = vl_output.mean(dim=1)
        #
        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(entity_embeds.device)
        # loss_etm = F.cross_entropy(logits, itm_labels)
        #
        # loss = loss_etm + loss_etc

        return entity_feats, text_feat, loss_etc

    def forward_entity(self, entity):
        entity_embeds = entity.unsqueeze(1)
        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(
            entity.device
        )

        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            return_dict=True,
        )

        entity_feats = self.entity_proj(query_output.last_hidden_state)

        cosine_sim = torch.matmul(entity_feats, entity.unsqueeze(-1)).squeeze(-1)  # [bs, 32]

        # 找到相似度最高的索引
        _, max_indices = torch.max(cosine_sim, dim=1)  # [bs]

        # 选择相似度最高的向量
        final_entity_feats = torch.stack([entity_feats[i, idx] for i, idx in enumerate(max_indices)],
                                         dim=0)  # [bs, 768]

        return final_entity_feats

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def user_embed_generate(self, entity_embeds, context):
        entity_embeds = entity_embeds
        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(
            entity_embeds.device
        )

        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            return_dict=True,
        )

        entity_feats = self.entity_proj(query_output.last_hidden_state) # [bs, 32, 768]

        text_tokens = self.tokenizer(
            context,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(entity_embeds.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :]) # [bs, 768]

        return entity_feats, text_feat

    def fuse_embeds(self, entity, text):
        entity_embeds = entity
        bs = entity_embeds.size(0)

        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(
            entity.device
        )

        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            use_cache=True,
            return_dict=True,
        )

        entity_feats = self.entity_proj(query_output.last_hidden_state)


        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(entity.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :])

        text_feat = text_feat.repeat(bs, 1)

        cosine_sim = torch.matmul(entity_feats, text_feat.unsqueeze(-1)).squeeze(-1)  # [bs, 32]

        # 找到相似度最高的索引
        _, max_indices = torch.max(cosine_sim, dim=1)  # [bs]

        # 选择相似度最高的向量
        final_entity_feats = torch.stack([entity_feats[i, idx] for i, idx in enumerate(max_indices)],
                                         dim=0)  # [bs, 768]

        return final_entity_feats