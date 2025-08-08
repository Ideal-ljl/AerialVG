import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss

class CrossSelfRelationTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8,num_layers=3):
        super(CrossSelfRelationTransformer, self).__init__()

        # 自注意力降维
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model * 2, num_heads=num_heads, batch_first=True)
        self.linear_proj = nn.Linear(d_model * 2, d_model)  # 新增线性层降维
        self.self_attn_norm = nn.LayerNorm(d_model)

        # 交替堆叠 Cross-Attention 和 Self-Attention 层
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True),
                "cross_norm": nn.LayerNorm(d_model,eps=1e-6),
                "self_attn": nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True),
                "self_norm": nn.LayerNorm(d_model,eps=1e-6)
            })
            for _ in range(num_layers)
        ])

        # 映射到 logits 维度
        self.fc_logits = ContrastiveEmbed()
        for layer in self.layers:
            nn.init.xavier_uniform_(layer["cross_attn"].in_proj_weight)
            nn.init.xavier_uniform_(layer["self_attn"].in_proj_weight)
            nn.init.xavier_uniform_(layer["cross_attn"].out_proj.weight)
            nn.init.xavier_uniform_(layer["self_attn"].out_proj.weight)

    def forward(self, image_features, text_dict):
        text_features = text_dict['encoded_text']
        text_attention_mask = text_dict["text_token_mask"]
        # print('image_features',image_features)
        bs, num, d_model = image_features.shape
        
        # Step 1: 两两组合图像特征
        combined_features = torch.cat([
            image_features.unsqueeze(2).expand(-1, -1, num, -1),
            image_features.unsqueeze(1).expand(-1, num, -1, -1)
        ], dim=-1)  # [bs, num, num, d_model * 2]
        combined_features = combined_features.view(bs, num * num, d_model * 2)  # [bs, num^2, d_model * 2]

        # 使用自注意力直接降维
        relation_features, _ = self.self_attention(combined_features, combined_features, combined_features)
        relation_features = self.linear_proj(relation_features)  # 降维至 [bs, num^2, d_model]
        relation_features = self.self_attn_norm(relation_features)
        # print('relation_features',relation_features.shape,relation_features)
        # print('text_attention_mask',text_attention_mask.shape,text_attention_mask)

        # Step 2: 交替堆叠 Cross-Attention 和 Self-Attention
        for layer in self.layers:
            # Cross-Attention
            cross_attn_output, _ = layer["cross_attn"](
                query=relation_features,          # 使用图像关系特征作为 query
                key=text_features,                # 使用文本特征作为 key
                value=text_features,              # 使用文本特征作为 value
                key_padding_mask= ~text_attention_mask  # 使用文本的掩码
            )
            relation_features = layer["cross_norm"](cross_attn_output + relation_features)  # 残差连接 + LayerNorm
            relation_features = F.relu(relation_features)

            # Self-Attention
            self_attn_output, _ = layer["self_attn"](relation_features, relation_features, relation_features)
            relation_features = layer["self_norm"](self_attn_output + relation_features)  # 残差连接 + LayerNorm

        # Step 3: 映射到 logits 维度
        logits = self.fc_logits(relation_features,text_dict)
        # print('logits',logits.shape,logits)
        logits_matrix = logits.view(bs, num, num,d_model)


        # 置零对角线元素
        # mask = 1 - torch.eye(num, device=logits.device).unsqueeze(0).unsqueeze(-1)  # [1, num, num, 1]
        # logits_matrix = logits_matrix * mask
        # logits_matrix = logits_matrix * (1 - torch.eye(num, device=logits.device).unsqueeze(0))

        # 分别在两个 num 维度上取最大值，得到两个 [bs, num] 的分数
        max_score_dim1= logits_matrix.mean(dim=1)  # 沿第一个 num 维度取最大值
        max_score_dim2 = logits_matrix.mean(dim=2)  # 沿第二个 num 维度取最大值

        # 取平均，得到每个图像特征的最终分数
        final_scores = (max_score_dim1 + max_score_dim2) / 2  # [bs, num]
        return final_scores

# # 测试
# batch_size = 4
# num_relations = 15
# d_model = 256
# num_tokens = 20
# text_dim = 256
# logits_dim = 256

# # 假设输入
# relation_features = torch.rand(batch_size, num_relations, d_model)
# text_features = torch.rand(batch_size, num_tokens, text_dim)
# text_attention_mask = torch.ones(batch_size, num_tokens).bool()

# model = CrossSelfRelationTransformer(d_model=d_model, num_heads=8, num_relations=num_relations, text_dim=text_dim, logits_dim=logits_dim)
# logits = model(relation_features, text_features)

# print("Logits shape:", logits.shape)  # 输出形状应为 (batch_size, num_relations, logits_dim)

def build_relation_transformer(args):
    """
    构建 CrossSelfRelationTransformer 模型。

    参数:
    - args (Namespace): 包含所有模型超参数的命名空间对象。

    返回:
    - CrossSelfRelationTransformer 模型实例
    """
    return CrossSelfRelationTransformer(
        d_model=args.hidden_dim,
        num_heads=args.nheads,
        num_layers=args.relation_num_layers
    )