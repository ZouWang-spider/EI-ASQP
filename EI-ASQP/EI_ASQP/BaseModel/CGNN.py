import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalModule(nn.Module):
    def __init__(self, num_parents, embed_dim, hidden_dim, output_dim):
        super(CausalModule, self).__init__()
        self.num_parents = num_parents
        self.hidden_dim = hidden_dim

        self.w_parents = nn.ModuleList([
            nn.Linear(embed_dim, hidden_dim, bias=False)
            for _ in range(num_parents)
        ])
        self.w_input = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.bias_k = nn.Parameter(torch.zeros(hidden_dim))

        self.w_out = nn.Linear(hidden_dim, output_dim)
        self.bias_out = nn.Parameter(torch.zeros(output_dim))

    def _match_length(self, src, target_len):
        if src.size(0) == target_len:
            return src
        elif src.size(0) == 1:
            return src.expand(target_len, -1)
        else:
            repeats = (target_len + src.size(0) - 1) // src.size(0)
            return src.repeat(repeats, 1)[:target_len, :]

    def forward(self, *inputs):
        """
        inputs: (*X_parents, E_i)
        每个 input shape: (seq_len, embed_dim)
        """
        *parents, E_i = inputs
        base_len = E_i.size(0)

        # 调整所有输入长度一致
        parents = [self._match_length(p, base_len) for p in parents]
        E_i = self._match_length(E_i, base_len)

        # ∑ w_{jk}^i Xj
        parent_sum = torch.zeros(base_len, self.hidden_dim, device=E_i.device)
        for j, Xj in enumerate(parents):
            parent_sum += self.w_parents[j](Xj)

        Ei_term = self.w_input(E_i)
        h = F.relu(parent_sum + Ei_term + self.bias_k)
        output = self.w_out(h) + self.bias_out

        return output


class CausalGenerationModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(CausalGenerationModel, self).__init__()
        self.f1 = CausalModule(num_parents=0, embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=embed_dim)
        self.f2 = CausalModule(num_parents=1, embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=embed_dim)
        self.f3 = CausalModule(num_parents=2, embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=embed_dim)
        self.f4 = CausalModule(num_parents=2, embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=embed_dim)

    def forward(self, E1, E2, E3, E4):
        x1 = self.f1(E1)                # x1 = f1(E1)
        x2 = self.f2(x1, E2)            # x2 = f2(x1, E2)
        x3 = self.f3(x1, x2, E3)        # x3 = f3(x1, x2, E3)
        x4 = self.f4(x2, x3, E4)        # x4 = f4(x2, x3, E4)
        return x1, x2, x3, x4



##########################  Test  ##########################
# E1 = torch.randn(4, 768)
# E2 = torch.randn(3, 768)
# E3 = torch.randn(5, 768)
# E4 = torch.randn(1, 768)
#
# CGNN_model = CausalGenerationModel(embed_dim=768, hidden_dim=512)
# x1, x2, x3, x4 = CGNN_model(E1, E2, E3, E4)
#
# print(x1.shape)
# print(x2.shape)
# print(x3.shape)
# print(x4.shape)