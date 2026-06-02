import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITIES = ("a", "t", "v")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PriMD(nn.Module):
    """Primitive Memory Distillation for incomplete multimodal emotion recognition.

    The public forward signature stays compatible with the previous trainer:
    it still receives masked utterance-level A/T/V features and returns the same
    six leading outputs. Extra PriMD losses are returned in the final dict.
    """

    def __init__(
        self,
        args,
        adim,
        tdim,
        vdim,
        D_e,
        n_classes,
        depth=4,
        num_heads=4,
        mlp_ratio=1,
        drop_rate=0,
        attn_drop_rate=0,
        no_cuda=False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.dims = {"a": adim, "t": tdim, "v": vdim}
        self.slices = {
            "a": (0, adim),
            "t": (adim, adim + tdim),
            "v": (adim + tdim, adim + tdim + vdim),
        }
        self.out_dropout = args.drop_rate
        hidden = max(D_e, int(D_e * mlp_ratio))

        self.lambda_align = getattr(args, "lambda_align", 0.1)
        self.lambda_indep = getattr(args, "lambda_indep", 0.01)
        self.lambda_vq = getattr(args, "lambda_vq", 0.1)
        self.vq_beta = getattr(args, "vq_beta", 0.2)
        self.kd_temp = getattr(args, "kd_temp", 4.0)
        self.gumbel_tau = getattr(args, "gumbel_tau", 0.5)
        self.primitive_capacity = getattr(args, "primitive_capacity", 64)
        self.k_max = min(getattr(args, "k_max", 16), self.primitive_capacity)
        self.retrieval_dim = getattr(args, "retrieval_dim", 64)

        self.teacher_encoders = nn.ModuleDict()
        self.teacher_shared = nn.ModuleDict()
        self.teacher_specific = nn.ModuleDict()
        self.teacher_heads = nn.ModuleDict()
        self.student_encoders = nn.ModuleDict()
        self.student_shared = nn.ModuleDict()
        self.student_specific = nn.ModuleDict()
        self.student_heads = nn.ModuleDict()

        for m in MODALITIES:
            self.teacher_encoders[m] = nn.Sequential(
                nn.Linear(self.dims[m], D_e),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
            )
            self.teacher_shared[m] = MLP(D_e, hidden, D_e, drop_rate)
            self.teacher_specific[m] = MLP(D_e, hidden, D_e, drop_rate)
            self.teacher_heads[m] = nn.Linear(2 * D_e, n_classes)

            self.student_encoders[m] = nn.Sequential(
                nn.Linear(self.dims[m], D_e),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
            )
            self.student_shared[m] = MLP(D_e, hidden, D_e, drop_rate)
            self.student_specific[m] = MLP(D_e, hidden, D_e, drop_rate)
            self.student_heads[m] = nn.Linear(2 * D_e, n_classes)

        cls_hidden = max(D_e, 2 * D_e)
        self.teacher_classifier = nn.Sequential(
            nn.Linear(4 * D_e, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(cls_hidden, n_classes),
        )
        self.student_classifier = nn.Sequential(
            nn.Linear(4 * D_e, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(cls_hidden, n_classes),
        )

        self.primitive_memory = nn.ParameterDict(
            {
                m: nn.Parameter(torch.randn(self.primitive_capacity, D_e) * 0.02)
                for m in MODALITIES
            }
        )
        self.retrieve_q = nn.ModuleDict(
            {m: nn.Linear(D_e, self.retrieval_dim) for m in MODALITIES}
        )
        self.retrieve_k = nn.ModuleDict(
            {m: nn.Linear(D_e, self.retrieval_dim) for m in MODALITIES}
        )
        self.retrieve_num = nn.ModuleDict(
            {m: MLP(D_e, hidden, self.k_max, drop_rate) for m in MODALITIES}
        )

        self.last_aux = {}
        self.student_initialized = False

    def start_student_stage(self):
        """Freeze the full-modality teacher/memory and seed the student."""
        if not self.student_initialized:
            for m in MODALITIES:
                self.student_encoders[m].load_state_dict(self.teacher_encoders[m].state_dict())
                self.student_shared[m].load_state_dict(self.teacher_shared[m].state_dict())
                self.student_specific[m].load_state_dict(self.teacher_specific[m].state_dict())
                self.student_heads[m].load_state_dict(self.teacher_heads[m].state_dict())
            self.student_classifier.load_state_dict(self.teacher_classifier.state_dict())
            self.student_initialized = True

        for module in [
            self.teacher_encoders,
            self.teacher_shared,
            self.teacher_specific,
            self.teacher_heads,
            self.teacher_classifier,
        ]:
            for p in module.parameters():
                p.requires_grad_(False)
        for p in self.primitive_memory.parameters():
            p.requires_grad_(False)

    def _split_inputs(self, inputfeats):
        return {
            m: inputfeats[:, :, beg:end].permute(1, 0, 2)
            for m, (beg, end) in self.slices.items()
        }

    def _prepare_mask(self, input_features_mask, umask, inputfeats):
        seq_len, batch = inputfeats.size(0), inputfeats.size(1)
        if input_features_mask is None:
            mask = inputfeats.new_ones(batch, seq_len, 3)
        else:
            mask = input_features_mask.permute(1, 0, 2).float()
        return mask * umask.unsqueeze(-1).float()

    def _encode(self, inputfeats, mask, prefix):
        inputs = self._split_inputs(inputfeats)
        encoders = getattr(self, f"{prefix}_encoders")
        shared_proj = getattr(self, f"{prefix}_shared")
        specific_proj = getattr(self, f"{prefix}_specific")
        shared, specific = {}, {}
        for idx, m in enumerate(MODALITIES):
            obs = mask[:, :, idx : idx + 1]
            h = encoders[m](inputs[m]) * obs
            shared[m] = shared_proj[m](h) * obs
            specific[m] = specific_proj[m](h) * obs
        return shared, specific

    def _fuse_shared(self, shared, mask):
        stacked = torch.stack([shared[m] for m in MODALITIES], dim=2)
        denom = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        return (stacked * mask.unsqueeze(-1)).sum(dim=2) / denom

    def _joint_logits(self, shared_repr, specific, classifier):
        features = torch.cat(
            [shared_repr, specific["a"], specific["t"], specific["v"]], dim=-1
        )
        return features, classifier(features)

    def _valid_flat(self, x, umask):
        valid = umask.reshape(-1).bool()
        flat = x.reshape(-1, x.size(-1))
        if valid.any():
            return flat[valid]
        return flat[:0]

    def _align_loss(self, shared, umask):
        loss = shared["a"].new_tensor(0.0)
        pairs = 0
        for i, m in enumerate(MODALITIES):
            for n in MODALITIES[i + 1 :]:
                xm = self._valid_flat(shared[m], umask)
                xn = self._valid_flat(shared[n], umask)
                if xm.numel() and xn.numel():
                    loss = loss + F.mse_loss(xm, xn)
                    pairs += 1
        return loss / max(pairs, 1)

    def _rbf_kernel(self, x):
        if x.size(0) > 256:
            x = x[:256]
        dist = torch.cdist(x, x, p=2).pow(2)
        sigma = torch.median(dist.detach())
        sigma = torch.clamp(sigma, min=1e-6)
        return torch.exp(-dist / (2.0 * sigma))

    def _hsic_loss(self, shared, specific, umask):
        loss = shared["a"].new_tensor(0.0)
        count = 0
        for m in MODALITIES:
            s = self._valid_flat(shared[m], umask)
            p = self._valid_flat(specific[m], umask)
            if s.size(0) < 2 or p.size(0) < 2:
                continue
            n = min(s.size(0), 256)
            s, p = s[:n], p[:n]
            ks = self._rbf_kernel(s)
            kp = self._rbf_kernel(p)
            center = torch.eye(n, device=s.device, dtype=s.dtype) - (1.0 / n)
            hsic = torch.trace(ks.mm(center).mm(kp).mm(center)) / ((n - 1) ** 2)
            loss = loss + hsic
            count += 1
        return loss / max(count, 1)

    def _vq_loss(self, specific, umask):
        total = specific["a"].new_tensor(0.0)
        for m in MODALITIES:
            p = self._valid_flat(specific[m], umask)
            if p.numel() == 0:
                continue
            memory = self.primitive_memory[m]
            dist = (
                p.pow(2).sum(dim=1, keepdim=True)
                - 2 * p.mm(memory.t())
                + memory.pow(2).sum(dim=1).unsqueeze(0)
            )
            idx = torch.argmin(dist, dim=1)
            z = F.embedding(idx, memory)
            total = total + F.mse_loss(z, p.detach()) + self.vq_beta * F.mse_loss(
                p, z.detach()
            )
        return total / len(MODALITIES)

    def _retrieve(self, query, modality, missing_mask):
        batch, seq_len, _ = query.shape
        flat_query = query.reshape(batch * seq_len, self.D_e)
        flat_missing = missing_mask.reshape(batch * seq_len).float()
        memory = self.primitive_memory[modality]
        q = self.retrieve_q[modality](flat_query)
        k = self.retrieve_k[modality](memory)
        scores = q.mm(k.t()) / math.sqrt(float(self.retrieval_dim))
        attn = F.softmax(scores, dim=-1)
        top_val, top_idx = torch.topk(attn, k=self.k_max, dim=-1)
        top_mem = memory[top_idx]
        weighted = top_val.unsqueeze(-1) * top_mem
        cumsum = weighted.cumsum(dim=1)
        denom = top_val.cumsum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        candidates = cumsum / denom
        logits = self.retrieve_num[modality](flat_query)
        if self.training:
            choose = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)
        else:
            choose = F.softmax(logits / max(self.gumbel_tau, 1e-6), dim=-1)
        comp = (choose.unsqueeze(-1) * candidates).sum(dim=1)
        ranks = torch.arange(1, self.k_max + 1, device=query.device, dtype=query.dtype)
        cost_table = torch.exp(ranks / float(self.k_max)) - 1.0
        cost = (choose * cost_table).sum(dim=-1) * flat_missing
        return comp.view(batch, seq_len, self.D_e), cost.view(batch, seq_len), top_val

    def _teacher_forward(self, inputfeats, mask, umask, compute_aux=True):
        shared, specific = self._encode(inputfeats, mask, "teacher")
        shared_mean = self._fuse_shared(shared, mask)
        hidden, out = self._joint_logits(shared_mean, specific, self.teacher_classifier)
        out_a = self.teacher_heads["a"](torch.cat([shared["a"], specific["a"]], dim=-1))
        out_t = self.teacher_heads["t"](torch.cat([shared["t"], specific["t"]], dim=-1))
        out_v = self.teacher_heads["v"](torch.cat([shared["v"], specific["v"]], dim=-1))
        if compute_aux:
            align = self._align_loss(shared, umask)
            indep = self._hsic_loss(shared, specific, umask)
            vq = self._vq_loss(specific, umask)
            aux = self.lambda_align * align + self.lambda_indep * indep + self.lambda_vq * vq
        else:
            align = inputfeats.new_tensor(0.0)
            indep = inputfeats.new_tensor(0.0)
            vq = inputfeats.new_tensor(0.0)
            aux = inputfeats.new_tensor(0.0)
        self.last_aux = {
            "teacher_loss": aux,
            "align_loss": align.detach(),
            "indep_loss": indep.detach(),
            "vq_loss": vq.detach(),
            "student_loss": inputfeats.new_tensor(0.0),
            "kd_loss": inputfeats.new_tensor(0.0),
            "cost_loss": inputfeats.new_tensor(0.0),
        }
        return hidden, out, out_a, out_t, out_v, np.array([])

    def _student_forward(self, inputfeats, mask, umask, teacher_inputfeats=None):
        shared, specific = self._encode(inputfeats, mask, "student")
        shared_obs = self._fuse_shared(shared, mask)
        final_specific = {}
        costs = []
        retrieval_weights = []
        for idx, m in enumerate(MODALITIES):
            obs = mask[:, :, idx : idx + 1]
            missing = (1.0 - obs) * umask.unsqueeze(-1).float()
            comp, cost, weights = self._retrieve(shared_obs, m, missing.squeeze(-1))
            final_specific[m] = obs * specific[m] + missing * comp
            costs.append(cost)
            retrieval_weights.append(weights.detach().cpu().numpy())

        hidden, out = self._joint_logits(shared_obs, final_specific, self.student_classifier)
        out_a = self.student_heads["a"](torch.cat([shared["a"], final_specific["a"]], dim=-1))
        out_t = self.student_heads["t"](torch.cat([shared["t"], final_specific["t"]], dim=-1))
        out_v = self.student_heads["v"](torch.cat([shared["v"], final_specific["v"]], dim=-1))

        cost_loss = torch.stack(costs, dim=0).sum(dim=0)
        valid = umask.float()
        cost_loss = (cost_loss * valid).sum() / valid.sum().clamp_min(1.0)
        kd_loss = inputfeats.new_tensor(0.0)
        if teacher_inputfeats is not None:
            full_mask = umask.unsqueeze(-1).repeat(1, 1, 3).float()
            with torch.no_grad():
                _, teacher_logits, _, _, _, _ = self._teacher_forward(
                    teacher_inputfeats, full_mask, umask, compute_aux=False
                )
            if self.n_classes > 1:
                temp = self.kd_temp
                kd = F.kl_div(
                    F.log_softmax(out / temp, dim=-1),
                    F.softmax(teacher_logits / temp, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
                kd_loss = (kd * valid).sum() / valid.sum().clamp_min(1.0) * (temp ** 2)
            else:
                kd = F.mse_loss(out.squeeze(-1), teacher_logits.squeeze(-1), reduction="none")
                kd_loss = (kd * valid).sum() / valid.sum().clamp_min(1.0)

        self.last_aux = {
            "teacher_loss": inputfeats.new_tensor(0.0),
            "align_loss": inputfeats.new_tensor(0.0),
            "indep_loss": inputfeats.new_tensor(0.0),
            "vq_loss": inputfeats.new_tensor(0.0),
            "student_loss": kd_loss + cost_loss,
            "kd_loss": kd_loss.detach(),
            "cost_loss": cost_loss.detach(),
        }
        return hidden, out, out_a, out_t, out_v, np.array(retrieval_weights, dtype=object)

    def forward(
        self,
        inputfeats,
        input_features_mask=None,
        umask=None,
        first_stage=False,
        teacher_inputfeats=None,
    ):
        if umask is None:
            umask = inputfeats.new_ones(inputfeats.size(1), inputfeats.size(0))
        mask = self._prepare_mask(input_features_mask, umask, inputfeats)
        if first_stage:
            return (*self._teacher_forward(inputfeats, mask, umask), self.last_aux)
        return (*self._student_forward(inputfeats, mask, umask, teacher_inputfeats), self.last_aux)


if __name__ == "__main__":
    class Args:
        device = torch.device("cpu")
        drop_rate = 0.3
        primitive_capacity = 64
        k_max = 16
        retrieval_dim = 64
        gumbel_tau = 0.5
        kd_temp = 4.0
        lambda_align = 0.1
        lambda_indep = 0.01
        lambda_vq = 0.1
        vq_beta = 0.2

    model = PriMD(Args(), 512, 1024, 1024, 128, 4)
    x = torch.randn(12, 3, 512 + 1024 + 1024)
    mask = torch.ones(12, 3, 3)
    umask = torch.ones(3, 12)
    print(model(x, mask, umask, first_stage=True)[1].shape)
