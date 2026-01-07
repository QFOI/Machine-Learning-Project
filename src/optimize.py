# src/optimize.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch

from .evaluator import normalize_rows, feasibility_report, FeasibilityReport


@dataclass
class TrainConfig:
    # 约束阈值与容差
    threshold: float = 0.5
    eps: float = 1e-6

    # 优化超参
    steps: int = 5000
    lr: float = 1e-2
    weight_decay: float = 0.0

    # 数值稳定
    grad_clip: Optional[float] = 1.0  # None 表示不裁剪
    proj_each_step: bool = True       # 每步投影回单位球面

    # 早停策略
    early_stop: bool = True
    check_every: int = 50             # 每隔多少步做一次 feasibility 检查

    # loss 形状
    use_repulsion: bool = False       # 是否添加分散项（可选）
    repulsion_alpha: float = 10.0
    repulsion_lambda: float = 1e-3


@dataclass
class TrainResult:
    success: bool
    U_best: torch.Tensor
    best_report: FeasibilityReport
    history: Dict[str, List[float]]


def _upper_triangle_values(G: torch.Tensor) -> torch.Tensor:
    """
    取 G 的 i<j 上三角（不含对角线）元素，返回 (num_pairs,) 向量。
    """
    m = G.shape[0]
    iu = torch.triu_indices(m, m, offset=1, device=G.device)
    return G[iu[0], iu[1]]


def loss_vio(
    U: torch.Tensor,
    threshold: float = 0.5,
    normalize: bool = True,
) -> torch.Tensor:
    """
    软约束损失：sum_{i<j} [max(0, <u_i,u_j> - threshold)]^2

    说明：
    - 这是可微的，适合 Adam/SGD。
    - normalize=True 时先按行归一化，避免长度漂移破坏几何含义。
    """
    if normalize:
        U_n = normalize_rows(U)
    else:
        U_n = U

    G = U_n @ U_n.T  # (m, m)
    vals = _upper_triangle_values(G)  # (num_pairs,)
    vio = torch.clamp(vals - threshold, min=0.0)
    return (vio * vio).sum()


def loss_total(
    U: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """
    总损失：违规损失 + （可选）分散项
    分散项用 exp(alpha * inner) 抑制过近点对（只是辅助，不是必须）。
    """
    L = loss_vio(U, threshold=cfg.threshold, normalize=True)

    if cfg.use_repulsion:
        U_n = normalize_rows(U)
        G = U_n @ U_n.T
        vals = _upper_triangle_values(G)
        rep = torch.exp(cfg.repulsion_alpha * vals).sum()
        L = L + cfg.repulsion_lambda * rep

    return L


def init_U(
    m: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    随机初始化 U (m,n)，并投影到单位球面。
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    U = torch.randn(m, n, device=device, dtype=dtype)
    U = normalize_rows(U)
    return U


def train_once(
    n: int,
    m: int,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> TrainResult:
    """
    单次训练：随机初始化 + Adam 优化，尝试找到可行点集。

    返回：
    - success：是否找到可行解
    - U_best：训练过程中最好的构型（按 max_inner 最小）
    - best_report：对应的评测
    - history：记录 loss / max_inner / violations 轨迹（用于画图写报告）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) init
    U = init_U(m=m, n=n, device=device, seed=seed)
    U = U.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([U], lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 2) bookkeeping
    best_report = feasibility_report(U.detach(), threshold=cfg.threshold, eps=cfg.eps, normalize=True)
    U_best = U.detach().clone()

    history: Dict[str, List[float]] = {"loss": [], "max_inner": [], "violations": []}

    # 3) loop
    for t in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        L = loss_total(U, cfg)
        L.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([U], max_norm=cfg.grad_clip)

        opt.step()

        # 投影回球面，保持 ||u_i||=1（非常重要）
        if cfg.proj_each_step:
            with torch.no_grad():
                U.copy_(normalize_rows(U))

        # 日志与可行性检查
        if (t % cfg.check_every == 0) or (t == cfg.steps):
            rep = feasibility_report(U.detach(), threshold=cfg.threshold, eps=cfg.eps, normalize=True)

            history["loss"].append(float(L.detach().item()))
            history["max_inner"].append(rep.max_inner)
            history["violations"].append(float(rep.num_violations))

            # 以 max_inner 作为“最好”的排序标准：越小越好
            if rep.max_inner < best_report.max_inner:
                best_report = rep
                U_best = U.detach().clone()

            if verbose:
                print(
                    f"[step {t:6d}] loss={history['loss'][-1]:.4e} "
                    f"max_inner={rep.max_inner:.6f} violations={rep.num_violations}"
                )

            if cfg.early_stop and rep.ok:
                return TrainResult(success=True, U_best=U.detach().clone(), best_report=rep, history=history)

    # 结束：返回最好的（可能不可行）
    final_rep = feasibility_report(U_best, threshold=cfg.threshold, eps=cfg.eps, normalize=True)
    return TrainResult(success=final_rep.ok, U_best=U_best, best_report=final_rep, history=history)


@dataclass
class SearchConfig:
    num_restarts: int = 200
    train_cfg: TrainConfig = TrainConfig()
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    verbose: bool = False


@dataclass
class SearchResult:
    success: bool
    n: int
    m: int
    U: torch.Tensor
    report: FeasibilityReport
    best_over_restarts: FeasibilityReport
    success_seed: Optional[int]


def search_feasible(
    n: int,
    m: int,
    cfg: SearchConfig,
    base_seed: int = 0,
) -> SearchResult:
    """
    多启动搜索：重复 train_once，找到一次可行解即可返回。
    同时记录所有重启中“max_inner 最小”的最好结果，便于调参/分析。
    """
    if cfg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = cfg.device

    best_rep_overall: Optional[FeasibilityReport] = None
    best_U_overall: Optional[torch.Tensor] = None

    for k in range(cfg.num_restarts):
        seed = base_seed + k
        res = train_once(n=n, m=m, cfg=cfg.train_cfg, device=device, seed=seed, verbose=False)

        if (best_rep_overall is None) or (res.best_report.max_inner < best_rep_overall.max_inner):
            best_rep_overall = res.best_report
            best_U_overall = res.U_best

        if cfg.verbose:
            print(
                f"[restart {k+1:4d}/{cfg.num_restarts}] "
                f"best max_inner={res.best_report.max_inner:.6f} "
                f"violations={res.best_report.num_violations} success={res.success}"
            )

        if res.success:
            return SearchResult(
                success=True,
                n=n,
                m=m,
                U=res.U_best,
                report=res.best_report,
                best_over_restarts=best_rep_overall,
                success_seed=seed,
            )

    # 没找到可行解，返回整体最优（通常是 max_inner 最小的那次）
    assert best_rep_overall is not None and best_U_overall is not None
    return SearchResult(
        success=False,
        n=n,
        m=m,
        U=best_U_overall,
        report=best_rep_overall,
        best_over_restarts=best_rep_overall,
        success_seed=None,
    )


# ------- 可选加分：突变 + 再优化（提高成功率） -------

@torch.no_grad()
def mutate_U(
    U: torch.Tensor,
    p_replace: float = 0.05,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    对构型做轻量“突变”：
    - 以概率 p_replace 随机替换一部分点
    - 对所有点加小高斯噪声
    - 再投影回单位球面
    """
    m, n = U.shape
    U_new = U.clone()

    # 替换部分点
    num_replace = max(1, int(p_replace * m)) if p_replace > 0 else 0
    if num_replace > 0:
        idx = torch.randperm(m, device=U.device)[:num_replace]
        U_new[idx] = torch.randn(num_replace, n, device=U.device, dtype=U.dtype)

    if noise_std > 0:
        U_new = U_new + noise_std * torch.randn_like(U_new)

    return normalize_rows(U_new)


def bump_and_refine(
    U_start: torch.Tensor,
    cfg: TrainConfig,
    steps: int = 2000,
    lr: float = 5e-3,
) -> TrainResult:
    """
    从已有构型出发，再跑一段局部优化（常用于“最优但不可行”的构型继续挤）。
    """
    device = U_start.device
    n = U_start.shape[1]
    m = U_start.shape[0]

    cfg2 = TrainConfig(**{**cfg.__dict__, "steps": steps, "lr": lr})

    U = normalize_rows(U_start).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([U], lr=cfg2.lr, weight_decay=cfg2.weight_decay)

    best_report = feasibility_report(U.detach(), threshold=cfg2.threshold, eps=cfg2.eps, normalize=True)
    U_best = U.detach().clone()

    history: Dict[str, List[float]] = {"loss": [], "max_inner": [], "violations": []}

    for t in range(1, cfg2.steps + 1):
        opt.zero_grad(set_to_none=True)
        L = loss_total(U, cfg2)
        L.backward()
        if cfg2.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([U], max_norm=cfg2.grad_clip)
        opt.step()
        if cfg2.proj_each_step:
            with torch.no_grad():
                U.copy_(normalize_rows(U))

        if (t % cfg2.check_every == 0) or (t == cfg2.steps):
            rep = feasibility_report(U.detach(), threshold=cfg2.threshold, eps=cfg2.eps, normalize=True)
            history["loss"].append(float(L.detach().item()))
            history["max_inner"].append(rep.max_inner)
            history["violations"].append(float(rep.num_violations))

            if rep.max_inner < best_report.max_inner:
                best_report = rep
                U_best = U.detach().clone()

            if cfg2.early_stop and rep.ok:
                return TrainResult(True, U.detach().clone(), rep, history)

    final_rep = feasibility_report(U_best, threshold=cfg2.threshold, eps=cfg2.eps, normalize=True)
    return TrainResult(final_rep.ok, U_best, final_rep, history)
