# src/optimize.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import torch

from .evaluator import normalize_rows, feasibility_report, FeasibilityReport


# -----------------------------
# Config / Result dataclasses
# -----------------------------

@dataclass
class TrainConfig:
    # 约束阈值与容差
    threshold: float = 0.5
    eps: float = 1e-6

    # 优化超参
    steps: int = 10000  # Increased for better convergence
    lr: float = 5e-3  # Slightly reduced for stability
    weight_decay: float = 0.0

    # 数值稳定
    grad_clip: Optional[float] = 5.0  # Increased clip value
    proj_each_step: bool = True       # 每步投影回单位球面

    # 早停策略
    early_stop: bool = True
    check_every: int = 100            # Less frequent checks to save time

    # loss 形状
    use_repulsion: bool = True        # Enabled to help push points apart
    repulsion_alpha: float = 20.0     # Increased for stronger repulsion near threshold
    repulsion_lambda: float = 1e-2    # Increased lambda for more weight

    # 平滑最大违规（聚焦最糟糕点对）
    use_smooth_max: bool = True
    smooth_max_alpha: float = 100.0   # Increased for closer approximation to max
    smooth_max_weight: float = 2.0    # Increased weight to focus on worst violations

    # 学习率调度
    use_scheduler: bool = True
    scheduler_eta_min_ratio: float = 0.05  # Lower min ratio for more annealing

    # -----------------------------
    # NEW: 贪心初始化（泛化到任意维度）
    # -----------------------------
    init_method: str = "greedy"  # "random" | "greedy"
    greedy_candidates: int = 4096  # Increased for better initial points
    greedy_first_candidates: int = 16384  # Increased for first steps
    greedy_first_steps: int = 12          # Increased steps with more candidates


@dataclass
class TrainResult:
    success: bool
    U_best: torch.Tensor
    best_report: FeasibilityReport
    history: Dict[str, List[float]]


# -----------------------------
# Helpers
# -----------------------------

def _upper_triangle_values(G: torch.Tensor) -> torch.Tensor:
    """
    取 G 的 i<j 上三角（不含对角线）元素，返回 (num_pairs,) 向量。
    """
    m = G.shape[0]
    iu = torch.triu_indices(m, m, offset=1, device=G.device)
    return G[iu[0], iu[1]]


@torch.no_grad()
def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_total(U: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """
    总损失：违规损失 + （可选）平滑最大 +（可选）分散项
    为优化效率，一次计算 U_n 和 G，然后复用。
    """
    U_n = normalize_rows(U)
    G = U_n @ U_n.T
    vals = _upper_triangle_values(G)
    vio = torch.clamp(vals - cfg.threshold, min=0.0)
    L = (vio ** 2).sum()

    if cfg.use_smooth_max:
        excess = vio  # 已 clamp
        if excess.numel() > 0:
            smax = torch.logsumexp(cfg.smooth_max_alpha * excess, dim=0) / cfg.smooth_max_alpha
            L += cfg.smooth_max_weight * smax

    if cfg.use_repulsion:
        rep = torch.exp(cfg.repulsion_alpha * (vals - cfg.threshold)).sum()  # Shifted to focus repulsion around threshold
        L += cfg.repulsion_lambda * rep

    return L


# -----------------------------
# Initialization
# -----------------------------

@torch.no_grad()
def init_U_random(
    m: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    _set_seed(seed)
    U = torch.randn(m, n, device=device, dtype=dtype)
    return normalize_rows(U)


@torch.no_grad()
def init_U_greedy_maximin(
    m: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    candidates: int = 4096,
    first_candidates: int = 16384,
    first_steps: int = 12,
) -> torch.Tensor:
    """
    贪心 maximin / farthest-point 初始化（泛化到任意 n, m）：

    逐点加入：每次从若干随机候选中选择一个，使其与当前已选点集的“最大内积”最小：
        choose x = argmin_x  max_{u in S} <x, u>

    经验上对 spherical code / packing 的 max-inner 约束极其有效。
    """
    _set_seed(seed)

    # 第一个点随便取
    u0 = normalize_rows(torch.randn(1, n, device=device, dtype=dtype))
    U_list = [u0]  # each is (1,n)

    for k in range(1, m):
        K = first_candidates if k < first_steps else candidates
        C = normalize_rows(torch.randn(K, n, device=device, dtype=dtype))  # (K,n)

        S = torch.cat(U_list, dim=0)  # (k,n)
        # inner: (K,k), max_inner: (K,)
        max_inner = (C @ S.T).max(dim=1).values

        best_idx = torch.argmin(max_inner)
        U_list.append(C[best_idx : best_idx + 1])

    return torch.cat(U_list, dim=0)


def init_U(
    m: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    cfg: Optional[TrainConfig] = None,
) -> torch.Tensor:
    """
    初始化 U (m,n)，并投影到单位球面。
    默认使用贪心 maximin 初始化（可通过 cfg.init_method 改回 random）。
    """
    method = "greedy" if cfg is None else cfg.init_method.lower().strip()

    if method == "random":
        return init_U_random(m=m, n=n, device=device, dtype=dtype, seed=seed)

    if method == "greedy":
        if cfg is None:
            # fallback default params
            return init_U_greedy_maximin(
                m=m, n=n, device=device, dtype=dtype, seed=seed,
                candidates=4096, first_candidates=16384, first_steps=12
            )
        return init_U_greedy_maximin(
            m=m,
            n=n,
            device=device,
            dtype=dtype,
            seed=seed,
            candidates=cfg.greedy_candidates,
            first_candidates=cfg.greedy_first_candidates,
            first_steps=cfg.greedy_first_steps,
        )

    raise ValueError(f"Unknown init_method={method!r}. Use 'random' or 'greedy'.")


# -----------------------------
# Training (Adam)
# -----------------------------

def train_once(
    n: int,
    m: int,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> TrainResult:
    """
    单次训练：初始化 + Adam 优化，尝试找到可行点集。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) init (NEW: greedy)
    U0 = init_U(m=m, n=n, device=device, dtype=torch.float64, seed=seed, cfg=cfg)
    U = U0.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([U], lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.use_scheduler:
        eta_min = cfg.lr * cfg.scheduler_eta_min_ratio
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=eta_min)

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
        if scheduler is not None:
            scheduler.step()

        if cfg.proj_each_step:
            with torch.no_grad():
                U.copy_(normalize_rows(U))

        if (t % cfg.check_every == 0) or (t == cfg.steps):
            rep = feasibility_report(U.detach(), threshold=cfg.threshold, eps=cfg.eps, normalize=True)

            history["loss"].append(float(L.detach().item()))
            history["max_inner"].append(rep.max_inner)
            history["violations"].append(float(rep.num_violations))

            if rep.max_inner < best_report.max_inner:
                best_report = rep
                U_best = U.detach().clone()

            if verbose:
                print(
                    f"[step {t:6d}] loss={history['loss'][-1]:.4e} "
                    f"max_inner={rep.max_inner:.6f} violations={rep.num_violations}"
                )

            if cfg.early_stop and rep.ok:
                return TrainResult(True, U.detach().clone(), rep, history)

    final_rep = feasibility_report(U_best, threshold=cfg.threshold, eps=cfg.eps, normalize=True)
    return TrainResult(final_rep.ok, U_best, final_rep, history)


# -----------------------------
# Search config / result
# -----------------------------

@dataclass
class SearchConfig:
    num_restarts: int = 50  # Increased for more chances to find feasible
    train_cfg: TrainConfig = field(default_factory=TrainConfig)
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64
    verbose: bool = False

    # 失败后的后续精炼设置
    post_refine_on_fail: bool = True
    post_refine_steps: int = 8000  # Increased refine steps
    post_refine_lr: Optional[float] = 1e-4  # Lower lr for fine-tuning

    # 针对“接近阈值但未可行”的多候选精炼
    refine_top_k: int = 10  # More candidates to refine
    refine_steps: int = 8000
    refine_lr: Optional[float] = 1e-4


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
    candidates: List[Tuple[float, torch.Tensor, FeasibilityReport, int]] = []  # (max_inner, U, report, seed)

    for k in range(cfg.num_restarts):
        seed = base_seed + k
        res = train_once(n=n, m=m, cfg=cfg.train_cfg, device=device, seed=seed, verbose=False)

        if (best_rep_overall is None) or (res.best_report.max_inner < best_rep_overall.max_inner):
            best_rep_overall = res.best_report
            best_U_overall = res.U_best

        candidates.append((res.best_report.max_inner, res.U_best, res.best_report, seed))

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

    # 若未找到可行解，按靠近阈值的若干候选做精炼
    assert best_rep_overall is not None and best_U_overall is not None
    if cfg.post_refine_on_fail and len(candidates) > 0:
        lr_ref = cfg.post_refine_lr if cfg.post_refine_lr is not None else max(1e-5, cfg.train_cfg.lr * 0.2)

        if cfg.verbose:
            print(f"[refine] start top-{min(len(candidates), max(1, cfg.refine_top_k))} candidates; lr_ref={lr_ref}")

        top_k = sorted(candidates, key=lambda x: x[0])[: max(1, cfg.refine_top_k)]
        refined_best_rep = best_rep_overall
        refined_best_U = best_U_overall

        for idx, (max_inner_cand, U_cand, rep_cand, seed_cand) in enumerate(top_k, start=1):
            if cfg.verbose:
                print(
                    f"[refine {idx}/{len(top_k)}] seed={seed_cand} "
                    f"start from max_inner={max_inner_cand:.6f}, violations={rep_cand.num_violations}"
                )
            refine_steps = cfg.refine_steps if cfg.refine_steps is not None else cfg.post_refine_steps
            refine_res = bump_and_refine(U_cand, cfg.train_cfg, steps=refine_steps, lr=lr_ref, verbose=cfg.verbose)
            refined_rep = refine_res.best_report

            if refined_rep.max_inner < refined_best_rep.max_inner:
                refined_best_rep = refined_rep
                refined_best_U = refine_res.U_best

            if refined_rep.ok:
                return SearchResult(
                    success=True,
                    n=n,
                    m=m,
                    U=refine_res.U_best,
                    report=refined_rep,
                    best_over_restarts=refined_rep,
                    success_seed=None,
                )

        best_rep_overall = refined_best_rep
        best_U_overall = refined_best_U

    return SearchResult(
        success=False,
        n=n,
        m=m,
        U=best_U_overall,
        report=best_rep_overall,
        best_over_restarts=best_rep_overall,
        success_seed=None,
    )


# -----------------------------
# Optional: mutate + refine
# -----------------------------

@torch.no_grad()
def mutate_U(
    U: torch.Tensor,
    p_replace: float = 0.1,  # Increased replacement probability
    noise_std: float = 0.02,  # Slightly increased noise
) -> torch.Tensor:
    """
    对构型做轻量“突变”：
    - 以概率 p_replace 随机替换一部分点
    - 对所有点加小高斯噪声
    - 再投影回单位球面
    """
    m, n = U.shape
    U_new = U.clone()

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
    steps: int = 8000,
    lr: float = 1e-4,
    verbose: bool = False,
) -> TrainResult:
    """
    从已有构型出发，再跑一段局部优化（常用于“最优但不可行”的构型继续挤）。
    """
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

            if verbose:
                print(
                    f"[refine step {t:6d}] loss={history['loss'][-1]:.4e} "
                    f"max_inner={rep.max_inner:.6f} violations={rep.num_violations}"
                )

            if rep.max_inner < best_report.max_inner:
                best_report = rep
                U_best = U.detach().clone()

            if cfg2.early_stop and rep.ok:
                return TrainResult(True, U.detach().clone(), rep, history)

    final_rep = feasibility_report(U_best, threshold=cfg2.threshold, eps=cfg2.eps, normalize=True)
    return TrainResult(final_rep.ok, U_best, final_rep, history)