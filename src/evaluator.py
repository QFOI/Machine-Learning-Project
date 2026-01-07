# src/evaluator.py
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class FeasibilityReport:
    """
    ok：是否可行（True/False）

    max_inner：所有点对的内积最大值（最糟糕那一对）

    num_violations：违规点对数量（内积 > threshold + eps 的点对数）

    max_norm_error：单位长度误差最大值（诊断用）
    """
    ok: bool
    max_inner: float
    num_violations: int
    max_norm_error: float


def normalize_rows(U: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    对矩阵 U 做“按行归一化”，使得每一行向量的 L2 范数都为 1。

    为什么要这样做：
      我们的构造点集定义在单位球面上（即 ||u_i|| = 1）。
      即使生成器输出的向量长度略有偏差，先做归一化也能让评测更稳定、更符合几何含义。

    eps 的作用：
      用于避免某一行全为 0 导致范数为 0，从而出现除以 0 的错误；
      会将范数的最小值截断到 eps。
    """
    norms = U.norm(dim=1, keepdim=True).clamp_min(eps)  # (m, 1) 每一行的范数（列向量）
    return U / norms  # 广播除法：第 i 行除以第 i 行的范数 → 每行变单位向量


@torch.no_grad()
def feasibility_report(
    U: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    normalize: bool = True,
) -> FeasibilityReport:
    """
    判断点集 U 是否满足 kissing number 的可行约束，并返回详细评测报告。

    输入
    ----
    U: torch.Tensor，形状 (m, n)
        每一行是一个候选向量 u_i（不要求已经归一化）。
    threshold:
        内积约束阈值。kissing number 的标准阈值为 1/2（即 <= 0.5）。
    eps:
        数值容忍度：把 <= threshold + eps 视为可接受（避免浮点误差误判）。
    normalize:
        若为 True，则在评测前先把每行向量归一化到单位长度（投影到单位球面）。

    返回
    ----
    FeasibilityReport:
        ok, max_inner, num_violations, max_norm_error
    """
    # 0) 形状检查：必须是二维矩阵 (m, n)
    if U.dim() != 2:
        raise ValueError(f"U 必须是形状为 (m, n) 的二维张量，但实际是 {tuple(U.shape)}")

    m, n = U.shape

    # 1) 空集：没有点对约束，按逻辑“真空可行”
    if m == 0:
        return FeasibilityReport(ok=True, max_inner=float("-inf"), num_violations=0, max_norm_error=0.0)

    # 2) （可选）投影到单位球面：u_i <- u_i / ||u_i||
    #    这对应“外球与中心球接触”的条件（点在单位球面上）
    if normalize:
        U_eval = normalize_rows(U)
    else:
        U_eval = U

    # 3) 范数误差（诊断项）：归一化后 ||u_i|| 应接近 1
    norms = U_eval.norm(dim=1)  # (m,) 每行一个范数
    max_norm_error = float((norms - 1.0).abs().max().item())

    # 4) Gram 矩阵：G_ij = <u_i, u_j>
    #    这是检查约束 <u_i, u_j> <= 1/2 的核心
    G = U_eval @ U_eval.T  # (m, m)

    # 5) 只有一个点：不存在点对 (i<j)，所以一定可行
    if m < 2:
        return FeasibilityReport(ok=True, max_inner=float("-inf"), num_violations=0, max_norm_error=max_norm_error)

    # 6) 只取 i<j 的点对（右上角上三角，不含对角线）
    #    因为 G 对称，取上三角即可避免重复；对角线 i==j 不属于点对约束
    iu = torch.triu_indices(m, m, offset=1, device=U.device)  # 形状 (2, num_pairs)
    pair_vals = G[iu[0], iu[1]]  # 形状 (num_pairs,)，包含所有 i<j 的内积值

    # 7) 核心指标：
    #    - max_inner：所有点对内积的最大值（最糟糕那对）
    #    - num_violations：超过 threshold+eps 的点对数量
    max_inner = float(pair_vals.max().item())
    num_violations = int((pair_vals > (threshold + eps)).sum().item())
    ok = (num_violations == 0)

    return FeasibilityReport(
        ok=ok,
        max_inner=max_inner,
        num_violations=num_violations,
        max_norm_error=max_norm_error,
    )


@torch.no_grad()
def is_feasible(
    U: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    normalize: bool = True,
) -> bool:
    """
    便捷接口：只返回 True/False（是否满足所有点对内积 <= threshold + eps）。
    """
    return feasibility_report(U, threshold=threshold, eps=eps, normalize=normalize).ok


@torch.no_grad()
def max_pairwise_inner(
    U: torch.Tensor,
    normalize: bool = True,
) -> float:
    """
    便捷接口：返回 max_{i<j} <u_i, u_j>（所有不同点对内积的最大值）。
    """
    rep = feasibility_report(U, threshold=0.5, eps=0.0, normalize=normalize)
    return rep.max_inner
