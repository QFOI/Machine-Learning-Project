import argparse
import time
import os
import json
import hashlib
import torch
from datetime import datetime
from src.optimize import search_feasible, SearchConfig, TrainConfig

def main():
    # 1. 用 argparse 读参数
    parser = argparse.ArgumentParser(description="PyTorch Kissing Number Optimizer")
    parser.add_argument("-n", type=int, default=3, help="空间维度 (Dimension)")
    parser.add_argument("-m", type=int, default=12, help="点集数量 (Number of points)")
    parser.add_argument("--restarts", type=int, default=5, help="随机重启次数")
    parser.add_argument("--steps", type=int, default=10000, help="每次训练的最大步数")
    parser.add_argument("--lr", type=float, default=0.00005, help="学习率")
    parser.add_argument("--output", type=str, default="result.pt", help="结果保存路径")
    # 可选优化策略
    parser.add_argument("--threshold", type=float, default=0.5, help="内积阈值 (默认 0.5)")
    parser.add_argument("--check-every", type=int, default=50, help="多少步检查一次可行性")
    parser.add_argument("--no-scheduler", action="store_true", help="关闭余弦调度")
    parser.add_argument("--no-smooth-max", action="store_true", help="关闭平滑最大违规损失")
    parser.add_argument("--smooth-max-alpha", type=float, default=50.0, help="平滑最大的 alpha")
    parser.add_argument("--smooth-max-weight", type=float, default=1.0, help="平滑最大损失的权重")
    parser.add_argument("--use-repulsion", action="store_true", help="启用分散项")
    parser.add_argument("--repulsion-alpha", type=float, default=10.0, help="分散项 alpha")
    parser.add_argument("--repulsion-lambda", type=float, default=1e-3, help="分散项权重")
    # 失败后精炼
    parser.add_argument("--no-post-refine", action="store_true", help="关闭失败后的后续精炼")
    parser.add_argument("--post-refine-steps", type=int, default=4000, help="后续精炼步数")
    parser.add_argument("--post-refine-lr", type=float, default=-1.0, help="后续精炼学习率，-1 表示使用 0.5*lr")
    parser.add_argument("--refine-top-k", type=int, default=5, help="失败后精炼时，取最接近阈值的前 K 个候选再微调")
    parser.add_argument("--refine-steps", type=int, default=4000, help="精炼阶段的步数（独立于 post-refine-steps）")
    parser.add_argument("--refine-lr", type=float, default=-1.0, help="精炼学习率，-1 表示使用 0.5*lr")
    
    args = parser.parse_args()

    # 2. 构造配置对象
    train_cfg = TrainConfig(
        steps=args.steps,
        lr=args.lr,
        threshold=args.threshold,
        early_stop=True,
        check_every=args.check_every,
        use_scheduler=(not args.no_scheduler),
        use_smooth_max=(not args.no_smooth_max),
        smooth_max_alpha=args.smooth_max_alpha,
        smooth_max_weight=args.smooth_max_weight,
        use_repulsion=args.use_repulsion,
        repulsion_alpha=args.repulsion_alpha,
        repulsion_lambda=args.repulsion_lambda,
    )
    
    search_cfg = SearchConfig(
        num_restarts=args.restarts,
        train_cfg=train_cfg,
        verbose=True,  # 实时打印每轮重启的状态
        post_refine_on_fail=(not args.no_post_refine),
        post_refine_steps=args.post_refine_steps,
        post_refine_lr=(None if args.post_refine_lr < 0 else args.post_refine_lr),
        refine_top_k=args.refine_top_k,
        refine_steps=args.refine_steps,
        refine_lr=(None if args.refine_lr < 0 else args.refine_lr),
    )

    print(f"开始搜索: n={args.n}, m={args.m} (最大重启次数: {args.restarts})")
    start_time = time.time()

    # 3. 调 search_feasible(n, m, ...)
    result = search_feasible(args.n, args.m, search_cfg)
    
    end_time = time.time()
    duration = end_time - start_time

    # 4. 打印结果 + 写 JSON (比 CSV 更适合保存详细报告)
    print("\n" + "="*40)
    if result.success:
        print(f"找到可行解 (耗时: {duration:.2f}秒)")
        print(f"成功种子: {result.success_seed}")
    else:
        print(f"未能找到完全可行解，已输出当前最优配置。")
    
    print(f"最终最大内积: {result.report.max_inner:.6f}")
    print(f"违规点对数: {result.report.num_violations}")
    print("="*40)

    # 5. 保存到 results/ 目录，使用唯一文件名（包含参数与时间 + 哈希）
    #    同时保存 .pt（张量）与 .json（报告与配置）
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 构建唯一 run_id：时间戳 + 配置哈希（稳定的短码）
    cfg_for_hash = {
        "n": args.n,
        "m": args.m,
        "restarts": args.restarts,
        "steps": args.steps,
        "lr": args.lr,
        "threshold": args.threshold,
        "check_every": args.check_every,
        "use_scheduler": (not args.no_scheduler),
        "use_smooth_max": (not args.no_smooth_max),
        "smooth_max_alpha": args.smooth_max_alpha,
        "smooth_max_weight": args.smooth_max_weight,
        "use_repulsion": args.use_repulsion,
        "repulsion_alpha": args.repulsion_alpha,
        "repulsion_lambda": args.repulsion_lambda,
        "post_refine_on_fail": (not args.no_post_refine),
        "post_refine_steps": args.post_refine_steps,
        "post_refine_lr": (None if args.post_refine_lr < 0 else args.post_refine_lr),
    }
    cfg_json = json.dumps(cfg_for_hash, sort_keys=True)
    short_hash = hashlib.md5(cfg_json.encode("utf-8")).hexdigest()[:8]
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    status_tag = "ok" if result.report.ok else "fail"
    base_name = f"n{args.n}_m{args.m}_r{args.restarts}_s{args.steps}_lr{args.lr}_{status_tag}_{time_tag}_{short_hash}"

    # 保存 .pt（含张量 U 与简要报告）
    pt_path = os.path.join(results_dir, base_name + ".pt")
    save_data = {
        "n": args.n,
        "m": args.m,
        "U": result.U,
        "report": {
            "ok": result.report.ok,
            "max_inner": result.report.max_inner,
            "num_violations": result.report.num_violations
        }
    }
    torch.save(save_data, pt_path)

    # 保存 .json（包含完整配置与时间信息、耗时与成功种子）
    json_path = os.path.join(results_dir, base_name + ".json")
    detail = {
        "config": cfg_for_hash,
        "duration_sec": duration,
        "success": result.success,
        "success_seed": result.success_seed,
        "best_over_restarts": {
            "max_inner": result.best_over_restarts.max_inner,
            "num_violations": result.best_over_restarts.num_violations,
            "ok": result.best_over_restarts.ok,
        },
        "saved_files": {
            "pt": pt_path,
            "json": json_path,
        },
        "timestamp": time_tag,
        "status": status_tag,
        "run_id": f"{time_tag}-{short_hash}",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    print(f"结果已保存至: {pt_path}\n详细报告: {json_path}")

if __name__ == "__main__":
    main()