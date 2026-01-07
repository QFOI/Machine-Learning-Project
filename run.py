import argparse
import time
import torch
import json
from src.optimize import search_feasible, SearchConfig, TrainConfig

def main():
    # 1. 用 argparse 读参数
    parser = argparse.ArgumentParser(description="PyTorch Kissing Number Optimizer")
    parser.add_argument("-n", type=int, default=3, help="空间维度 (Dimension)")
    parser.add_argument("-m", type=int, default=12, help="点集数量 (Number of points)")
    parser.add_argument("--restarts", type=int, default=5, help="随机重启次数")
    parser.add_argument("--steps", type=int, default=5000, help="每次训练的最大步数")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--output", type=str, default="result.pt", help="结果保存路径")
    
    args = parser.parse_args()

    # 2. 构造配置对象
    train_cfg = TrainConfig(
        steps=args.steps,
        lr=args.lr,
        threshold=0.5,
        early_stop=True
    )
    
    search_cfg = SearchConfig(
        num_restarts=args.restarts,
        train_cfg=train_cfg,
        verbose=True  # 实时打印每轮重启的状态
    )

    print(f"🚀 开始搜索: n={args.n}, m={args.m} (最大重启次数: {args.restarts})")
    start_time = time.time()

    # 3. 调 search_feasible(n, m, ...)
    result = search_feasible(args.n, args.m, search_cfg)
    
    end_time = time.time()
    duration = end_time - start_time

    # 4. 打印结果 + 写 JSON (比 CSV 更适合保存详细报告)
    print("\n" + "="*40)
    if result.success:
        print(f"✅ 找到可行解! (耗时: {duration:.2f}秒)")
        print(f"成功种子: {result.success_seed}")
    else:
        print(f"❌ 未能找到完全可行解，已输出当前最优配置。")
    
    print(f"最终最大内积: {result.report.max_inner:.6f}")
    print(f"违规点对数: {result.report.num_violations}")
    print("="*40)

    # 5. 成功则保存 U.pt
    # 无论是否完全成功，我们都保存当前最好的构型以便观察
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
    torch.save(save_data, args.output)
    print(f"💾 结果已保存至: {args.output}")

if __name__ == "__main__":
    main()