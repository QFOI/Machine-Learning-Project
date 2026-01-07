import torch
import matplotlib.pyplot as plt

def plot_kissing_spheres(file_path):
    data = torch.load(file_path)
    U = data['U'].cpu().numpy() # 转为 numpy 方便绘图
    
    if data['n'] != 3:
        print("可视化目前仅支持 3D 结果。")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画出中心球（半径为 0.5 的参考球）
    u, v = torch.linspace(0, 2 * torch.pi, 20), torch.linspace(0, torch.pi, 20)
    x = 0.5 * torch.outer(torch.cos(u), torch.sin(v))
    y = 0.5 * torch.outer(torch.sin(u), torch.sin(v))
    z = 0.5 * torch.outer(torch.ones_like(u), torch.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

    # 画出计算出的 12 个点（在单位球面上，即距离中心 1.0 的位置）
    ax.scatter(U[:, 0], U[:, 1], U[:, 2], s=100, c='red', label='Points')

    # 连接原点到这些点（显示放射状分布）
    for point in U:
        ax.plot([0, point[0]], [0, point[1]], [0, point[2]], 'k--', alpha=0.3)

    ax.set_title(f"Kissing Number Visualization (n=3, m={data['m']})")
    plt.show()

if __name__ == "__main__":
    plot_kissing_spheres("result.pt")