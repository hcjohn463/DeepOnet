import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

class Elliptic2D:
    """二維橢圓PDE求解器示例
    
    形式: A∂²u/∂x² + B∂²u/∂x∂y + C∂²u/∂y² + D∂u/∂x + E∂u/∂y + Fu + G = 0
    
    為了簡化，我們考慮特殊情況：
    -∂²u/∂x² - ∂²u/∂y² = f(x,y)  (Poisson方程)
    即：A=1, B=0, C=1, D=E=F=0, G=f(x,y)
    """
    
    def __init__(self, nx, ny, Lx=1.0, Ly=1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def build_laplacian_2d(self):
        """構建二維拉普拉斯算子矩陣"""
        # 一維二階差分算子
        d2dx2 = diags([1, -2, 1], [-1, 0, 1], shape=(self.nx, self.nx)) / self.dx**2
        d2dy2 = diags([1, -2, 1], [-1, 0, 1], shape=(self.ny, self.ny)) / self.dy**2
        
        # 二維拉普拉斯算子 (Kronecker積)
        Ix = eye(self.nx)
        Iy = eye(self.ny)
        
        # ∇² = ∂²/∂x² + ∂²/∂y²
        laplacian_2d = kron(Iy, d2dx2) + kron(d2dy2, Ix)
        
        return laplacian_2d
    
    def solve_poisson_2d(self, f_values):
        """求解二維Poisson方程: -∇²u = f"""
        L = self.build_laplacian_2d()
        
        # 邊界條件: u = 0 在邊界上
        # 將二維問題轉換為一維向量問題
        f_vector = f_values.flatten()
        
        # 應用邊界條件
        for i in range(self.nx):
            for j in range(self.ny):
                idx = j * self.nx + i
                if i == 0 or i == self.nx-1 or j == 0 or j == self.ny-1:
                    # 邊界點
                    L[idx, :] = 0
                    L[idx, idx] = 1
                    f_vector[idx] = 0
        
        # 求解線性系統
        u_vector = spsolve(L, f_vector)
        u_solution = u_vector.reshape((self.ny, self.nx))
        
        return u_solution

def demo_2d_elliptic():
    """演示二維橢圓PDE"""
    # 網格設置
    nx, ny = 25, 25  # 較小的網格以節省計算
    solver = Elliptic2D(nx, ny)
    
    # 定義右端項 f(x,y)
    f_values = 2 * np.sin(np.pi * solver.X) * np.sin(np.pi * solver.Y)
    
    # 求解PDE
    u_solution = solver.solve_poisson_2d(f_values)
    
    # 可視化結果
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 右端項 f(x,y)
    im1 = axes[0].contourf(solver.X, solver.Y, f_values, levels=20, cmap='RdBu')
    axes[0].set_title('右端項 f(x,y)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # 解 u(x,y)
    im2 = axes[1].contourf(solver.X, solver.Y, u_solution, levels=20, cmap='viridis')
    axes[1].set_title('解 u(x,y)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # 3D視圖
    ax3d = fig.add_subplot(133, projection='3d')
    surf = ax3d.plot_surface(solver.X, solver.Y, u_solution, cmap='viridis', alpha=0.8)
    ax3d.set_title('3D視圖 u(x,y)')
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('u')
    
    plt.tight_layout()
    plt.show()
    
    print("二維橢圓PDE求解完成！")
    print(f"網格大小: {nx} × {ny}")
    print(f"解的範圍: [{np.min(u_solution):.4f}, {np.max(u_solution):.4f}]")

def compare_1d_vs_2d():
    """比較一維和二維的復雜度"""
    print("一維 vs 二維橢圓PDE復雜度比較:")
    print("=" * 50)
    
    # 一維情況
    n1d = 64
    unknowns_1d = n1d
    matrix_size_1d = n1d**2
    
    print(f"一維情況 (n={n1d}):")
    print(f"  - 未知數個數: {unknowns_1d}")
    print(f"  - 矩陣大小: {matrix_size_1d}")
    print(f"  - 內存需求: ~{matrix_size_1d * 8 / 1024:.1f} KB")
    
    # 二維情況
    n2d = 64
    unknowns_2d = n2d * n2d
    matrix_size_2d = unknowns_2d**2
    
    print(f"\n二維情況 (n={n2d}×{n2d}):")
    print(f"  - 未知數個數: {unknowns_2d}")
    print(f"  - 矩陣大小: {matrix_size_2d}")
    print(f"  - 內存需求: ~{matrix_size_2d * 8 / 1024 / 1024:.1f} MB")
    
    print(f"\n複雜度比值:")
    print(f"  - 未知數比值: {unknowns_2d / unknowns_1d:.0f}倍")
    print(f"  - 矩陣大小比值: {matrix_size_2d / matrix_size_1d:.0f}倍")

if __name__ == "__main__":
    print("二維橢圓PDE示例")
    print("=" * 30)
    
    # 比較復雜度
    compare_1d_vs_2d()
    
    print("\n運行二維PDE示例...")
    demo_2d_elliptic()
    
    print("\n說明:")
    print("這個例子展示了如何擴展到二維情況，")
    print("但實際的DeepONet實現會需要更多的計算資源和更複雜的數據生成策略。") 