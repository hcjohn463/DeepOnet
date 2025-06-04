import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import cholesky
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import seaborn as sns

# 設置隨機種子
torch.manual_seed(42)
np.random.seed(42)

class DeepONet(nn.Module):
    """DeepONet for learning elliptic PDE operators"""
    
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim=100, output_dim=100):
        super(DeepONet, self).__init__()
        
        # Branch Network (編碼輸入函數)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Trunk Network (編碼查詢位置)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 偏置項
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, branch_input, trunk_input):
        # Branch network output
        branch_out = self.branch_net(branch_input)  # [batch_size, output_dim]
        
        # Trunk network output  
        trunk_out = self.trunk_net(trunk_input)      # [n_points, output_dim]
        
        # 計算內積
        if branch_out.dim() == 2 and trunk_out.dim() == 2:
            # branch_out: [batch_size, output_dim]
            # trunk_out: [n_points, output_dim]
            output = torch.mm(branch_out, trunk_out.T) + self.bias
        else:
            output = torch.sum(branch_out * trunk_out, dim=-1) + self.bias
            
        return output

class GaussianRandomField:
    """生成高斯隨機場"""
    
    def __init__(self, x_points, sigma=1.0, length_scale=0.1):
        self.x_points = x_points
        self.sigma = sigma
        self.length_scale = length_scale
        self.n_points = len(x_points)
        
        # 預計算協方差矩陣
        self.cov_matrix = self._compute_covariance_matrix()
        self.L = cholesky(self.cov_matrix, lower=True)
        
    def _compute_covariance_matrix(self):
        """計算RBF核的協方差矩陣"""
        X1, X2 = np.meshgrid(self.x_points, self.x_points)
        distances = np.abs(X1 - X2)
        cov = self.sigma**2 * np.exp(-distances**2 / (2 * self.length_scale**2))
        # 添加小的對角線項以確保數值穩定性
        cov += 1e-6 * np.eye(self.n_points)
        return cov
    
    def sample(self, n_samples=1):
        """生成高斯隨機場樣本"""
        z = np.random.normal(0, 1, (self.n_points, n_samples))
        samples = self.L @ z
        return samples.T if n_samples > 1 else samples.flatten()

class EllipticPDESolver:
    """橢圓PDE求解器: -div(a(x) * grad(u)) = f(x)"""
    
    def __init__(self, x_points):
        self.x_points = x_points
        self.dx = x_points[1] - x_points[0]
        self.n = len(x_points)
        
    def solve(self, a_values, f_values):
        """
        求解橢圓PDE
        a_values: 擴散係數 a(x)
        f_values: 右端項 f(x)
        """
        # 確保 a(x) > 0
        a_values = np.maximum(a_values, 0.01)
        
        # 構建有限差分矩陣
        # -d/dx(a(x) du/dx) ≈ -(a_{i+1/2}(u_{i+1}-u_i) - a_{i-1/2}(u_i-u_{i-1}))/dx^2
        
        # 在網格中點插值 a 值
        a_half = 0.5 * (a_values[:-1] + a_values[1:])
        
        # 構建係數
        diag_main = np.zeros(self.n)
        diag_upper = np.zeros(self.n-1)
        diag_lower = np.zeros(self.n-1)
        
        # 內部點
        for i in range(1, self.n-1):
            diag_lower[i-1] = -a_half[i-1] / self.dx**2
            diag_main[i] = (a_half[i-1] + a_half[i]) / self.dx**2
            diag_upper[i] = -a_half[i] / self.dx**2
            
        # 邊界條件 u(0) = u(1) = 0
        diag_main[0] = 1.0
        diag_main[-1] = 1.0
        
        # 構建稀疏矩陣
        A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csr')
        
        # 右端項
        rhs = f_values.copy()
        rhs[0] = 0.0  # 邊界條件
        rhs[-1] = 0.0
        
        # 求解線性系統
        solution = spsolve(A, rhs)
        return solution

def generate_training_data(n_samples, x_points, grf_generator):
    """生成訓練數據"""
    solver = EllipticPDESolver(x_points)
    
    # 存儲數據
    a_samples = []
    u_solutions = []
    
    print(f"生成 {n_samples} 個訓練樣本...")
    
    for i in range(n_samples):
        # 生成擴散係數 a(x)
        a_sample = grf_generator.sample() + 1.0  # 確保 a(x) > 0
        
        # 生成右端項 f(x) (簡單的正弦函數)
        f_sample = np.sin(2 * np.pi * x_points) + 0.5 * np.sin(4 * np.pi * x_points)
        
        # 求解PDE
        u_solution = solver.solve(a_sample, f_sample)
        
        a_samples.append(a_sample)
        u_solutions.append(u_solution)
        
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{n_samples} 樣本")
    
    return np.array(a_samples), np.array(u_solutions)

def prepare_data_for_deeponet(a_samples, u_solutions, x_points):
    """準備DeepONet的訓練數據"""
    n_samples, n_points = a_samples.shape
    
    # Branch network 輸入: a(x) 在感測器點的值
    branch_inputs = torch.FloatTensor(a_samples)
    
    # Trunk network 輸入: 查詢點座標
    trunk_inputs = torch.FloatTensor(x_points.reshape(-1, 1))
    
    # 目標輸出: u(x) 的值
    targets = torch.FloatTensor(u_solutions)
    
    return branch_inputs, trunk_inputs, targets

def train_deeponet(model, branch_inputs, trunk_inputs, targets, epochs=5000, lr=1e-3):
    """訓練DeepONet"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=400, factor=0.7, verbose=True)
    criterion = nn.MSELoss()
    
    losses = []
    best_loss = float('inf')
    
    print("開始訓練DeepONet...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(branch_inputs, trunk_inputs)
        loss = criterion(outputs, targets)
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        losses.append(loss.item())
        
        # 記錄最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Best: {best_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    return losses

def test_deeponet(model, x_points, grf_generator, n_test=5):
    """測試DeepONet"""
    solver = EllipticPDESolver(x_points)
    
    plt.figure(figsize=(15, 10))
    
    for i in range(n_test):
        # 生成測試樣本
        a_test = grf_generator.sample() + 1.0
        f_test = np.sin(2 * np.pi * x_points) + 0.5 * np.sin(4 * np.pi * x_points)
        u_true = solver.solve(a_test, f_test)
        
        # DeepONet 預測
        branch_input = torch.FloatTensor(a_test).unsqueeze(0)
        trunk_input = torch.FloatTensor(x_points.reshape(-1, 1))
        
        with torch.no_grad():
            u_pred = model(branch_input, trunk_input).squeeze().numpy()
        
        # 可視化
        plt.subplot(2, 3, i+1)
        plt.plot(x_points, u_true, 'b-', label='True', linewidth=2)
        plt.plot(x_points, u_pred, 'r--', label='DeepONet', linewidth=2)
        plt.title(f'Test Sample {i+1}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(True)
    
    # 顯示一個輸入函數 a(x)
    plt.subplot(2, 3, 6)
    plt.plot(x_points, a_test, 'g-', linewidth=2)
    plt.title('Last Input Function a(x)')
    plt.xlabel('x')
    plt.ylabel('a(x)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # 設定參數
    n_points = 64  # 增加網格點數
    x_points = np.linspace(0, 1, n_points)
    
    # 初始化高斯隨機場生成器 - 調整參數以獲得更多樣的數據
    grf_generator = GaussianRandomField(x_points, sigma=0.8, length_scale=0.08)
    
    # 生成訓練數據 - 大幅增加訓練樣本數
    n_train = 5000  # 從1000增加到5000
    a_samples, u_solutions = generate_training_data(n_train, x_points, grf_generator)
    
    # 準備DeepONet數據
    branch_inputs, trunk_inputs, targets = prepare_data_for_deeponet(a_samples, u_solutions, x_points)
    
    # 初始化DeepONet - 增加網路容量
    model = DeepONet(
        branch_input_dim=n_points,
        trunk_input_dim=1,
        hidden_dim=150,  # 從100增加到150
        output_dim=150   # 從100增加到150
    )
    
    # 訓練模型 - 增加訓練epoch數並改善訓練策略
    losses = train_deeponet(model, branch_inputs, trunk_inputs, targets, epochs=5000, lr=1e-3)  # 從2000增加到5000
    
    # 繪製收斂性
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.yscale('log')
    plt.title('Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(losses[-1000:])  # 顯示最後1000個epoch
    plt.title('Training Loss (Last 1000 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # 新增：移動平均線
    plt.subplot(1, 3, 3)
    window = 100
    if len(losses) > window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label='Moving Average')
        plt.plot(losses[window-1:], alpha=0.3, label='Raw Loss')
    else:
        plt.plot(losses, label='Training Loss')
    plt.yscale('log')
    plt.title(f'Moving Average (window={window})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 測試模型
    test_deeponet(model, x_points, grf_generator)
    
    # 顯示一些高斯隨機場樣本
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i in range(10):
        sample = grf_generator.sample() + 1.0
        plt.plot(x_points, sample, alpha=0.7)
    plt.title('Gaussian Random Field Samples for a(x)')
    plt.xlabel('x')
    plt.ylabel('a(x)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    # 顯示協方差矩陣
    plt.imshow(grf_generator.cov_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Covariance Matrix')
    
    plt.subplot(2, 2, 3)
    # 顯示一些解的樣本
    for i in range(5):
        plt.plot(x_points, u_solutions[i], alpha=0.7)
    plt.title('PDE Solution Samples')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # 顯示輸入函數與解的關係
    plt.scatter(a_samples[:100, n_points//2], u_solutions[:100, n_points//2], alpha=0.6)
    plt.xlabel('a(x=0.5)')
    plt.ylabel('u(x=0.5)')
    plt.title('Input-Output Relationship at x=0.5')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("橢圓PDE算子學習完成！")

if __name__ == "__main__":
    main() 