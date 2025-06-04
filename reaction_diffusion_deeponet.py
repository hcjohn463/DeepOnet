import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from numpy.polynomial.chebyshev import chebval
import warnings
warnings.filterwarnings('ignore')

# 設置隨機種子
torch.manual_seed(42)
np.random.seed(42)

class DeepONet(nn.Module):
    """DeepONet for learning reaction-diffusion operators"""
    
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim=100, output_dim=100):
        super(DeepONet, self).__init__()
        
        # Branch Network (編碼初始條件)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Trunk Network (編碼時空座標)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
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

class ChebyshevGenerator:
    """切比雪夫多項式函數生成器"""
    
    def __init__(self, x_points, max_degree=10):
        self.x_points = x_points
        self.max_degree = max_degree
        # 將 x_points 從 [0,1] 映射到 [-1,1] (切比雪夫多項式的標準區間)
        self.x_cheb = 2 * x_points - 1
        
    def sample_coefficients(self, degree=None):
        """采樣切比雪夫多項式係數"""
        if degree is None:
            degree = np.random.randint(3, self.max_degree + 1)
        
        # 生成係數，高階項係數較小
        coeffs = np.random.normal(0, 1, degree + 1)
        for i in range(degree + 1):
            coeffs[i] *= 1.0 / (i + 1)**0.5  # 減少高階項的貢獻
            
        return coeffs
    
    def evaluate_chebyshev(self, coeffs):
        """評估切比雪夫多項式"""
        return chebval(self.x_cheb, coeffs)
    
    def sample_function(self, degree=None):
        """生成隨機的切比雪夫多項式函數"""
        coeffs = self.sample_coefficients(degree)
        values = self.evaluate_chebyshev(coeffs)
        return values, coeffs

class ReactionDiffusionSolver:
    """反應-擴散方程求解器: ∂u/∂t = D∇²u + f(u)"""
    
    def __init__(self, x_points, D=0.01, reaction_type='fisher'):
        self.x_points = x_points
        self.dx = x_points[1] - x_points[0]
        self.n = len(x_points)
        self.D = D  # 擴散係數
        self.reaction_type = reaction_type
        
        # 構建拉普拉斯算子矩陣 (二階中心差分)
        self.laplacian = self._build_laplacian_matrix()
    
    def _build_laplacian_matrix(self):
        """構建拉普拉斯算子的有限差分矩陣"""
        # 二階中心差分: u''(x) ≈ (u(x+h) - 2u(x) + u(x-h))/h²
        diag_main = -2 * np.ones(self.n) / self.dx**2
        diag_off = np.ones(self.n - 1) / self.dx**2
        
        # 構建矩陣
        L = np.zeros((self.n, self.n))
        np.fill_diagonal(L, diag_main)
        np.fill_diagonal(L[1:], diag_off)
        np.fill_diagonal(L[:, 1:], diag_off)
        
        # 邊界條件: Neumann (零梯度)
        L[0, 1] = 2 / self.dx**2
        L[-1, -2] = 2 / self.dx**2
        
        return L
    
    def reaction_term(self, u):
        """反應項 f(u)"""
        if self.reaction_type == 'fisher':
            # Fisher equation: f(u) = r*u*(1-u)
            r = 1.0
            return r * u * (1 - u)
        elif self.reaction_type == 'allen_cahn':
            # Allen-Cahn equation: f(u) = u - u³
            return u - u**3
        else:
            # 線性反應
            return -0.1 * u
    
    def ode_system(self, t, u):
        """ODE系統: du/dt = D*∇²u + f(u)"""
        diffusion = self.D * self.laplacian @ u
        reaction = self.reaction_term(u)
        return diffusion + reaction
    
    def solve(self, u0, t_final):
        """求解反應-擴散方程"""
        # 時間網格
        t_span = (0, t_final)
        t_eval = np.linspace(0, t_final, 50)
        
        # 求解ODE
        sol = solve_ivp(self.ode_system, t_span, u0, t_eval=t_eval, 
                       method='RK45', rtol=1e-6, atol=1e-8)
        
        if sol.success:
            return sol.y[:, -1]  # 返回最終時刻的解
        else:
            print("Warning: ODE solver failed")
            return u0  # 返回初始條件

def generate_training_data(n_samples, x_points, cheb_generator, solver, t_final=1.0):
    """生成訓練數據"""
    
    # 存儲數據
    u0_samples = []
    uf_solutions = []
    
    print(f"生成 {n_samples} 個訓練樣本...")
    
    successful_samples = 0
    attempts = 0
    max_attempts = n_samples * 2  # 最多嘗試2倍的樣本數
    
    while successful_samples < n_samples and attempts < max_attempts:
        attempts += 1
        
        # 生成初始條件使用切比雪夫多項式
        # 增加多樣性：隨機選擇不同的參數
        degree_variation = np.random.randint(3, cheb_generator.max_degree + 1)
        u0_values, coeffs = cheb_generator.sample_function(degree=degree_variation)
        
        # 確保初始條件在合理範圍內，並增加多樣性
        scaling_factor = np.random.uniform(0.5, 1.5)  # 添加隨機縮放
        u0_values = np.tanh(u0_values * scaling_factor)  # 壓縮到 [-1, 1]
        u0_values = (u0_values + 1) / 2  # 映射到 [0, 1]
        
        # 添加小的隨機擾動以增加多樣性
        noise = np.random.normal(0, 0.02, len(u0_values))
        u0_values = np.clip(u0_values + noise, 0, 1)
        
        # 求解反應-擴散方程
        try:
            uf_solution = solver.solve(u0_values, t_final)
            
            # 檢查解的質量
            if np.isfinite(uf_solution).all() and not np.isnan(uf_solution).any():
                u0_samples.append(u0_values)
                uf_solutions.append(uf_solution)
                successful_samples += 1
                
                if successful_samples % 200 == 0:
                    print(f"已完成 {successful_samples}/{n_samples} 樣本 (嘗試次數: {attempts})")
            
        except Exception as e:
            if attempts % 500 == 0:
                print(f"跳過樣本 (嘗試 {attempts}): {str(e)[:50]}...")
            continue
    
    if successful_samples < n_samples:
        print(f"警告：只成功生成了 {successful_samples}/{n_samples} 個樣本")
    
    return np.array(u0_samples), np.array(uf_solutions)

def prepare_data_for_deeponet(u0_samples, uf_solutions, x_points, t_final):
    """準備DeepONet的訓練數據"""
    n_samples, n_points = u0_samples.shape
    
    # Branch network 輸入: 初始條件 u0(x)
    branch_inputs = torch.FloatTensor(u0_samples)
    
    # Trunk network 輸入: 空間座標和時間 [x, t]
    trunk_inputs = []
    for x in x_points:
        trunk_inputs.append([x, t_final])
    trunk_inputs = torch.FloatTensor(trunk_inputs)
    
    # 目標輸出: 最終時刻的解 u(x, t_final)
    targets = torch.FloatTensor(uf_solutions)
    
    return branch_inputs, trunk_inputs, targets

def train_deeponet(model, branch_inputs, trunk_inputs, targets, epochs=5000, lr=1e-3):
    """訓練DeepONet"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.6, verbose=True)
    criterion = nn.MSELoss()
    
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    
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
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Best: {best_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
        # 早停策略（可選）
        if patience_counter > 2000:  # 如果2000個epoch沒有改善就停止
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return losses

def test_deeponet(model, x_points, cheb_generator, solver, t_final=1.0, n_test=5):
    """測試DeepONet"""
    
    plt.figure(figsize=(15, 12))
    
    for i in range(n_test):
        # 生成測試樣本
        u0_test, coeffs = cheb_generator.sample_function()
        u0_test = np.tanh(u0_test)
        u0_test = (u0_test + 1) / 2
        
        # 真實解
        uf_true = solver.solve(u0_test, t_final)
        
        # DeepONet 預測
        branch_input = torch.FloatTensor(u0_test).unsqueeze(0)
        trunk_input = torch.FloatTensor([[x, t_final] for x in x_points])
        
        with torch.no_grad():
            uf_pred = model(branch_input, trunk_input).squeeze().numpy()
        
        # 可視化初始條件
        plt.subplot(3, n_test, i+1)
        plt.plot(x_points, u0_test, 'g-', linewidth=2)
        plt.title(f'Initial Condition {i+1}')
        plt.xlabel('x')
        plt.ylabel('u₀(x)')
        plt.grid(True)
        plt.ylim([-0.1, 1.1])
        
        # 可視化最終解
        plt.subplot(3, n_test, n_test + i + 1)
        plt.plot(x_points, uf_true, 'b-', label='True', linewidth=2)
        plt.plot(x_points, uf_pred, 'r--', label='DeepONet', linewidth=2)
        plt.title(f'Final Solution {i+1}')
        plt.xlabel('x')
        plt.ylabel(f'u(x, t={t_final})')
        plt.legend()
        plt.grid(True)
        
        # 計算誤差
        error = np.abs(uf_true - uf_pred)
        plt.subplot(3, n_test, 2*n_test + i + 1)
        plt.plot(x_points, error, 'k-', linewidth=2)
        plt.title(f'Absolute Error {i+1}')
        plt.xlabel('x')
        plt.ylabel('|u_true - u_pred|')
        plt.grid(True)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def visualize_time_evolution(model, x_points, cheb_generator, solver, t_final=1.0):
    """可視化時間演化"""
    # 生成一個測試樣本
    u0_test, coeffs = cheb_generator.sample_function()
    u0_test = np.tanh(u0_test)
    u0_test = (u0_test + 1) / 2
    
    # 多個時間點
    time_points = np.linspace(0, t_final, 6)
    
    plt.figure(figsize=(18, 10))
    
    for i, t in enumerate(time_points):
        # 真實解（重新求解到時間t）
        if t == 0:
            u_true = u0_test
        else:
            u_true = solver.solve(u0_test, t)
        
        # DeepONet預測
        if t == 0:
            u_pred = u0_test
        else:
            branch_input = torch.FloatTensor(u0_test).unsqueeze(0)
            trunk_input = torch.FloatTensor([[x, t] for x in x_points])
            
            with torch.no_grad():
                u_pred = model(branch_input, trunk_input).squeeze().numpy()
        
        plt.subplot(2, 3, i+1)
        plt.plot(x_points, u_true, 'b-', label='True Solution', linewidth=2)
        if t > 0:
            plt.plot(x_points, u_pred, 'r--', label='DeepONet', linewidth=2)
        plt.title(f't = {t:.2f}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.legend()
        plt.grid(True)
        plt.ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.show()

def main():
    # 設定參數
    n_points = 80  # 增加網格點數 從64增加到80
    x_points = np.linspace(0, 1, n_points)
    
    # 初始化切比雪夫多項式生成器 - 增加多項式階數
    cheb_generator = ChebyshevGenerator(x_points, max_degree=12)  # 從8增加到12
    
    # 初始化反應-擴散方程求解器
    solver = ReactionDiffusionSolver(x_points, D=0.01, reaction_type='fisher')
    
    # 生成訓練數據 - 大幅增加訓練樣本數
    n_train = 3000  # 從800增加到3000
    t_final = 0.5
    u0_samples, uf_solutions = generate_training_data(n_train, x_points, cheb_generator, solver, t_final)
    
    print(f"成功生成 {len(u0_samples)} 個訓練樣本")
    
    # 計算一些統計信息
    print(f"訓練數據統計:")
    print(f"  - 初始條件範圍: [{np.min(u0_samples):.3f}, {np.max(u0_samples):.3f}]")
    print(f"  - 最終解範圍: [{np.min(uf_solutions):.3f}, {np.max(uf_solutions):.3f}]")
    print(f"  - 初始條件標準差: {np.std(u0_samples):.3f}")
    print(f"  - 最終解標準差: {np.std(uf_solutions):.3f}")
    
    # 準備DeepONet數據
    branch_inputs, trunk_inputs, targets = prepare_data_for_deeponet(u0_samples, uf_solutions, x_points, t_final)
    
    print(f"DeepONet輸入維度:")
    print(f"  - Branch輸入: {branch_inputs.shape}")
    print(f"  - Trunk輸入: {trunk_inputs.shape}")
    print(f"  - 目標輸出: {targets.shape}")
    
    # 初始化DeepONet - 增加網路容量
    model = DeepONet(
        branch_input_dim=n_points,
        trunk_input_dim=2,  # [x, t]
        hidden_dim=150,  # 從100增加到150
        output_dim=150   # 從100增加到150
    )
    
    # 計算模型參數數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型參數統計:")
    print(f"  - 總參數數: {total_params:,}")
    print(f"  - 可訓練參數數: {trainable_params:,}")
    
    # 訓練模型 - 增加訓練epoch數
    import time
    start_time = time.time()
    losses = train_deeponet(model, branch_inputs, trunk_inputs, targets, epochs=5000)  # 從3000增加到5000
    training_time = time.time() - start_time
    
    print(f"訓練完成!")
    print(f"  - 訓練時間: {training_time:.2f} 秒")
    print(f"  - 最終損失: {losses[-1]:.6f}")
    print(f"  - 最佳損失: {min(losses):.6f}")
    
    # 繪製收斂性
    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 4, 1)
    plt.plot(losses)
    plt.yscale('log')
    plt.title('Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(losses[-1500:])  # 顯示最後1500個epoch
    plt.title('Training Loss (Last 1500 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    # 移動平均
    window = 100
    if len(losses) > window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label='Moving Average', linewidth=2)
        plt.plot(losses[window-1:], alpha=0.3, label='Raw Loss')
    else:
        plt.plot(losses, label='Training Loss')
    plt.yscale('log')
    plt.title(f'Moving Average (window={window})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    # 損失改善率
    if len(losses) > 50:
        improvement_rate = []
        for i in range(50, len(losses)):
            recent_min = min(losses[i-50:i])
            current = losses[i]
            improvement_rate.append((recent_min - current) / recent_min if recent_min > 0 else 0)
        plt.plot(improvement_rate)
        plt.title('Loss Improvement Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Relative Improvement')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 測試模型
    test_deeponet(model, x_points, cheb_generator, solver, t_final)
    
    # 可視化時間演化
    visualize_time_evolution(model, x_points, cheb_generator, solver, t_final)
    
    # 顯示切比雪夫多項式和數據特性
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    # 顯示一些切比雪夫多項式樣本
    for i in range(10):
        u0, coeffs = cheb_generator.sample_function()
        u0 = np.tanh(u0)
        u0 = (u0 + 1) / 2
        plt.plot(x_points, u0, alpha=0.7)
    plt.title('Initial Condition Samples (Chebyshev)')
    plt.xlabel('x')
    plt.ylabel('u₀(x)')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    # 顯示一些解的樣本
    for i in range(min(5, len(uf_solutions))):
        plt.plot(x_points, uf_solutions[i], alpha=0.7)
    plt.title('Final Solution Samples')
    plt.xlabel('x')
    plt.ylabel(f'u(x, t={t_final})')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    # 顯示係數分佈
    sample_coeffs = []
    for _ in range(100):
        _, coeffs = cheb_generator.sample_function()
        sample_coeffs.append(coeffs)
    
    max_len = max(len(c) for c in sample_coeffs)
    coeffs_matrix = np.zeros((len(sample_coeffs), max_len))
    for i, coeffs in enumerate(sample_coeffs):
        coeffs_matrix[i, :len(coeffs)] = coeffs
    
    plt.boxplot([coeffs_matrix[:, i] for i in range(min(8, max_len))])
    plt.title('Chebyshev Coefficient Distribution')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Coefficient Value')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    # 輸入輸出關係
    if len(u0_samples) > 0:
        plt.scatter(u0_samples[:100, n_points//2], uf_solutions[:100, n_points//2], alpha=0.6)
        plt.xlabel('u₀(x=0.5)')
        plt.ylabel(f'u(x=0.5, t={t_final})')
        plt.title('Input-Output Relationship at x=0.5')
        plt.grid(True)
    
    plt.subplot(2, 3, 5)
    # 反應項可視化
    u_range = np.linspace(0, 1, 100)
    reaction_values = solver.reaction_term(u_range)
    plt.plot(u_range, reaction_values, 'b-', linewidth=2)
    plt.title(f'Reaction Term f(u) - {solver.reaction_type}')
    plt.xlabel('u')
    plt.ylabel('f(u)')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    # 損失收斂性（線性尺度）
    plt.plot(losses[-500:])
    plt.title('Final Convergence (Linear Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("反應-擴散方程算子學習完成！")

if __name__ == "__main__":
    main() 