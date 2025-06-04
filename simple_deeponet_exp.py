import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 設置隨機種子
torch.manual_seed(42)
np.random.seed(42)

class SimpleDeepONet(nn.Module):
    """簡單的DeepONet學習指數函數算子"""
    
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim=64, output_dim=64):
        super(SimpleDeepONet, self).__init__()
        
        # Branch Network (編碼輸入函數)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
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
            output = torch.mm(branch_out, trunk_out.T) + self.bias
        else:
            output = torch.sum(branch_out * trunk_out, dim=-1) + self.bias
            
        return output

def generate_exponential_data(n_samples, x_points):
    """生成指數算子的訓練數據
    
    算子: 學習指數算子 exp(f(x))
    輸入: a*sin(b*x + c) + d (有偏移的正弦函數)
    輸出: exp(a*sin(b*x + c) + d)
    """
    
    input_functions = []
    output_functions = []
    
    print(f"生成 {n_samples} 個指數算子樣本...")
    
    for i in range(n_samples):
        # 隨機生成參數 (控制輸入函數的範圍，避免指數爆炸)
        a = np.random.uniform(0.3, 1.0)   # 振幅參數
        b = np.random.uniform(0.5, 2.0)   # 頻率參數
        c = np.random.uniform(0, 2*np.pi) # 相位參數
        d = np.random.uniform(-0.5, 0.5)  # 偏移參數 (控制在小範圍內)
        
        # 輸入函數: a*sin(b*x + c) + d
        input_func = a * np.sin(b * x_points + c) + d
        
        # 輸出函數: exp(input_func)
        output_func = np.exp(np.clip(input_func, -10, 5))  # 防止數值溢出
        
        input_functions.append(input_func)
        output_functions.append(output_func)
        
        if (i + 1) % 1000 == 0:
            print(f"已完成 {i + 1}/{n_samples} 樣本")
    
    return np.array(input_functions), np.array(output_functions)

def generate_logarithm_data(n_samples, x_points):
    """生成對數算子的訓練數據
    
    算子: 學習對數算子 ln(f(x))
    輸入: a*sin(b*x + c) + d + 2 (確保函數值為正)
    輸出: ln(a*sin(b*x + c) + d + 2)
    """
    
    input_functions = []
    output_functions = []
    
    print(f"生成 {n_samples} 個對數算子樣本...")
    
    for i in range(n_samples):
        # 隨機生成參數
        a = np.random.uniform(0.3, 0.8)   # 振幅參數 (較小，確保函數值為正)
        b = np.random.uniform(0.5, 2.0)   # 頻率參數
        c = np.random.uniform(0, 2*np.pi) # 相位參數
        d = np.random.uniform(0.5, 1.5)   # 偏移參數 (確保函數值 > 0)
        
        # 輸入函數: a*sin(b*x + c) + d + 1 (確保恆為正)
        input_func = a * np.sin(b * x_points + c) + d + 1.0
        
        # 確保所有值都大於0
        input_func = np.maximum(input_func, 0.1)
        
        # 輸出函數: ln(input_func)
        output_func = np.log(input_func)
        
        input_functions.append(input_func)
        output_functions.append(output_func)
        
        if (i + 1) % 1000 == 0:
            print(f"已完成 {i + 1}/{n_samples} 樣本")
    
    return np.array(input_functions), np.array(output_functions)

def train_simple_deeponet(model, input_funcs, output_funcs, x_points, epochs=5000, lr=1e-3):
    """訓練簡單DeepONet"""
    
    # 準備數據
    branch_inputs = torch.FloatTensor(input_funcs)
    trunk_inputs = torch.FloatTensor(x_points.reshape(-1, 1))
    targets = torch.FloatTensor(output_funcs)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    print("開始訓練DeepONet...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(branch_inputs, trunk_inputs)
        loss = criterion(outputs, targets)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses

def test_simple_deeponet(model, x_points, operator_type='exponential', n_test=4):
    """測試簡單DeepONet"""
    
    if (operator_type == 'exponential' or operator_type == 'logarithm') and n_test == 4:
        # 針對指數/對數算子的2x2佈局
        plt.figure(figsize=(12, 10))
        
        for i in range(n_test):
            if operator_type == 'exponential':
                # 生成指數算子測試樣本
                a_test = np.random.uniform(0.3, 1.0)
                b_test = np.random.uniform(0.5, 2.0)
                c_test = np.random.uniform(0, 2*np.pi)
                d_test = np.random.uniform(-0.5, 0.5)
                
                # 輸入函數
                input_test = a_test * np.sin(b_test * x_points + c_test) + d_test
                
                # 真實輸出
                output_true = np.exp(np.clip(input_test, -10, 5))
                
                true_label = 'True exp(f)'
                pred_label = 'DeepONet exp(f)'
                title = f'Sample {i+1}: Exponential Operator'
                input_label = f'Input: {a_test:.2f}sin({b_test:.2f}x+{c_test:.2f})+{d_test:.2f}'
                
            else:  # logarithm
                # 生成對數算子測試樣本
                a_test = np.random.uniform(0.3, 0.8)
                b_test = np.random.uniform(0.5, 2.0)
                c_test = np.random.uniform(0, 2*np.pi)
                d_test = np.random.uniform(0.5, 1.5)
                
                # 輸入函數 (確保為正)
                input_test = a_test * np.sin(b_test * x_points + c_test) + d_test + 1.0
                input_test = np.maximum(input_test, 0.1)
                
                # 真實輸出
                output_true = np.log(input_test)
                
                true_label = 'True ln(f)'
                pred_label = 'DeepONet ln(f)'
                title = f'Sample {i+1}: Logarithm Operator'
                input_label = f'Input: {a_test:.2f}sin({b_test:.2f}x)+{d_test+1:.2f}'
            
            # DeepONet 預測
            branch_input = torch.FloatTensor(input_test).unsqueeze(0)
            trunk_input = torch.FloatTensor(x_points.reshape(-1, 1))
            
            with torch.no_grad():
                output_pred = model(branch_input, trunk_input).squeeze().numpy()
            
            # 2x2佈局
            plt.subplot(2, 2, i+1)
            plt.plot(x_points, input_test, 'g-', linewidth=2, label=input_label)
            plt.plot(x_points, output_true, 'b-', label=true_label, linewidth=2)
            plt.plot(x_points, output_pred, 'r--', label=pred_label, linewidth=2)
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('Function Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def demo_exponential_operator():
    """演示指數算子學習"""
    print("=" * 60)
    print("DeepONet學習指數算子: f(x) → exp(f(x))")
    print("=" * 60)
    
    # 設定參數
    n_points = 50
    x_points = np.linspace(0, 2*np.pi, n_points)
    
    # 生成訓練數據
    n_train = 10000
    input_funcs, output_funcs = generate_exponential_data(n_train, x_points)
    
    # 初始化DeepONet
    model = SimpleDeepONet(
        branch_input_dim=n_points,
        trunk_input_dim=1,
        hidden_dim=64,
        output_dim=64
    )
    
    # 訓練模型
    losses = train_simple_deeponet(model, input_funcs, output_funcs, x_points, epochs=5000)
    
    # 繪製收斂性 - 只顯示一張圖
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.title('Exponential Operator Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 測試模型 - 4個樣本排成2x2
    test_simple_deeponet(model, x_points, operator_type='exponential', n_test=4)
    
    print(f"指數算子學習完成！最終損失: {losses[-1]:.6f}")

def demo_logarithm_operator():
    """演示對數算子學習"""
    print("=" * 60)
    print("DeepONet學習對數算子: f(x) → ln(f(x))")
    print("=" * 60)
    
    # 設定參數
    n_points = 50
    x_points = np.linspace(0, 2*np.pi, n_points)
    
    # 生成訓練數據
    n_train = 10000
    input_funcs, output_funcs = generate_logarithm_data(n_train, x_points)
    
    # 初始化DeepONet
    model = SimpleDeepONet(
        branch_input_dim=n_points,
        trunk_input_dim=1,
        hidden_dim=64,
        output_dim=64
    )
    
    # 訓練模型
    losses = train_simple_deeponet(model, input_funcs, output_funcs, x_points, epochs=5000)
    
    # 繪製收斂性 - 只顯示一張圖
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.title('Logarithm Operator Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 測試模型 - 4個樣本排成2x2
    test_simple_deeponet(model, x_points, operator_type='logarithm', n_test=4)
    
    print(f"對數算子學習完成！最終損失: {losses[-1]:.6f}")

def show_concept_explanation():
    """解釋指數/對數 DeepONet的基本概念"""
    print("=" * 70)
    print("指數/對數 DeepONet 基本概念解釋")
    print("=" * 70)
    print()
    print("1. 指數算子:")
    print("   f(x) → exp(f(x))")
    print("   - 非線性變換，快速增長")
    print("   - 將小的變化放大為大的變化")
    print()
    print("2. 對數算子:")
    print("   f(x) → ln(f(x))  (要求 f(x) > 0)")
    print("   - 指數的逆運算")
    print("   - 將大的變化壓縮為小的變化")
    print()
    print("3. DeepONet學習的算子:")
    print("   ┌─────────────────────┐    ┌─────────────────────┐")
    print("   │    指數算子 exp     │    │    對數算子 ln      │")
    print("   │                     │    │                     │")
    print("   │ a*sin(bx+c)+d →     │    │ a*sin(bx+c)+d+1 →   │")
    print("   │ exp(a*sin(bx+c)+d)  │    │ ln(a*sin(bx+c)+d+1) │")
    print("   └─────────────────────┘    └─────────────────────┘")
    print()
    print("4. 本例子特點:")
    print("   - 學習強非線性變換")
    print("   - 展示DeepONet處理複雜數學函數的能力")
    print("   - 指數和對數是互逆操作")
    print("=" * 70)

def main():
    """主函數"""
    show_concept_explanation()
    
    print("\n選擇要演示的算子:")
    print("1. 指數算子 exp(f(x))")
    print("2. 對數算子 ln(f(x))")
    print("3. 兩個都演示")
    
    choice = input("請輸入選項 (1/2/3, 直接按Enter默認選3): ").strip()
    
    if choice == '1':
        demo_exponential_operator()
    elif choice == '2':
        demo_logarithm_operator()
    else:
        print("\n演示兩個算子...")
        demo_exponential_operator()
        print("\n" + "="*50)
        demo_logarithm_operator()
    
    print("\n指數/對數 DeepONet演示完成!")
    print("這個例子展示了DeepONet如何學習強非線性的數學變換。")
    print("指數和對數算子展現了深度學習在複雜函數映射上的潛力。")

if __name__ == "__main__":
    main() 