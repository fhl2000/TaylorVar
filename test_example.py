from taylorvar.taylor_mode_utils import *
import torch

def test_swish_derivatives():
    """测试Swish激活函数的导数"""
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], requires_grad=True,dtype=torch.float64)
    
    # 获取Swish及其导数函数
    fn, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives('swish')
    # fn, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives('tanh')
    # fn, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives('sigmoid')
    # fn, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives('relu')
    # fn, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives('cube')
    # fn, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives('square')
    print("\n=== 测试 Swish 激活函数 ===")
    print("函数值:", fn(x))
    print("一阶导:", fn_prime(x))
    print("二阶导:", fn_double_prime(x))
    print("三阶导:", fn_triple_prime(x))
    
    # 与数值微分对比验证
    from torch.autograd import grad
    def autograd_derivative(f, x, order=1):
        derivatives = []
        current_grad = f(x)
        derivatives.append(current_grad)
        
        for i in range(1, order + 1):
            current_grad = grad(current_grad.sum(), x, create_graph=True)[0]
            derivatives.append(current_grad)
        
        return derivatives[-1]
        
    
    print("\n微分对比:")
    print("一阶导数差异:", torch.abs(fn_prime(x) - autograd_derivative(fn, x, 1)).max().item())
    print("二阶导数差异:", torch.abs(fn_double_prime(x) - autograd_derivative(fn, x, 2)).max().item())
    print("三阶导数差异:", torch.abs(fn_triple_prime(x) - autograd_derivative(fn, x, 3)).max().item())

def test_custom_activation():
    """测试自定义激活函数的导数计算"""
    x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]], requires_grad=True, dtype=torch.float64)
    
    def custom_fn(x): 
        return torch.sin(x) * x**2
    
    fn, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives(custom_fn)
    
    print("\n=== 测试自定义函数 sin(x)*x² ===")
    print("函数值:", fn(x))
    print("一阶导:", fn_prime(x))
    print("二阶导:", fn_double_prime(x))
    print("三阶导:", fn_triple_prime(x))
    
    # 与自动微分对比验证
    from torch.autograd import grad
    def autograd_derivative(f, x, order=1):
        derivatives = []
        current_grad = f(x)
        derivatives.append(current_grad)
        
        for i in range(1, order + 1):
            current_grad = grad(current_grad.sum(), x, create_graph=True)[0]
            derivatives.append(current_grad)
        
        return derivatives[-1]
    
    print("\n自动微分对比:")
    print("一阶导数差异:", torch.abs(fn_prime(x) - autograd_derivative(fn, x, 1)).max().item())
    print("二阶导数差异:", torch.abs(fn_double_prime(x) - autograd_derivative(fn, x, 2)).max().item())
    print("三阶导数差异:", torch.abs(fn_triple_prime(x) - autograd_derivative(fn, x, 3)).max().item())

    #  数值微分
    def numerical_derivative(f, x, order=1, eps=1e-4):
        if order == 1:
            return (f(x + eps) - f(x - eps)) / (2 * eps)
        else:
            def wrapped(x): 
                return numerical_derivative(f, x, order-1, eps)
            return (wrapped(x + eps) - wrapped(x - eps)) / (2 * eps)
    
    print("\n数值微分对比:")
    print("一阶导数差异:", torch.abs(fn_prime(x) - numerical_derivative(fn, x, 1)).max().item())
    print("二阶导数差异:", torch.abs(fn_double_prime(x) - numerical_derivative(fn, x, 2)).max().item())
    print("三阶导数差异:", torch.abs(fn_triple_prime(x) - numerical_derivative(fn, x, 3)).max().item())

    def swish(x):
        return x * torch.sigmoid(x)
    _, ground_truth_prime, ground_truth_double_prime, ground_truth_triple_prime = get_activation_with_derivatives(swish)
    _, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives(swish)
    print("\nswish 激活函数真解对比:")
    print("一阶导数差异:", torch.abs(fn_prime(x) - ground_truth_prime(x)).max().item())
    print("二阶导数差异:", torch.abs(fn_double_prime(x) - ground_truth_double_prime(x)).max().item())
    print("三阶导数差异:", torch.abs(fn_triple_prime(x) - ground_truth_triple_prime(x)).max().item())


def test_elementwise_activation():
    """
    测试激活函数的Taylor展开。
    使用简单的立方函数 f(x) = x^3 作为激活函数。
    
    对于 f: R^n → R^n, f(x) = x^3：
    - Jacobian: J_{ij} = 3x_i^2 if i=j else 0
    - Hessian: H_{ijk} = 6x_i if i=j=k else 0
    - Third derivative: 6 if i=j=k else 0
    """
    x = torch.tensor([[2.0, -1.0], [1.0, 2.0]], requires_grad=True)  # (2,2)
    
    def cube_fn(x): return x**3
    def cube_prime(x): return 3*x**2
    def cube_double_prime(x): return 6*x
    def cube_triple_prime(x): return 6*torch.ones_like(x)
    
    from torch.func import jacrev, vmap
    
    # 计算 Jacobian (first derivative)
    def f(x_in): 
        return cube_fn(x_in).squeeze(0)
    jac = vmap(jacrev(f), (0,))(x)  # shape: (2,2,1,2)
    
    hess = vmap(jacrev(jacrev(f)), (0,))(x)

    third = vmap(jacrev(jacrev(jacrev(f))), (0,))(x)
    
    # 2. TaylorVar
    val_init = x
    first_init = torch.zeros((1, 2, 2))
    first_init[:,0,0] = 1.0
    first_init[:,1,1] = 1.0
    
    x_tvar = TaylorVar(val_init, first_init)
    f_tvar = x_tvar.elementwise_fn(cube_fn, cube_prime, 
                                  cube_double_prime, cube_triple_prime)
    
    print("\n=== 测试激活函数 f(x) = x^3 ===")
    print("在点 x =", x[0].tolist())
    
    print("\nJacobian 对比:")
    print("Autograd functional:")
    print(jac)  # 去掉batch维度
    print("TaylorVar:")
    print(f_tvar.first[...])
    
    print("\nHessian 对比:")
    print("Autograd functional:")
    print(hess)
    print("TaylorVar:")
    print(f_tvar.second[...])
    
    print("\n三阶导数对比:")
    print("Autograd functional:")
    print(third)
    print("TaylorVar:")
    print(f_tvar.third[...])

def test_linear_layer():
    """
    测试线性层的Taylor展开。
    对于线性变换 f(x) = Wx + b：
    - Jacobian: J = W
    - Hessian: H = 0
    - Third derivative: 0
    """
    # 构造输入和线性层参数
    x = torch.tensor([[2.0, -1.0], [1.0, 2.0]], requires_grad=True)  # (2,2)
    W = torch.tensor([[2.0, 1.0],
                     [1.0, 3.0]])  # (2,2)
    b = torch.tensor([0.5, -1.0])  # (2,)
    
    from torch.func import jacrev, vmap
    
    # 1. PyTorch autograd.functional
    def f(x_in):
        return (x_in @ W.T + b).squeeze(0)
    
    # 计算 Jacobian (应该等于 W)
    jac = vmap(jacrev(f), (0,))(x)
    
    # 计算 Hessian (应该全为0)
    hess = vmap(jacrev(jacrev(f)), (0,))(x)
    
    # 计算三阶导数 (应该全为0)
    third = vmap(jacrev(jacrev(jacrev(f))), (0,))(x)
    
    # 2. TaylorVar
    val_init = x
    first_init = torch.zeros((2, 2, 2))  # (batch,d,d)
    first_init[:,0,0] = 1.0  # ∂/∂x1
    first_init[:,1,1] = 1.0  # ∂/∂x2
    
    x_tvar = TaylorVar(val_init, first_init)
    f_tvar = x_tvar.linear(W, b)
    
    print("\n=== 测试线性层 f(x) = Wx + b ===")
    print("W =\n", W)
    print("b =", b.tolist())
    print("在点 x =", x.tolist())
    
    print("\nJacobian 对比:")
    print("Autograd functional:")
    print(jac)
    print("TaylorVar:")
    print(f_tvar.first[...])  # 调整维度顺序以匹配
    
    print("\nHessian 对比:")
    print("Autograd functional:")
    print(hess)
    print("TaylorVar:")
    print(f_tvar.second[...])
    
    print("\n三阶导数对比:")
    print("Autograd functional:")
    print(third)
    print("TaylorVar:")
    print(f_tvar.third[...])

def test_linear_activation_composition():
    """
    测试线性层和激活函数的组合。
    f(x) = cube(Wx + b)，其中 cube(x) = x^3
    
    对于这个复合函数：
    1. 线性层 g(x) = Wx + b 的导数:
       - Jacobian: J_g = W
       - Hessian: H_g = 0
       - Third: 0
       
    2. 立方函数 h(y) = y^3 的导数:
       - h'(y) = 3y^2
       - h''(y) = 6y
       - h'''(y) = 6
       
    3. 复合函数 f = h∘g 的导数通过链式法则计算
    """
    # 构造输入和参数
    x = torch.tensor([[2.0, -1.0], [1.0, 2.0]], requires_grad=True)  # (2,2)
    W = torch.tensor([[2.0, 1.0],
                     [1.0, 3.0]])  # (2,2)
    b = torch.tensor([0.5, -1.0])  # (2,)
    
    from torch.func import jacrev, vmap
    
    # 1. PyTorch autograd.functional
    def f(x_in):
        y = x_in @ W.T + b
        return y**4
    
    # 计算各阶导数
    jac = vmap(jacrev(f), (0,))(x)
    hess = vmap(jacrev(jacrev(f)), (0,))(x)
    third = vmap(jacrev(jacrev(jacrev(f))), (0,))(x)
    
    # 2. TaylorVar
    val_init = x
    first_init = torch.zeros((2, 2, 2))  # (batch,d,d)
    first_init[:,0,0] = 1.0  # ∂/∂x1
    first_init[:,1,1] = 1.0  # ∂/∂x2

    x_tvar = TaylorVar(val_init, first_init)
    
    # 先线性层
    y_tvar = x_tvar.linear(W, b)
    
    # 再过激活函数
    def cube_fn(x): return x**4
    def cube_prime(x): return 4*x**3
    def cube_double_prime(x): return 12*x**2
    def cube_triple_prime(x): return 24*x
    
    z_tvar = y_tvar.elementwise_fn(cube_fn, cube_prime, 
                                  cube_double_prime, cube_triple_prime)
    
    print("\n=== 测试复合函数 f(x) = cube(Wx + b) ===")
    print("W =\n", W)
    print("b =", b.tolist())
    print("在点 x =\n", x.tolist())
    
    print("\nJacobian 对比:")
    print("Autograd functional:")
    print(jac)
    print("TaylorVar:")
    print(z_tvar.first[...])
    
    print("\nHessian 对比:")
    print("Autograd functional:")
    print(hess)
    print("TaylorVar:")
    print(z_tvar.second[...])
    
    print("\n三阶导数对比:")
    print("Autograd functional:")
    print(third)
    print("TaylorVar:")
    print(z_tvar.third[...])

def test_linear_multiply_composition():
    """
    测试线性层和乘法的组合。
    f(x) = (A₁x + b₁)(A₂x + b₂)
    
    对于这个复合函数：
    1. 两个线性变换:
       g₁(x) = A₁x + b₁
       g₂(x) = A₂x + b₂
       
    2. 然后做逐元素乘法:
       f(x) = g₁(x) * g₂(x)
    """
    # 构造输入和参数
    x = torch.tensor([[2.0, -1.0], [1.0, 2.0]], requires_grad=True)  # (2,2)
    
    # 第一个线性层参数
    W1 = torch.tensor([[2.0, 1.0],
                      [1.0, 3.0]])  # (2,2)
    b1 = torch.tensor([0.5, -1.0])  # (2,)
    
    # 第二个线性层参数
    W2 = torch.tensor([[1.0, -1.0],
                      [2.0, 1.0]])  # (2,2)
    b2 = torch.tensor([-0.5, 1.0])  # (2,)
    
    from torch.func import jacrev, vmap
    
    # 1. PyTorch autograd.functional
    def f(x_in):
        g1 = x_in @ W1.T + b1
        g2 = x_in @ W2.T + b2
        return g1 * g2
    
    # 计算各阶导数
    jac = vmap(jacrev(f), (0,))(x)
    hess = vmap(jacrev(jacrev(f)), (0,))(x)
    third = vmap(jacrev(jacrev(jacrev(f))), (0,))(x)
    
    # 2. TaylorVar
    val_init = x
    first_init = torch.zeros((2, 2, 2))  # (batch,d,d)
    first_init[:,0,0] = 1.0  # ∂/∂x1
    first_init[:,1,1] = 1.0  # ∂/∂x2

    
    x_tvar = TaylorVar(val_init, first_init)
    
    # 计算两个线性变换
    g1_tvar = x_tvar.linear(W1, b1)
    g2_tvar = x_tvar.linear(W2, b2)
    
    # 做逐元素乘法
    f_tvar = g1_tvar * g2_tvar
    
    print("\n=== 测试复合函数 f(x) = (A₁x + b₁)(A₂x + b₂) ===")
    print("W1 =\n", W1)
    print("b1 =", b1.tolist())
    print("W2 =\n", W2)
    print("b2 =", b2.tolist())
    print("在点 x =\n", x.tolist())
    
    print("\nJacobian 对比:")
    print("Autograd functional:")
    print(jac)
    print("TaylorVar:")
    print(f_tvar.first[...])
    
    print("\nHessian 对比:")
    print("Autograd functional:")
    print(hess)
    print("TaylorVar:")
    print(f_tvar.second[...])
    
    print("\n三阶导数对比:")
    print("Autograd functional:")
    print(third)
    print("TaylorVar:")
    print(f_tvar.third[...])

def test_linear_activation_multiply():
    """
    测试激活函数、线性层和乘法的组合。
    f(x) = φ(A₁x + b₁) * φ(A₂x + b₂)，其中 φ(x) = x⁴
    
    复合过程：
    1. 两个线性变换:
       g₁(x) = A₁x + b₁
       g₂(x) = A₂x + b₂
       
    2. 分别过激活函数:
       h₁(x) = φ(g₁(x))
       h₂(x) = φ(g₂(x))
       
    3. 最后做逐元素乘法:
       f(x) = h₁(x) * h₂(x)
    """
    # 构造输入和参数
    x = torch.tensor([[2.0, -1.0], [1.0, 2.0]], requires_grad=True)  # (2,2)
    
    # 第一个线性层参数
    W1 = torch.tensor([[2.0, 1.0],
                      [1.0, 3.0]])  # (2,2)
    b1 = torch.tensor([0.5, -1.0])  # (2,)
    
    # 第二个线性层参数
    W2 = torch.tensor([[1.0, -1.0],
                      [2.0, 1.0]])  # (2,2)
    b2 = torch.tensor([-0.5, 1.0])  # (2,)
    
    from torch.func import jacrev, vmap
    
    # 1. PyTorch autograd.functional
    def f(x_in):
        g1 = x_in @ W1.T + b1
        g2 = x_in @ W2.T + b2
        return (g1**4) * (g2**4)
    
    # 计算各阶导数
    jac = vmap(jacrev(f), (0,))(x)
    hess = vmap(jacrev(jacrev(f)), (0,))(x)
    third = vmap(jacrev(jacrev(jacrev(f))), (0,))(x)
    
    # 2. TaylorVar
    val_init = x
    first_init = torch.zeros((2, 2, 2))  # (batch,d,d)
    first_init[:,0,0] = 1.0  # ∂/∂x1
    first_init[:,1,1] = 1.0  # ∂/∂x2
    
    x_tvar = TaylorVar(val_init, first_init)
    
    # 计算两个分支
    def phi(x): return x**4
    def phi_prime(x): return 4*x**3
    def phi_double_prime(x): return 12*x**2
    def phi_triple_prime(x): return 24*x
    
    # 第一个分支: h₁(x) = φ(A₁x + b₁)
    g1_tvar = x_tvar.linear(W1, b1)
    h1_tvar = g1_tvar.elementwise_fn(phi, phi_prime, 
                                    phi_double_prime, phi_triple_prime)
    
    # 第二个分支: h₂(x) = φ(A₂x + b₂)
    g2_tvar = x_tvar.linear(W2, b2)
    h2_tvar = g2_tvar.elementwise_fn(phi, phi_prime, 
                                    phi_double_prime, phi_triple_prime)
    
    # 最后做乘法: f(x) = h₁(x) * h₂(x)
    f_tvar = h1_tvar * h2_tvar
    
    print("\n=== 测试复合函数 f(x) = φ(A₁x + b₁) * φ(A₂x + b₂) ===")
    print("W1 =\n", W1)
    print("b1 =", b1.tolist())
    print("W2 =\n", W2)
    print("b2 =", b2.tolist())
    print("在点 x =\n", x.tolist())
    
    print("\nJacobian 对比:")
    print("Autograd functional:")
    print(jac)
    print("TaylorVar:")
    print(f_tvar.first[...])
    
    print("\nHessian 对比:")
    print("Autograd functional:")
    print(hess)
    print("TaylorVar:")
    print(f_tvar.second[...])
    
    print("\n三阶导数对比 ([..., 1,1,0]):")
    print("Autograd functional:")
    print(third[...,1,1,0])
    print("TaylorVar:")
    print(f_tvar.third[1,1,0])




def test_simple_example():
    """
    测试原始示例函数 f(x₁,x₂) = (x₁² + 3x₂)³
    
    计算过程可以分解为：
    1. g₁(x) = x₁²     (用激活函数)
    2. g₂(x) = 3x₂     (用乘法)
    3. h(x) = g₁ + g₂  (用加法)
    4. f(x) = h³       (用激活函数)
    """
    x = torch.tensor([1.5, -2.0], requires_grad=True)  # (2)
    
    from torch.func import jacrev, vmap
    
    # 1. PyTorch autograd.functional
    def f(x_in):
        return (x_in[0]**2 + 3*x_in[1])**3
    
    # 计算各阶导数
    jac = jacrev(f)(x)
    hess = jacrev(jacrev(f))(x)
    third = jacrev(jacrev(jacrev(f)))(x)
    
    # 2. TaylorVar
    val_init = x
    first_init = torch.zeros((2, 2))  # (d,d)
    first_init[0,0] = 1.0  # ∂/∂x₁
    first_init[1,1] = 1.0  # ∂/∂x₂

    
    x_tvar = TaylorVar(val_init, first_init)
    
    # 计算 x₁²
    def square_fn(x): return x**2
    def square_prime(x): return 2*x
    def square_double_prime(x): return 2*torch.ones_like(x)
    def square_triple_prime(x): return torch.zeros_like(x)
    
    x1_tvar = x_tvar[0]
    g1_tvar = x1_tvar.elementwise_fn(square_fn, square_prime,
                                    square_double_prime, square_triple_prime)
    
    # 计算 3x₂
    x2_tvar = x_tvar[1]
    g2_tvar = 3 * x2_tvar
    
    # 计算 g₁ + g₂
    h_tvar = g1_tvar + g2_tvar
    
    # 计算 h³
    def cube_fn(x): return x**3
    def cube_prime(x): return 3*x**2
    def cube_double_prime(x): return 6*x
    def cube_triple_prime(x): return 6*torch.ones_like(x)
    
    f_tvar = h_tvar.elementwise_fn(cube_fn, cube_prime,
                                  cube_double_prime, cube_triple_prime)
    
    print("\n=== 测试原始示例 f(x₁,x₂) = (x₁² + 3x₂)³ ===")
    print("在点 x =", x[0].tolist())
    
    print("\nJacobian 对比:")
    print("Autograd functional:")
    print(jac)
    print("TaylorVar:")
    print(f_tvar.first[...])
    
    print("\nHessian 对比:")
    print("Autograd functional:")
    print(hess)
    print("TaylorVar:")
    print(f_tvar.second[...])
    
    print("\n三阶导数对比:")
    print("Autograd functional:")
    print(third)
    print("TaylorVar:")
    print(f_tvar.third[...])

def test_other_components():
    """
    测试新添加的组件：
    1. 减法
    2. 形状操作 (reshape, view, flatten)
    3. 张量组合 (cat, stack)
    4. 维度操作 (squeeze, unsqueeze)
    """
    # 构造测试数据
    x = torch.tensor([[2.0, -1.0], [1.0, 2.0]], requires_grad=True)  # (2,2)
    val_init = x
    first_init = torch.zeros((2, 2, 2))  # (batch,d,d)
    first_init[:,0,0] = 1.0  # ∂/∂x1
    first_init[:,1,1] = 1.0  # ∂/∂x2

    
    x_tvar = TaylorVar(val_init, first_init)
    
    # 1. 测试减法
    print("\n=== 测试减法 ===")
    def f_sub(x): return x - torch.tensor([1.0, 0.5])
    y_sub = f_sub(x)
    from torch.func import jacrev, vmap
    jac_sub = vmap(jacrev(f_sub), (0,))(x)
    
    y_tvar = f_sub(x_tvar)
    print("减法结果对比:")
    print("Autograd:", y_sub)
    print("TaylorVar:", y_tvar.val)
    print("Jacobian对比:")
    print("Autograd:", jac_sub)
    print("TaylorVar:", y_tvar.first[...])
    
    # 2. 测试形状操作
    print("\n=== 测试形状操作 ===")
    # reshape
    reshaped_tvar = x_tvar.reshape(4)
    print("Reshape后的形状:")
    print("val:", reshaped_tvar.val.shape)
    print("first:", reshaped_tvar.first[...].shape)
    print("second:", reshaped_tvar.second[...].shape)
    print("third:", reshaped_tvar.third[...].shape)

    flatten_tvar = x_tvar.flatten()
    print("Flatten后的形状:")
    print("val:", flatten_tvar.val.shape)
    print("first:", flatten_tvar.first[...].shape)
    print("second:", flatten_tvar.second[...].shape)
    print("third:", flatten_tvar.third[...].shape)
    
    # 3. 测试张量组合
    print("\n=== 测试张量组合 ===")
    # cat
    cat_tvar = TaylorVar.cat([x_tvar, x_tvar], dim=1)
    print("Cat后的形状:")
    print("val:", cat_tvar.val.shape)
    print("first:", cat_tvar.first[...].shape)
    print("second:", cat_tvar.second[...].shape)
    print("third:", cat_tvar.third[...].shape)
    
    # stack
    stack_tvar = TaylorVar.stack([x_tvar, x_tvar], dim=0)
    print("Stack后的形状:")
    print("val:", stack_tvar.val.shape)
    print("first:", stack_tvar.first[...].shape)
    print("second:", stack_tvar.second[...].shape)
    print("third:", stack_tvar.third[...].shape)
    
    # 4. 测试维度操作
    print("\n=== 测试维度操作 ===")
    # unsqueeze
    unsqueezed_tvar = x_tvar.unsqueeze(1)
    print("Unsqueeze后的形状:")
    print("val:", unsqueezed_tvar.val.shape)
    print("first:", unsqueezed_tvar.first[...].shape)
    print("second:", unsqueezed_tvar.second[...].shape)
    print("third:", unsqueezed_tvar.third[...].shape)

def test_functional_compatibility():
    """
    测试 TaylorVar 与 PyTorch functional API 的兼容性
    主要测试:
    1. vmap 对 TaylorVar 的批处理
    2. jacrev 与 TaylorVar 的交互
    3. 组合使用的情况
    """
    from torch.func import vmap, jacrev
    
    # 构造测试数据
    batch_size = 3
    x = torch.tensor([[1.0, -1.0], [2.0, 0.0], [0.0, 2.0]], requires_grad=True)  # (3,2)
    
    # swish 激活函数
    activation_fn = taylor_activation_wrapper(*get_activation_with_derivatives("swish"))

    # 1. 测试基本的 vmap 兼容性
    print("\n=== 测试 vmap 兼容性 ===")
    
    def forward_fn(x_single):
        """单样本的前向函数"""
        # 构造 TaylorVar
        val_init = x_single # (2,)
        first_init = torch.zeros((2, 2))
        first_init[0,0] = 1.0
        first_init[1,1] = 1.0
        x_tvar = TaylorVar(val_init, first_init)
        
        y_tvar = x_tvar.linear(torch.tensor([[2.0, 1.0], [1.0, 3.0]]), 
                              torch.tensor([0.5, -1.0]))
        z_tvar = activation_fn(y_tvar)
        
        return z_tvar.val
    
    # 使用 vmap 批处理
    batched_forward = vmap(forward_fn)
    result = batched_forward(x)
    print("vmap 批处理结果:", result.shape)
    
    # 2. 测试 vmap 内自动微分
    print("\n=== 测试 vmap 内外Taylor mode自动微分 对比 ===")
    
    # swish 激活函数
    activation_fn = taylor_activation_wrapper(*get_activation_with_derivatives("swish"))
    def batched_forward_jac(x):
        """批量前向函数"""
        val_init = x
        first_init = torch.zeros(( x.shape[0], 2, 2))
        first_init[:,0,0] = 1.0
        first_init[:,1,1] = 1.0
        x_tvar = TaylorVar(val_init, first_init)
        
        y_tvar = x_tvar.linear(torch.tensor([[2.0, 1.0], [1.0, 3.0]]), 
                              torch.tensor([0.5, -1.0]))
        z_tvar = activation_fn(y_tvar)
        return z_tvar.first[...]
    def forward_fn_jac(x_single):
        """单样本前向函数"""
        val_init = x_single
        first_init = torch.zeros((2, 2))
        first_init[0,0] = 1.0
        first_init[1,1] = 1.0
        x_tvar = TaylorVar(val_init, first_init)
        
        y_tvar = x_tvar.linear(torch.tensor([[2.0, 1.0], [1.0, 3.0]]), 
                              torch.tensor([0.5, -1.0]))
        z_tvar = activation_fn(y_tvar)
        return z_tvar.first[...]
        
    
    
    
    vm_jac = vmap(forward_fn_jac)(x)
    print("vmap 内自动微分输出形状:", vm_jac.shape)
    print("TaylorVar first 形状:", batched_forward_jac(x).shape)
    
    # 3. 测试组合使用
    print("\n=== 测试组合使用 ===")
    
    def combined_fn(x_single):
        """结合 TaylorVar 和 jacrev"""
        x_tvar = TaylorVar(x_single.unsqueeze(0))
        y_tvar = x_tvar.linear(torch.tensor([[2.0, 1.0], [1.0, 3.0]]))
        return y_tvar.val.squeeze(0)
    
    # 先 vmap 再 jacrev
    batched_jac = vmap(jacrev(combined_fn))(x)
    print("Batched Jacobian 形状:", batched_jac.shape)
    
    # 4. 测试与激活函数的兼容性
    print("\n=== 测试与激活函数的兼容性 ===")
    
    def activation_fn(x_single):
        x_tvar = TaylorVar(x_single.unsqueeze(0))
        fn, fn_prime, fn_double_prime, fn_triple_prime = get_activation_with_derivatives('swish')
        y_tvar = x_tvar.elementwise_fn(fn, fn_prime, fn_double_prime, fn_triple_prime)
        return y_tvar.val.squeeze(0)
    
    # 计算批量 Jacobian
    batched_act_jac = vmap(jacrev(activation_fn))(x)
    print("Batched Activation Jacobian 形状:", batched_act_jac.shape)
    
    # 5. 对比结果
    print("\n=== 结果对比 ===")
    
    def standard_fn(x):
        """标准 PyTorch 函数用于对比"""
        y = x @ torch.tensor([[2.0, 1.0], [1.0, 3.0]]).T
        return y * y
    
    standard_jac = vmap(jacrev(standard_fn))(x)
    first_init = torch.zeros(( x.shape[0], 2, 2))
    first_init[:,0,0] = 1.0
    first_init[:,1,1] = 1.0
    taylor_x = TaylorVar(x, first_init)
    y = taylor_x.linear(torch.tensor([[2.0, 1.0], [1.0, 3.0]]))
    y= y*y
    taylor_jac = y.first[...]
    
    print("标准 Jacobian 与 TaylorVar 结果的最大差异:",
          torch.abs(standard_jac - taylor_jac).max().item())

def test_modified_fourier_net():
    """
    计算流程：
    1. 输入变换: x -> (x-shift_t)*scale
    2. 特征扩充: x -> [x, sin(xB1), cos(xB2)]
    3. 双分支计算:
       - U = act(linear1(x))
       - V = act(linear2(x))
    4. 中间层循环:
       for linear in linears:
           out = sigmoid(linear(x))
           x = out*U + (1-out)*V
    5. 最后两层:
       x = act(linear3(x))
       x = head(x)
    6. 重塑输出: x -> x.reshape(..., 2, -1)
    """
    import torch
    from torch.func import vmap, jacrev
    import time

    # 构造测试数据
    batch_size = 1000
    in_dim = 3
    h_dim = 50
    out_dim = 2
    num_freq = 5
    dtype = torch.float64
    
    x = torch.randn(batch_size, in_dim, dtype=dtype, requires_grad=True)
    
    # 构造网络参数
    B1 = torch.randn(num_freq, in_dim, dtype=dtype)
    B2 = torch.randn(num_freq, in_dim, dtype=dtype)
    t0 = torch.tensor(0.0, dtype=dtype)
    scale = torch.tensor([1.0, 1.0, 2.0], dtype=dtype)
    shift_t = torch.tensor([0.0, 0.0, t0], dtype=dtype)
    
    # 构造线性层参数
    aug_dim = in_dim + 2*num_freq
    W1 = torch.randn(h_dim, aug_dim, dtype=dtype)
    b1 = torch.randn(h_dim, dtype=dtype)
    W2 = torch.randn(h_dim, aug_dim, dtype=dtype)
    b2 = torch.randn(h_dim, dtype=dtype)
    W3 = torch.randn(h_dim, h_dim, dtype=dtype)
    b3 = torch.randn(h_dim, dtype=dtype)
    W4 = torch.randn(h_dim, h_dim, dtype=dtype)
    b4 = torch.randn(h_dim, dtype=dtype)
    W5 = torch.randn(h_dim, h_dim, dtype=dtype)
    b5 = torch.randn(h_dim, dtype=dtype)
    Wh = torch.randn(out_dim, h_dim, dtype=dtype)
    
    # 1. PyTorch autograd 版本
    def forward_fn(x_in):
        # 输入变换
        x = (x_in - shift_t) * scale
        
        # 特征扩充
        x_aug = torch.cat([x, 
                          torch.sin(x @ B1.T), 
                          torch.cos(x @ B2.T)], dim=-1)
        
        # # 双分支
        U = torch.tanh(x_aug @ W1.T + b1)
        V = torch.tanh(x_aug @ W2.T + b2)
        # return U
        # # 中间层
        out = torch.sigmoid(x_aug @ W1.T + b1)
        x = out * U + (1-out) * V
        # return (1-out)*V
        # # 最后两层
        x = torch.tanh(x @ W3.T + b3)
        x = x @ Wh.T
        return x
    
    # 2. TaylorVar 版本
    def taylor_forward(x_batch):
        # 构造输入 TaylorVar
        first_init = torch.zeros(x_batch.shape + (in_dim,))
        for i in range(in_dim):
            first_init[...,i,i] = 1.0
        x_tvar = TaylorVar(x_batch, first_init)
        
        # 输入变换
        # breakpoint()
        # x_tvar1 = (x_tvar - shift_t)
        x_tvar = (x_tvar - shift_t) * scale
        
        # x_tvar.third[0,1,2]
        # 特征扩充
        sin_B1 = taylor_activation_wrapper(*get_activation_with_derivatives('sin'))
        cos_B2 = taylor_activation_wrapper(*get_activation_with_derivatives('cos'))
        
        x_B1 = x_tvar.linear(B1)
        x_B2 = x_tvar.linear(B2)
        sin_x_B1 = sin_B1(x_B1)
        cos_x_B2 = cos_B2(x_B2)
        
        x_aug = TaylorVar.cat([x_tvar, sin_x_B1, cos_x_B2], dim=-1)
        
        
        # # 双分支
        tanh = taylor_activation_wrapper(*get_activation_with_derivatives('tanh'))
        sigmoid = taylor_activation_wrapper(*get_activation_with_derivatives('sigmoid'))
        
        U = tanh(x_aug.linear(W1, b1))
        V = tanh(x_aug.linear(W2, b2))
        
        # # 中间层
        out = sigmoid(x_aug.linear(W1, b1))
        
        x = out * U + (1-out) * V
        
        # 最后两层
        x = tanh(x.linear(W3, b3))
        x = x.linear(Wh)
        
        return x
    
    # 计算并比较结果
    print("\n=== 测试 modified_fourier_net ===")
    
    print(f"batch_size: {batch_size}, in_dim: {in_dim}, h_dim: {h_dim}, out_dim: {out_dim}, num_freq: {num_freq}")

    # # 1. 前向传播对比
    # y_autograd = forward_fn(x)
    # y_taylor = taylor_forward(x)
    # print("\n前向传播最大差异:", 
    #       torch.abs(y_autograd - y_taylor.val).max().item())
    
    # 2. Jacobian 对比
    time_start = time.time()
    for i in range(50):
        jac_taylor = taylor_forward(x).first[...]
    time_end = time.time()
    print(f"Jacobian taylor 计算时间: {(time_end - time_start)/50} 秒")
    time_start = time.time()
    for i in range(50):
        jac_autograd = vmap(jacrev(forward_fn))(x)
    time_end = time.time()
    print(f"Jacobian autograd 计算时间: {(time_end - time_start)/50} 秒")

    
    
    print("\nJacobian 最大差异:", 
          torch.abs(jac_autograd - jac_taylor).max().item())
    # 索引后导数对比
    out = taylor_forward(x)
    y = out[:,0]+out[:,1]
    print(y.shape)
    # print(y.third[0,1,1:2])
    temp = out.third[0,1,1:2]
    # print(temp[:,0]+temp[:,1])
    # 3. Hessian 对比
    time_start = time.time() 
    for i in range(50):
        hess_autograd = vmap(jacrev(jacrev(forward_fn)))(x)
    time_end = time.time()
    print(f"Hessian autograd 计算时间: {(time_end - time_start)/50} 秒")
    
    time_start = time.time()
    for i in range(50):
        hess_taylor = taylor_forward(x).second[...]
    time_end = time.time()
    print(f"Hessian taylor 计算时间: {(time_end - time_start)/50} 秒")
    
    print("Hessian 最大差异:", 
          torch.abs(hess_autograd - hess_taylor).max().item())
    # 4. 三阶导数对比
    time_start = time.time()
    for i in range(50):
        third_autograd = vmap(jacrev(jacrev(jacrev(forward_fn))))(x)
    time_end = time.time()
    print(f"三阶导数 autograd 计算时间: {(time_end - time_start)/50} 秒")
    
    time_start = time.time()
    for i in range(50):
        third_taylor = taylor_forward(x).third[...]
    time_end = time.time()
    print(f"全量三阶导数 taylor 计算时间: {(time_end - time_start)/50} 秒")

    time_start = time.time()
    third_taylor_012 = taylor_forward(x).third[0,1,2]
    time_end = time.time()
    print(f" third[0,1,2] 局部计算 taylor 计算时间: {time_end - time_start} 秒")
    
    print("third[0,1,2] 局部计算和全量计算最大差异:", 
          torch.abs(third_taylor[..., 0,1,2] - third_taylor_012).max().item())
    second_taylor_12 = taylor_forward(x).second[1,2]
    print("second[1,2] 局部计算和全量计算最大差异:", 
          torch.abs(second_taylor_12 - hess_taylor[..., 1,2]).max().item())
    first_taylor_0 = taylor_forward(x).first[0]
    print("first[3] 局部计算和全量计算最大差异:", 
          torch.abs(first_taylor_0 - jac_taylor[..., 0]).max().item())
    # breakpoint()
    print("\n三阶导数最大差异:", 
          torch.abs(third_autograd - third_taylor).max().item())
    
    # # # 5. vmap 内计算taylor 测试
    # def single_forward(x_single):
    #     return taylor_forward(x_single).val
    # def single_forward_jac(x_single):
    #     return taylor_forward(x_single).first
    # def single_forward_hess(x_single):
    #     return taylor_forward(x_single).second
    # def single_forward_third(x_single):
    #     return taylor_forward(x_single).third
    # y_vmap = vmap(single_forward)(x)
    # y_vmap_jac = vmap(single_forward_jac)(x)
    # y_vmap_hess = vmap(single_forward_hess)(x)
    # y_vmap_third = vmap(single_forward_third)(x)
    # print("\nvmap 批处理最大差异:", 
    #       torch.abs(y_vmap - y_taylor.val).max().item())
    # print("\nvmap 批处理 Jacobian 最大差异:", 
    #       torch.abs(y_vmap_jac - y_taylor.first).max().item())
    # print("\nvmap 批处理 Hessian 最大差异:", 
    #       torch.abs(y_vmap_hess - y_taylor.second).max().item())
    # print("\nvmap 批处理 三阶导数 最大差异:", 
    #       torch.abs(y_vmap_third - y_taylor.third).max().item())


def test_derivative_tensor():
    """
    测试 DerivativeTensor 的基本功能：
    1. 初始化和存储
    2. 全量计算 ([...] 或 [:,:,:])
    3. 单个导数分量访问 ([i,j,k])
    4. 导数分量块访问 ([i1:i2, j1:j2, k1:k2])
    5. 验证计算结果缓存
    """
    # 构造测试数据
    batch_size, d = 2, 3
    x = torch.randn(batch_size, 2)  # (batch_size, feature_dim)
    
    # 模拟计算函数
    compute_count = 0
    def mock_compute(order, parent, idx=None):
        nonlocal compute_count
        compute_count += 1
        if idx is None:  # 全量计算
            if order == 1:
                return torch.ones(batch_size, 2, d)
            elif order == 2:
                return torch.ones(batch_size, 2, d, d) * 2
            else:  # order == 3
                return torch.ones(batch_size, 2, d, d, d) * 3
        else:  # 部分计算
            if order == 1:
                return torch.ones(batch_size, 2) * (idx[0] + 1)
            elif order == 2:
                return torch.ones(batch_size, 2) * (idx[0] + idx[1] + 1)
            else:  # order == 3
                return torch.ones(batch_size, 2) * (idx[0] + idx[1] + idx[2] + 1)
    
    # 1. 测试初始化
    print("\n=== 测试 DerivativeTensor 初始化 ===")
    parent = TaylorVar(x, compute_fn=mock_compute, input_dim=d)
    deriv = DerivativeTensor(parent, order=3)
    print("初始化成功")
    
    # 2. 测试全量计算
    print("\n=== 测试全量计算 ===")
    compute_count = 0
    full_tensor1 = deriv[...]  # 使用 Ellipsis
    print("第一次全量计算调用次数:", compute_count)
    print("full_tensor1.shape:", full_tensor1.shape)
    
    compute_count = 0
    full_tensor2 = deriv[:,:,:]  # 使用 :
    print("第二次全量计算调用次数:", compute_count)  # 应该是0，因为已缓存
    print("full_tensor2.shape:", full_tensor2.shape)
    
    assert torch.all(full_tensor1 == full_tensor2), "两种全量索引方式结果不一致"
    
    # 3. 测试单个导数分量访问
    print("\n=== 测试单个导数分量访问 ===")
    compute_count = 0
    partial = deriv[0,1,2]  # 计算 ∂³f/∂x₀∂x₁∂x₂
    print("单个导数分量计算调用次数:", compute_count)
    print("partial.shape:", partial.shape)
    # 4. 测试导数分量块访问
    print("\n=== 测试导数分量块访问 ===")
    compute_count = 0
    block = deriv[0:2, 1:3, 2:3]  # 计算一个导数块
    print("导数分量块计算调用次数:", compute_count)
    print("block.shape:", block.shape)
    # 5. 测试错误处理
    print("\n=== 测试错误处理 ===")
    try:
        wrong_idx = deriv[0]  # 索引维度不匹配
        print("错误：应该捕获索引维度不匹配错误")
    except AssertionError:
        print("成功捕获索引维度不匹配错误")
    
    try:
        wrong_idx = deriv[0, 1]  # 三阶导数需要3个索引
        print("错误：应该捕获索引数量不匹配错误")
    except AssertionError:
        print("成功捕获索引数量不匹配错误")
    
    # 6. 测试 to_tensor
    print("\n=== 测试 to_tensor ===")
    compute_count = 0
    tensor = deriv.to_tensor()
    print("to_tensor 调用次数:", compute_count)  # 应该是0，因为已经计算过全量
    print("tensor.shape:", tensor.shape)

def test_taylor_multiplication():
    """
    测试 TaylorVar 的乘法操作：
    1. 标量乘法
    2. 两个 TaylorVar 相乘
    3. 测试全量计算和部分计算
    4. 验证形状正确性
    """
    # 构造测试数据
    batch_size, feature_dim, d = 1, 3, 3
    x = torch.randn(batch_size, feature_dim)
    y = torch.randn(batch_size, feature_dim)
    
    # 初始化 TaylorVar
    x_first= torch.zeros(batch_size, feature_dim, d)
    for i in range(batch_size):
        x_first[i]=torch.eye(d)
    y_first= torch.zeros(batch_size, feature_dim, d)
    for i in range(batch_size):
        y_first[i]=torch.eye(d)
    x_var = TaylorVar(x, input_dim=d, first=x_first)
    y_var = TaylorVar(y, input_dim=d, first=y_first)
    
    print("\n=== 测试标量乘法 ===")
    scalar = 2.0
    z_scalar = x_var * scalar
    
    # 测试标量乘法的形状
   
    # print("\n标量乘法 - 部分计算:")
    # first_part = z_scalar.first[1]  # 单个导数
    # second_part = z_scalar.second[0:2, 1:3]  # 导数块
    # third_part = z_scalar.third[0,1,2]  # 单个三阶导数
    # print(f"first[1] shape: {first_part.shape}")  # 应为 (batch_size, feature_dim)
    # print(f"second[0:2,1:3] shape: {second_part.shape}")  # 应为 (batch_size, feature_dim, 2, 2)
    # print(f"third[0,1,2] shape: {third_part.shape}")  # 应为 (batch_size, feature_dim)
    
    # print("标量乘法 - 全量计算:")
    # first_full = z_scalar.first[...]
    # second_full = z_scalar.second[...]
    # third_full = z_scalar.third[...]
    # print(f"first shape: {first_full.shape}")  # 应为 (batch_size, feature_dim, d)
    # print(f"second shape: {second_full.shape}")  # 应为 (batch_size, feature_dim, d, d)
    # print(f"third shape: {third_full.shape}")  # 应为 (batch_size, feature_dim, d, d, d)
    
    print("\n=== 测试 TaylorVar 相乘 ===")
    z_var = x_var * y_var
    
    # 不使用全量计算，因为正确结果会被缓存，无法验证部分计算
    
    print("\nTaylorVar 乘法 - 部分计算:")
    print(z_var.first[...])
    first_part = z_var.first[1]
    second_part = z_var.second[0:2, 1:3]
    third_part = z_var.third[0,1,2]
    print(f"first[1] shape: {first_part.shape}")
    print(f"second[0:2,1:3] shape: {second_part.shape}")
    print(f"third[0,1,2] shape: {third_part.shape}")
    
    # 验证计算正确性
    print("\n=== 验证计算正确性 ===")
    # 验证标量乘法
    assert torch.allclose(z_scalar.val, x_var.val * scalar), "标量乘法值计算错误"
    # assert torch.allclose((x_var.first[...]*scalar)[...,1:2], z_scalar.first[1]), "标量乘法一阶导数计算错误"
    print(z_scalar.first[1])
    print((x_var.first[...]*scalar)[...,1])
    # 验证 TaylorVar 乘法
    assert torch.allclose(z_var.val, x_var.val * y_var.val), "TaylorVar 乘法值计算错误"
    # 验证一阶导数公式
    first_1 = z_var.first[1]
    print(first_1)
    print(z_var.first[1:2])
    expected_first_1 = (x_var.first[1] * y_var.val + x_var.val * y_var.first[1])
    assert torch.allclose(first_1, expected_first_1), "TaylorVar 乘法一阶导数计算错误"
    
    print("所有测试通过!")

def test_taylor_indexing():
    # 创建测试数据
    x = torch.randn(4, 3, 2)
    x_tvar = TaylorVar(x, input_dim=5)
    
    # # 测试基本索引
    # assert torch.allclose(x_tvar[0].val, x[0])
    # print(x_tvar[:, 1].third[1,2,2].shape) #部分计算
    # print(x_tvar[:, 1].third[1:2,1:2,1:2].shape) #部分计算
    # print(x_tvar[:, 1].third[...].shape) #全量计算
    
    # 测试Ellipsis
    assert torch.allclose(x_tvar[...].val, x)
    print(x_tvar[...].third[1,2,2].shape)
    print(x_tvar[...].third[1:2,1:2,1:2].shape)
    print(x_tvar[...,0].third[1,2,2].shape)
    print(x_tvar[0,...,0].third[1:2,1:2,1:2].shape)
    # print(x_tvar[...,0].third[...].shape)
    print(x_tvar[0].third[...].shape)

def test_broadcast_multiplication():
    # 测试不同形状的乘法
    x = torch.randn(3, 4, 5)  # (batch, m, n)
    y = torch.randn(5)        # (n,)
    
    x_tvar = TaylorVar(x, input_dim=2)
    y_tvar = TaylorVar(y, input_dim=2)
    
    # 测试乘法
    z_tvar = x_tvar * y_tvar
    
    # 验证形状
    print("Value shape:", z_tvar.val.shape)  # 应该是 (3, 4, 5)
    print("First derivative shape:", z_tvar.first[...].shape)
    print("Second derivative shape:", z_tvar.second[...].shape)
    print("Third derivative shape:", z_tvar.third[...].shape)
    # 验证与torch的广播结果一致
    assert torch.allclose(z_tvar.val, x * y)
    print(torch.einsum('...i,...j->...ij', torch.randn(3,4,5), torch.randn(5,1,1, 5)).shape)

def test_broadcast_operations():
    # 测试不同形状的加法和乘法
    x = torch.randn(3, 4, 5)  # (batch, m, n)
    y = torch.randn(5)        # (n,)
    
    x_tvar = TaylorVar(x, input_dim=2)
    y_tvar = TaylorVar(y, input_dim=2)
    
    # 测试加法
    z_add = x_tvar + y_tvar
    print("\n=== 测试加法广播 ===")
    print("Value shape:", z_add.val.shape)  # 应该是 (3, 4, 5)
    print("First derivative shape:", z_add.first[...].shape)
    print("Second derivative shape:", z_add.second[...].shape)
    print("Third derivative shape:", z_add.third[...].shape)
    # 验证与torch的广播结果一致
    assert torch.allclose(z_add.val, x + y)
    
    # 测试乘法
    z_mul = x_tvar * y_tvar
    print("\n=== 测试乘法广播 ===")
    print("Value shape:", z_mul.val.shape)  # 应该是 (3, 4, 5)
    print("First derivative shape:", z_mul.first[...].shape)
    print("Second derivative shape:", z_mul.second[...].shape)
    print("Third derivative shape:", z_mul.third[...].shape)
    # 验证与torch的广播结果一致
    assert torch.allclose(z_mul.val, x * y)
    
    # 测试更复杂的广播
    w = torch.randn(1, 4, 1)  # (1, m, 1)
    w_tvar = TaylorVar(w, input_dim=2)
    
    # 三个张量的运算
    result = x_tvar + y_tvar * w_tvar
    print("\n=== 测试复合广播 ===")
    print("Value shape:", result.val.shape)  # 应该是 (3, 4, 5)
    print("First derivative shape:", result.first[...].shape)
    print("Second derivative shape:", result.second[...].shape)
    print("Third derivative shape:", result.third[...].shape)
    # 验证与torch的广播结果一致
    assert torch.allclose(result.val, x + y * w)

from torch import nn
class taylor_modified_fourier_net(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, layers=4, num_freq=10, t0=0, scale_t=1, activation='tanh', dtype=torch.float64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        layer_list=[self.in_dim+2*num_freq]+[h_dim]*layers+[out_dim]
        self.num_freq = num_freq
        self.act = get_activation_with_derivatives(activation)[0]
        self.act_wrapper = taylor_activation_wrapper(*get_activation_with_derivatives(activation))
        self.linears = nn.ModuleList()

        for i in range(1,len(layer_list)-2):
            self.linears.append(nn.Linear(layer_list[i-1],layer_list[i], dtype=dtype))

        self.linear1=nn.Linear(layer_list[0], layer_list[1], dtype=dtype)
        self.linear2=nn.Linear(layer_list[0], layer_list[1], dtype=dtype)
        self.linear3=nn.Linear(layer_list[-3], layer_list[-2], dtype=dtype)
        
        self.head = nn.Linear(layer_list[-2], layer_list[-1], bias=False, dtype=dtype)
        self.register_buffer('B1', torch.randn(num_freq, self.in_dim, dtype=dtype))
        self.register_buffer('B2', torch.randn(num_freq, self.in_dim, dtype=dtype))
        self.register_buffer('t0', torch.tensor(t0, dtype=dtype))
        self.register_buffer('scale', torch.tensor([1.0]*(in_dim-1)+[scale_t], dtype=dtype))
        self.register_buffer('shift_t', torch.tensor([0.0]*(in_dim-1)+[t0], dtype=dtype))
        
        # 获取sigmoid激活函数
        self.sigmoid = taylor_activation_wrapper(*get_activation_with_derivatives('sigmoid'))
        self.sin = taylor_activation_wrapper(*get_activation_with_derivatives('sin'))
        self.cos = taylor_activation_wrapper(*get_activation_with_derivatives('cos'))

    def reset_parameters(self):
        self.linear3.reset_parameters()
        self.head.reset_parameters()

    def forward(self, inputs, compute_taylor=False):
        if not compute_taylor:
            # 原始前向传播逻辑
            inputs = (inputs - self.shift_t) * self.scale
            inputs = torch.cat([inputs, torch.sin(inputs@self.B1.T), torch.cos(inputs@self.B2.T)], dim=-1)
            U = self.act(self.linear1(inputs))
            V = self.act(self.linear2(inputs))
            for linear in self.linears:
                outputs = torch.sigmoid(linear(inputs))
                inputs = outputs*U + (1-outputs)*V
            inputs = self.act(self.linear3(inputs))
            outputs = self.head(inputs)
            return outputs.reshape(*outputs.shape[:-1],2,-1)
        
        # 构造输入的TaylorVar
        first_init = torch.zeros(inputs.shape + (self.in_dim,), dtype=inputs.dtype, device=inputs.device)
        for i in range(self.in_dim):
            first_init[...,i,i] = 1.0
        x_tvar = TaylorVar(inputs, first_init)
        
        x_tvar = (x_tvar - self.shift_t) * self.scale
        
        x_aug = TaylorVar.cat([x_tvar, self.sin(x_tvar.linear(self.B1)), self.cos(x_tvar.linear(self.B2))], dim=-1)
        U = self.act_wrapper(x_aug.linear(self.linear1.weight, self.linear1.bias))
        V = self.act_wrapper(x_aug.linear(self.linear2.weight, self.linear2.bias))
        
        x = x_aug
        for linear in self.linears:
            outputs = self.sigmoid(x.linear(linear.weight, linear.bias))
            x = outputs*U + (1-outputs)*V
        
        x = self.act_wrapper(x.linear(self.linear3.weight, self.linear3.bias))
        x = x.linear(self.head.weight, None)
        return x.reshape(*x.shape[:-1], 2, -1)
    
def test_performance_comparison():
    """
    对比 Taylor mode 和 autograd 在计算偏导数、构造loss和反向传播时的性能
    """
    import time
    import torch
    from torch.func import vmap, jacrev
    import gc
    device = 'cuda'
    # 测试参数
    batch_size = 10000
    in_dim = 3
    h_dim = 50
    out_dim = 2
    num_freq = 5
    num_iterations = 50
    dtype = torch.float64

    from gradient_torch import jacobian, hessian, caches_clear
    def clear_cuda_cache():
        caches_clear()
        gc.collect()
        torch.cuda.empty_cache()
    def get_peak_memory():
        """获取当前GPU峰值内存占用"""
        return torch.cuda.max_memory_allocated() / 1024**2  # 转换为MB
        
    def reset_peak_memory():
        """重置GPU峰值内存统计"""
        torch.cuda.reset_peak_memory_stats()
        
    # 构造测试数据和参数
    x = torch.randn(batch_size, in_dim, dtype=dtype, requires_grad=True, device=device)
    
    model = taylor_modified_fourier_net(in_dim, h_dim, out_dim, num_freq).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 定义前向函数
    def original_forward(x_batch):
        return model.forward(x_batch).reshape(batch_size, -1)  # 直接返回模型输出
    def original_forward_fn(x_single):
        return model.forward(x_single).reshape(-1)  # 直接返回模型输出
    def taylor_forward(x_batch):
        x = model.forward(x_batch, compute_taylor=True)
        return x.reshape(batch_size, -1)
    
    print("\n=== 性能对比测试 ===")
    print(f"batch_size: {batch_size}, in_dim: {in_dim}, h_dim: {h_dim}")
    print(f"测试重复次数: {num_iterations}")
    print(f"dtype: {dtype}")
    
    print("\n1阶求导(含loss.backward()):")
    # 1. Taylor mode方法测试
    taylor_times = []
    taylor_memories = []
    
    for i in range(num_iterations):
        reset_peak_memory()
        # gc.collect()
        # torch.cuda.empty_cache()
        
        start_time = time.time()
        
        # 计算Taylor展开
        result = taylor_forward(x)
        # 构造相同的loss并反向传播
        jac = result.first[...]
        
        loss = (jac[:,0]+jac[:,1]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        taylor_times.append(time.time() - start_time)
        taylor_memories.append(get_peak_memory())
        clear_cuda_cache()


    # 2. Autograd方法测试
    autograd_times = []
    autograd_memories = []
    
    for i in range(num_iterations):
        reset_peak_memory()
        
        start_time = time.time()
        
        # 计算Jacobian
        y = original_forward(x)  # 现在y是模型的输出
        # print(y)
        # 构造loss并反向传播
        
        loss = (jacobian(y, x, i=0)+jacobian(y, x, i=1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        autograd_times.append(time.time() - start_time)
        autograd_memories.append(get_peak_memory())
        clear_cuda_cache()
    
    # 输出统计结果
    print("\nAutograd方法:")
    print(f"平均时间: {sum(autograd_times)/len(autograd_times):.4f} 秒")
    print(f"最小时间: {min(autograd_times):.4f} 秒")
    print(f"最大时间: {max(autograd_times):.4f} 秒")
    print(f"平均显存占用: {sum(autograd_memories)/len(autograd_memories):.1f} MB")
    print(f"最大显存占用: {max(autograd_memories):.1f} MB")

    
    print("\nTaylor mode方法:")
    print(f"平均时间: {sum(taylor_times)/len(taylor_times):.4f} 秒")
    print(f"最小时间: {min(taylor_times):.4f} 秒")
    print(f"最大时间: {max(taylor_times):.4f} 秒")
    print(f"平均显存占用: {sum(taylor_memories)/len(taylor_memories):.1f} MB")
    print(f"最大显存占用: {max(taylor_memories):.1f} MB")

    print("\n2阶求导(Laplace算子, 不含loss.backward()):")
    # 1. Taylor mode方法测试
    taylor_times = []
    taylor_memories = []
    
    for i in range(num_iterations):
        reset_peak_memory()
        # gc.collect()
        # torch.cuda.empty_cache()
        
        start_time = time.time()
        
        # 计算Taylor展开
        result = taylor_forward(x)
        # 构造相同的loss并反向传播
        
        loss = (result.second[0,0]+result.second[1,1]).mean()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        taylor_times.append(time.time() - start_time)
        taylor_memories.append(get_peak_memory())
        clear_cuda_cache()


    # # 2. Autograd方法测试
    autograd_times = []
    autograd_memories = []
    
    for i in range(num_iterations):
        reset_peak_memory()
        
        start_time = time.time()
        
        # 计算Jacobian
        y = original_forward(x)  # 现在y是模型的输出
        # print(y)
        # 构造loss并反向传播
        
        loss = (hessian(y, x, component=0, i=0,j=0)+hessian(y, x, component=0, i=1,j=1)).mean()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        
        autograd_times.append(time.time() - start_time)
        autograd_memories.append(get_peak_memory())
        clear_cuda_cache()
    
    
    # 输出统计结果
    print("\nAutograd方法:")
    print(f"平均时间: {sum(autograd_times)/len(autograd_times):.4f} 秒")
    print(f"最小时间: {min(autograd_times):.4f} 秒")
    print(f"最大时间: {max(autograd_times):.4f} 秒")
    print(f"平均显存占用: {sum(autograd_memories)/len(autograd_memories):.1f} MB")
    print(f"最大显存占用: {max(autograd_memories):.1f} MB")

    
    print("\nTaylor mode方法:")
    print(f"平均时间: {sum(taylor_times)/len(taylor_times):.4f} 秒")
    print(f"最小时间: {min(taylor_times):.4f} 秒")
    print(f"最大时间: {max(taylor_times):.4f} 秒")
    print(f"平均显存占用: {sum(taylor_memories)/len(taylor_memories):.1f} MB")
    print(f"最大显存占用: {max(taylor_memories):.1f} MB")

    print("\n3阶部分求导(含loss.backward()):")
    # 1. Taylor mode方法测试
    taylor_times = []
    taylor_memories = []
    
    for i in range(num_iterations):
        reset_peak_memory()
        # gc.collect()
        # torch.cuda.empty_cache()
        
        start_time = time.time()
        
        # 计算Taylor展开
        result = taylor_forward(x)
        # 构造相同的loss并反向传播
        
        loss = (result.third[0,0,0]+result.third[1,1,1]).mean()   # （bs, y_dim, 0，0，0)+ （bs, y_dim, 1，1，1）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        taylor_times.append(time.time() - start_time)
        taylor_memories.append(get_peak_memory())
        clear_cuda_cache()


    # # 2. Autograd方法测试
    autograd_times = []
    autograd_memories = []
    
    for i in range(num_iterations):
        reset_peak_memory()
        
        start_time = time.time()
        
        # 计算Jacobian
        y = original_forward(x)  # 现在y是模型的输出
        # print(y)
        # 构造loss并反向传播
        
        jac0 = jacobian(y, x, i=0,j=0)
        hess0 = jacobian(jac0, x, i=0, j=0)
        third0 = jacobian(hess0, x, i=0, j=0)   # （bs, y0, 0，0，0）
        jac1 = jacobian(y, x, i=0,j=1)
        hess1 = jacobian(jac1, x, i=0, j=1)
        third1 = jacobian(hess1, x, i=0, j=1)   # （bs, y0, 1，1，1）
        jac2 = jacobian(y, x, i=1,j=0)
        hess2 = jacobian(jac2, x, i=0, j=0)
        third2 = jacobian(hess2, x, i=0, j=0)   # （bs, y1, 0，0，0）
        jac3 = jacobian(y, x, i=1,j=1)
        hess3 = jacobian(jac3, x, i=0, j=1)
        third3 = jacobian(hess3, x, i=0, j=1)   # （bs, y1, 1，1，1）
        loss = (third0+third1+third2+third3).mean()  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        autograd_times.append(time.time() - start_time)
        autograd_memories.append(get_peak_memory())
        clear_cuda_cache()
    
    
    # 输出统计结果
    print("\nAutograd方法:")
    print(f"平均时间: {sum(autograd_times)/len(autograd_times):.4f} 秒")
    print(f"最小时间: {min(autograd_times):.4f} 秒")
    print(f"最大时间: {max(autograd_times):.4f} 秒")
    print(f"平均显存占用: {sum(autograd_memories)/len(autograd_memories):.1f} MB")
    print(f"最大显存占用: {max(autograd_memories):.1f} MB")

    
    print("\nTaylor mode方法:")
    print(f"平均时间: {sum(taylor_times[1:])/len(taylor_times[1:]):.4f} 秒")
    print(f"最小时间: {min(taylor_times[1:]):.4f} 秒")
    print(f"最大时间: {max(taylor_times[1:]):.4f} 秒")
    print(f"平均显存占用: {sum(taylor_memories[1:])/len(taylor_memories[1:]):.1f} MB")
    print(f"最大显存占用: {max(taylor_memories[1:]):.1f} MB")

    print("\n2阶求导全量:(含loss.backward())")
    # 1. Taylor mode方法测试
    taylor_times = []
    taylor_memories = []
    
    for i in range(num_iterations):
        reset_peak_memory()
        # gc.collect()
        # torch.cuda.empty_cache()
        x = torch.randn(batch_size, in_dim, dtype=dtype, requires_grad=True, device=device)
        start_time = time.time()
        
        # 计算Taylor展开
        result = taylor_forward(x)
        # 构造相同的loss并反向传播
        
        loss = (result.second[...]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        taylor_times.append(time.time() - start_time)
        taylor_memories.append(get_peak_memory())
        clear_cuda_cache()


    # # 2. Autograd方法测试
    autograd_times = []
    autograd_memories = []
    
    for i in range(num_iterations):
        reset_peak_memory()
        x = torch.randn(batch_size, in_dim, dtype=dtype, requires_grad=True, device=device)
        start_time = time.time()
        
        # 计算Jacobian
        y = original_forward(x)  # 现在y是模型的输出
        # print(y)
        # 构造loss并反向传播
        
        # loss = (hessian(y, x, component=0, i=0,j=0)+hessian(y, x, component=0, i=1,j=1)).mean()
        jac1 = jacobian(y, x, i=0)
        jac2 = jacobian(y, x, i=1)

        
        hess0 = hessian(y, x, component=0, i=0,j=0, grad_y=jac1)+hessian(y, x, component=0, i=0,j=1, grad_y=jac1) + hessian(y, x, component=0, i=0,j=2, grad_y=jac1) 
        + hessian(y, x, component=0, i=1,j=0, grad_y=jac1) + hessian(y, x, component=0, i=1,j=1, grad_y=jac1) + hessian(y, x, component=0, i=1,j=2, grad_y=jac1)
        + hessian(y, x, component=0, i=2,j=0, grad_y=jac1) + hessian(y, x, component=0, i=2,j=1, grad_y=jac1) + hessian(y, x, component=0, i=2,j=2, grad_y=jac1)
        hess1 = hessian(y, x, component=1, i=0,j=0, grad_y=jac2)+hessian(y, x, component=1, i=0,j=1, grad_y=jac2) + hessian(y, x, component=1, i=0,j=2, grad_y=jac2) 
        + hessian(y, x, component=1, i=1,j=0, grad_y=jac2) + hessian(y, x, component=1, i=1,j=1, grad_y=jac2) + hessian(y, x, component=1, i=1,j=2, grad_y=jac2)
        + hessian(y, x, component=1, i=2,j=0, grad_y=jac2) + hessian(y, x, component=1, i=2,j=1, grad_y=jac2) + hessian(y, x, component=1, i=2,j=2, grad_y=jac2)
        loss = (hess0+hess1).mean()
        # breakpoint()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        autograd_times.append(time.time() - start_time)
        autograd_memories.append(get_peak_memory())
        clear_cuda_cache()
    
    # 输出统计结果
    print("\nAutograd方法:")
    print(f"平均时间: {sum(autograd_times)/len(autograd_times):.4f} 秒")
    print(f"最小时间: {min(autograd_times):.4f} 秒")
    print(f"最大时间: {max(autograd_times):.4f} 秒")
    print(f"平均显存占用: {sum(autograd_memories)/len(autograd_memories):.1f} MB")
    print(f"最大显存占用: {max(autograd_memories):.1f} MB")
    
    
    print("\nTaylor mode方法:")
    print(f"平均时间: {sum(taylor_times[1:])/len(taylor_times[1:]):.4f} 秒")
    print(f"最小时间: {min(taylor_times[1:]):.4f} 秒")
    print(f"最大时间: {max(taylor_times[1:]):.4f} 秒")
    print(f"平均显存占用: {sum(taylor_memories[1:])/len(taylor_memories[1:]):.1f} MB")
    print(f"最大显存占用: {max(taylor_memories[1:]):.1f} MB")



if __name__ == "__main__":
    # test_elementwise_activation()
    # test_linear_layer()
    # test_swish_derivatives()
    # test_custom_activation()
    # test_simple_example()
    # test_linear_activation_multiply()
    # test_other_components()
    # test_linear_activation_composition()
    # test_linear_multiply_composition()
    # test_functional_compatibility()
    
    # test_derivative_tensor()
    # test_taylor_multiplication()
    # test_taylor_indexing()
    # test_broadcast_multiplication()
    # test_broadcast_operations()
    
    test_modified_fourier_net()
    # test_performance_comparison()