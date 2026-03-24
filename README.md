# 🔢 Unconstrained Numerical Optimization Algorithms
 
Hands-on comparison of unconstrained optimization algorithms in Python: gradient descent (fixed & variable step via backtracking line search), Newton's method (exact Hessian), quasi-Newton BFGS (approximate Hessian), and Nesterov accelerated gradient — tested on a quadratic 3-variable function and visualized with 2D convergence path plots.
 
---
 
## 📁 Structure
 
| File | Description |
|---|---|
| `Algorithms.ipynb` | Jupyter Notebook — step-by-step implementation and comparison of optimization methods using pure numpy|
| `path_tracking.py` | Optimization methods with full convergence path tracking + 2D contour plot |
| `minima.py` | Optimization methods returning final optimal point + 2D convergence visualization |
 
---
 
## 📌 Methods Covered
 
| Method | Description |
|---|---|
| **Gradient Descent (Fixed Step)** | Constant learning rate, stops when step norm < tolerance |
| **Gradient Descent (Variable Step)** | Decaying learning rate via backtracking line search |
| **Newton's Method** | Uses exact Hessian matrix inverse at each iteration |
| **Quasi-Newton (BFGS)** | Approximates Hessian iteratively — also via `scipy.optimize.minimize` |
| **Nesterov Accelerated Gradient** | Momentum-based gradient descent with lookahead step |
 
> All methods are applied to the quadratic function:
> **f(x, y, z) = x² + βxy + y² + βyz + z²** with β = 1.41 and β = 0.11
 
---
 
## 📊 Visualizations
 
- 2D contour plot with convergence paths of all methods overlaid
- Comparison of convergence speed across methods
 
---
 
## 🛠️ Requirements
 
```bash
pip install numpy scipy matplotlib autograd jupyter
```
 
---
 
## 🚀 Getting Started
 
```bash
git clone https://github.com/your-username/unconstrained-numerical-optimization-algorithms.git
cd unconstrained-numerical-optimization-algorithms
```
 
Run the notebook:
```bash
jupyter notebook MASD_TP1_Série2.ipynb
```
 
Or run the scripts directly:
```bash
python td_print.py
python TD2_coupe_niveau_3D.py
```
 
---
