import autograd.numpy as np  
import matplotlib.pyplot as plt
from autograd import grad, hessian

# Définition des fonctions d'optimisation (les mêmes que dans le code initial)
def func(x):
    return x[0]**2 + 1.41*x[0]*x[1] + x[1]**2 + 1.41*x[1]*x[2] + x[2]**2

grad_func_auto = grad(func)
def grad_func(x):
    return grad_func_auto(x)

hessian_func_auto = hessian(func)
def hessian_func(x):
    return hessian_func_auto(x)


def gradient_descent_fixed_step(f, grad_f, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    x = x0
    for i in range(max_iter):
        gradient = grad_f(x)
        x_new = x - learning_rate * gradient
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def gradient_descent_variable_step(f, grad_f, x0, initial_lr=0.1, decay_rate=0.8, max_iter=1000, tol=1e-6):
    x = x0
    learning_rate = initial_lr
    for i in range(max_iter):
        gradient = grad_f(x)
        x_new = x - learning_rate * gradient
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        learning_rate *= decay_rate  # Diminution du pas
    return x

def newtons_method(f, grad_f, hessian_f, x0, max_iter=100, tol=1e-6):
    x = x0
    for i in range(max_iter):
        gradient = grad_f(x)
        hessian = hessian_f(x)
        if np.linalg.norm(gradient) < tol:
            break
        x_new = x - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def quasi_newtons_method(f, grad_f, x0, max_iter=100, tol=1e-6):
    x = x0
    n = len(x0)
    H = np.eye(n)  # Matrice d'identité initiale comme approximation de la Hessienne
    for i in range(max_iter):
        gradient = grad_f(x)
        if np.linalg.norm(gradient) < tol:
            break
        p = -H @ gradient
        x_new = x + p
        s = x_new - x
        y = grad_f(x_new) - gradient
        rho = 1.0 / (y @ s)
        H = (np.eye(n) - rho * np.outer(s, y)) @ H @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
    return x

def nesterov_accelerated_gradient(f, grad_f, x0, learning_rate=0.01, momentum=0.9, max_iter=1000, tol=1e-6):
    x = x0
    v = np.zeros_like(x0)  # Initialiser la vitesse à zéro
    for i in range(max_iter):
        x_ahead = x - momentum * v  # Calculer le point "anticipé"
        gradient = grad_f(x_ahead)
        v = momentum * v + learning_rate * gradient
        x_new = x - v
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x


def plot_convergence_2d(paths):
    """Affiche les chemins de convergence pour chaque méthode d'optimisation en 2D avec des lignes de niveau."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Créer un espace 2D pour visualiser les trajectoires
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 1.41*X*Y + Y**2  # Fonction en 2D dans l’espace x-y

    # Contours de niveau
    contour = ax.contour(X, Y, Z, levels=20, cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8)

    # Ajouter les trajectoires de chaque méthode
    for method, path in paths.items():
        x_vals = [p[0] for p in path]
        y_vals = [p[1] for p in path]

        ax.plot(x_vals, y_vals, marker='o', linestyle='-', label=method)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title("Convergence Paths of Different Optimization Methods (2D Contour Plot)")
    plt.show()

# Modification de main pour appeler la fonction en 2D
def main():
    # Point de départ pour la descente
    x0 = np.array([1.0, 2.0, 3.0])

    # Paramètres de descente
    learning_rate = 0.1
    max_iter = 100000

    # Stockage des chemins de convergence pour affichage
    paths = {
        'Gradient Descent Fixed': [],
        'Gradient Descent Variable': [],
        'Newton\'s Method': [],
        'Quasi-Newton (BFGS)': [],
        'Nesterov Accelerated Gradient': []
    }

    # 1. Test de Gradient Descent à Pas Fixe
    x_opt_fixed = gradient_descent_fixed_step(func, grad_func, x0, learning_rate, max_iter)
    paths['Gradient Descent Fixed'].append(x_opt_fixed)

    # 2. Test de Gradient Descent à Pas Variable
    x_opt_variable = gradient_descent_variable_step(func, grad_func, x0, learning_rate, max_iter=max_iter)
    paths['Gradient Descent Variable'].append(x_opt_variable)

    # 3. Test de Newton's Method
    x_opt_newton = newtons_method(func, grad_func, hessian_func, x0, max_iter=max_iter)
    paths['Newton\'s Method'].append(x_opt_newton)

    # 4. Test de Quasi-Newton (BFGS)
    x_opt_quasi_newton = quasi_newtons_method(func, grad_func, x0, max_iter=max_iter)
    paths['Quasi-Newton (BFGS)'].append(x_opt_quasi_newton)

    # 5. Test de Nesterov Accelerated Gradient
    x_opt_nesterov = nesterov_accelerated_gradient(func, grad_func, x0, learning_rate, max_iter=max_iter)
    paths['Nesterov Accelerated Gradient'].append(x_opt_nesterov)

    # Affichage des chemins de convergence en 2D
    plot_convergence_2d(paths)

# Appeler la fonction main pour exécuter le test
if __name__ == "__main__":
    main()
