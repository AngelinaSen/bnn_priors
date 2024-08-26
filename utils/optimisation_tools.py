import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize


class Optimization:
    def __init__(self, d, potential=None, logtarget=None, logprior=None, loglike=None, min=None, max=None):
        self.d = d
        self.logprior = logprior
        self.loglike = loglike
        self.bonds = [(min, max)] * d

        # set the potential
        if callable(potential):  # user directly passes the log-potential
            self.potential = potential

        elif callable(logtarget):  # user directly passes the log-target

            def target_fun(theta):
                eval, grad = logtarget(theta, True)
                return -eval, -grad

            self.potential = lambda theta: target_fun(theta)

        elif callable(logprior) and callable(loglike):  # user passes the log-prior and log-likelihood

            def target_fun(theta):
                eval_pr, grad_pr = logprior(theta, True)
                eval_like, grad_like = loglike(theta, True)
                return -eval_pr - eval_like, -grad_pr - grad_like

            self.potential = lambda theta: target_fun(theta)

    def adam(self, theta_0, num_iters=5000, step_size=1e-3, beta1=0.9, beta2=0.999, weight_decay=0, eps=1e-8):
        """
        Adam optimizer
        :param theta_0: starting point
        :param num_iters: number of iterations
        :param step_size: step size
        :param beta1: algorithm parameter
        :param beta2: algorithm parameter
        :param weight_decay: specifies weight decay if any
        :param eps:
        :return:
        """
        x = theta_0
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        feval = np.empty(num_iters + 1)
        feval[0] = 1e20
        print("===== ADAM optimization =====")
        for t in range(num_iters):
            # evaluate fun and its gradient
            feval[t + 1], grad = self.potential(x)
            if weight_decay != 0:
                grad += weight_decay * x
            # print(feval[t+1])
            if abs(feval[t + 1] > 2 * feval[t]):
                feval[t + 1] = feval[t]
                x += (step_size * m_hat) / (np.sqrt(v_hat) + eps)
                break

            # Compute first and second moment estimates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)

            # Compute corrected first and second moment estimates
            m_hat = m / (1 - beta1 ** (t + 1))
            v_hat = v / (1 - beta2 ** (t + 1))

            # Update parameters
            x -= (step_size * m_hat) / (np.sqrt(v_hat) + eps)

            # msg
            if np.mod(t, 10) == 0:
                print(f"ADAM at iteration {t}, f = {feval[t+1]}")
        fun_min_adam = feval[-1]

        return x, fun_min_adam

    def bfgs(self, theta_0, num_iters=3000, maxls=100):
        """
        Implementation of the BFGS optimisation algorithm
        :param theta_0: starting point
        :param num_iters: number of iterations
        :param maxls: maximum number of line search steps per iteration
        :return:
        """
        print("===== BFGS optimization =====")

        # Define the callback function to print information every n iterations
        n = 10

        def callback_function(xk):
            if callback_function.iteration % n == 0:
                print(f"L-BFGS-B at iteration {callback_function.iteration}, f = {self.potential(xk)[0]}")
            callback_function.iteration += 1

        callback_function.iteration = 0

        # run optimizer
        result = minimize(
            self.potential,
            theta_0,
            method="L-BFGS-B",
            bounds=self.bonds,
            jac=True,
            callback=callback_function,
            options={"maxiter": num_iters, "maxls": maxls},
        )
        theta_min, fun_min_bfgs = result.x, result.fun
        return theta_min, fun_min_bfgs

    def adam_bfgs(self, theta_0, num_iters=1500):
        """
        Function that first performs iterations of Adam optimization and next run BFGS optimization
        for faster convergence of the results
        :param theta_0: starting point
        :param num_iters: number of iterations to perform for Adam and BFGS
        :return:
        """

        theta_adam, funmin_adam = self.adam(theta_0, num_iters)

        n_bfgs = int(2 * num_iters)
        theta_bfgs, funmin_bfgs = self.bfgs(theta_adam, n_bfgs)

        return theta_bfgs, funmin_bfgs, theta_adam, funmin_adam

