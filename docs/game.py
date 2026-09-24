# Browser (PyScript / Pyodide) port of AIvsHuman/color_game_code.py.
#
# This file holds only the game logic. All drawing happens in app.js, which
# talks to Python through two functions exposed on `window`:
#
#   window.pyNewGame()            -> JSON string of the game state
#   window.pySubmit(c, m, y, k)   -> JSON string of the game state
#
# To move to a pure-JavaScript version later, reimplement those two functions
# in JS with the same JSON shape and app.js will keep working unchanged.

import json
import random

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from pyodide.ffi import create_proxy
from pyscript import window


class ColorGame:
    def __init__(self, seed=None):
        self.seed = random.randint(0, 10000) if seed is None else seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Target color in RGB format
        self.target_color = (np.random.rand(3) * 255).astype(int)
        self.target_color_CMYK = self.RGB_to_CMYK(self.target_color)

        # Optimization parameters for Bayesian Optimization
        self.hyperparameter = 0.01
        self.scaling_factor = 50
        self.win_threshold = 5  # % error needed to win

        # Initial random CMYK samples, shared by the player and the AI
        self.n_samples = 8
        self.max_well_volume = 250
        self.random_CMYK = self.latin_hypercube_sum_constraint(self.n_samples, self.max_well_volume) / self.max_well_volume

        self.X = self.random_CMYK.copy()
        self.y = -np.linalg.norm(self.X - self.target_color_CMYK, axis=1)
        self.bounds = [(0, 1)] * 4  # CMYK values are normalized between 0 and 1

        # Gaussian Process Regression model
        kernel = C(constant_value=1, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-4, normalize_y=False)

        self.cmyk_attempts = []
        self.user_distances = []
        self.bo_guesses = []
        self.bo_distances = []
        self.winner = None

    # Latin hypercube with the "maximin" criterion (same idea as pyDOE's
    # lhs(criterion='maximin')): draw a few designs, keep the most spread out.
    def latin_hypercube(self, num_samples, dim=4, iterations=5):
        best, best_dist = None, -1
        edges = np.linspace(0, 1, num_samples + 1)[:num_samples]
        for _ in range(iterations):
            samples = edges[:, None] + np.random.rand(num_samples, dim) / num_samples
            for j in range(dim):
                samples[:, j] = np.random.permutation(samples[:, j])
            d = pdist(samples).min()
            if d > best_dist:
                best, best_dist = samples, d
        return best

    def latin_hypercube_sum_constraint(self, num_samples, sum_constraint):
        lhs_samples = self.latin_hypercube(num_samples)

        # Adjust each sample to meet the sum constraint
        for i in range(num_samples):
            current_sum = np.sum(lhs_samples[i, :])
            lhs_samples[i, :] *= sum_constraint / current_sum

        return lhs_samples

    # Function to convert CMYK to RGB
    def CMYK_to_RGB(self, c, m, y, k):
        R = (1 - c) * (1 - k)
        G = (1 - m) * (1 - k)
        B = (1 - y) * (1 - k)
        return np.array([R, G, B])

    def RGB_to_CMYK(self, RGB_sample):
        Red, Green, Blue = np.asarray(RGB_sample) / 255

        Black = min(1 - Red, 1 - Green, 1 - Blue)
        if Black >= 1:  # pure black: avoid dividing by zero
            return np.array([0.0, 0.0, 0.0, 1.0])
        Cyan = (1 - Red - Black) / (1 - Black)
        Magenta = (1 - Green - Black) / (1 - Black)
        Yellow = (1 - Blue - Black) / (1 - Black)

        return np.array([Cyan, Magenta, Yellow, Black])

    # Expected Improvement maximizes implicitly
    def expected_improvement(self, X, X_sample, y_sample, model):
        mu, sigma = model.predict(X, return_std=True)

        optiminal_sample = np.min(y_sample)

        with np.errstate(divide="ignore", invalid="ignore"):
            imp = mu - optiminal_sample - self.hyperparameter
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei.flatten()

    # Function to propose the next sample point using Expected Improvement
    def propose_location(self, acquisition, X_sample, y_sample, bounds, n_restarts=25):
        dim = X_sample.shape[1]

        def min_obj(x):
            return -acquisition(x.reshape(-1, dim), X_sample, y_sample, self.gp)

        # Starting points for optimization
        x0_list = np.random.uniform(0, 1, size=(n_restarts, dim))
        best_x, best_val = None, float("inf")
        for x0 in x0_list:
            res = minimize(min_obj, x0, bounds=bounds, method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x

    def submit_color(self, c, m, y, k):
        if self.winner:
            return

        cmyk_values = np.array([c, m, y, k], dtype=float)
        distance = np.linalg.norm(cmyk_values - self.target_color_CMYK) * self.scaling_factor
        if np.isnan(distance) or np.isinf(distance):
            distance = 0.0

        self.cmyk_attempts.append(cmyk_values)
        self.user_distances.append(float(distance))

        # The AI makes one Bayesian Optimization guess per player guess
        self.gp.fit(self.X, self.y)
        X_next = self.propose_location(self.expected_improvement, self.X, self.y, self.bounds)
        y_next = -np.linalg.norm(X_next - self.target_color_CMYK)

        self.X = np.vstack((self.X, X_next))
        self.y = np.append(self.y, y_next)
        self.bo_guesses.append(X_next)
        self.bo_distances.append(float(-y_next * self.scaling_factor))

        # Determine the winner (the player wins a tie)
        if self.user_distances[-1] < self.win_threshold:
            self.winner = "user"
        elif self.bo_distances[-1] < self.win_threshold:
            self.winner = "ai"

    def to_rgb255(self, cmyk):
        return [int(v) for v in (self.CMYK_to_RGB(*cmyk) * 255).astype(int)]

    def state(self):
        def swatch(cmyk, error=None):
            return {
                "cmyk": [float(v) for v in cmyk],
                "rgb": self.to_rgb255(cmyk),
                "error": error,
            }

        return {
            "target": {
                "rgb": [int(v) for v in self.target_color],
                "cmyk": [float(v) for v in self.target_color_CMYK],
            },
            "random_samples": [swatch(s) for s in self.random_CMYK],
            "user_attempts": [swatch(s, d) for s, d in zip(self.cmyk_attempts, self.user_distances)],
            "ai_attempts": [swatch(s, d) for s, d in zip(self.bo_guesses, self.bo_distances)],
            "best_error": min(self.user_distances) if self.user_distances else None,
            "win_threshold": self.win_threshold,
            "winner": self.winner,
        }


game = None


def new_game():
    global game
    game = ColorGame()
    return json.dumps(game.state())


def submit(c, m, y, k):
    game.submit_color(c, m, y, k)
    return json.dumps(game.state())


window.pyNewGame = create_proxy(new_game)
window.pySubmit = create_proxy(submit)
window.dispatchEvent(window.CustomEvent.new("python-ready"))
