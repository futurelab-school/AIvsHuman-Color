// JavaScript port of AIvsHuman/color_game_code.py (the game logic).
//
// The notebook's ipywidgets/matplotlib display code is not ported here; the
// web page's UI lives in app.js. Python libraries are replaced as follows:
//   numpy random          -> a small seeded random number generator (mulberry32)
//   pyDOE3 lhs            -> latinHypercube()
//   sklearn GP regressor  -> GaussianProcessRegressor below (C * RBF kernel)
//   scipy L-BFGS-B        -> minimizeBounded(), a projected BFGS method
//   scipy.stats.norm      -> normPdf() / normCdf()

// ---------- Random numbers ----------

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function uniform(rand, low, high) {
  return low + (high - low) * rand();
}

function standardNormal(rand) {
  // Box-Muller transform
  const u = 1 - rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function permutation(rand, arr) {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ---------- Small linear algebra helpers ----------

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const norm2 = (a) => Math.sqrt(dot(a, a));
const identity = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const matVec = (M, v) => M.map((row) => dot(row, v));

function sqDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
}

// Lower-triangular Cholesky factor of A, or null if A is not positive definite
function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(s > 0)) return null;
        L[i][i] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

// Solve L x = b
function solveLower(L, b) {
  const n = b.length;
  const x = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

// Solve L^T x = b
function solveUpperT(L, b) {
  const n = b.length;
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

// Solve (L L^T) x = b
const choSolve = (L, b) => solveUpperT(L, solveLower(L, b));

// ---------- Normal distribution (scipy.stats.norm) ----------

function normPdf(z) {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
}

// Complementary error function (Numerical Recipes erfcc, fractional error < 1.2e-7)
function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

function normCdf(z) {
  return 0.5 * erfc(-z / Math.SQRT2);
}

// ---------- Bounded minimizer (stand-in for scipy's L-BFGS-B) ----------

// fg(x) returns { f, g }: the function value and its gradient.
function minimizeBounded(fg, x0, bounds, { maxIter = 15000, pgtol = 1e-5, ftol = 2.220446049250313e-9 } = {}) {
  const n = x0.length;
  const lower = bounds.map((b) => b[0]);
  const upper = bounds.map((b) => b[1]);
  const clip = (x) => x.map((v, i) => Math.min(upper[i], Math.max(lower[i], v)));

  let x = clip(x0);
  let { f, g } = fg(x);
  let H = identity(n); // inverse Hessian approximation
  let fresh = true;

  for (let iter = 0; iter < maxIter; iter++) {
    // Stop when the projected gradient is small
    const pg = x.map((v, i) => Math.min(upper[i], Math.max(lower[i], v - g[i])) - v);
    if (Math.max(...pg.map(Math.abs)) <= pgtol) break;

    // Variables pinned at a bound with the gradient pushing outward stay fixed
    const free = x.map((v, i) => !((v <= lower[i] && g[i] > 0) || (v >= upper[i] && g[i] < 0)));
    let d = matVec(H, g).map((v, i) => (free[i] ? -v : 0));
    if (!(dot(d, g) < 0)) {
      H = identity(n);
      fresh = true;
      d = g.map((v, i) => (free[i] ? -v : 0));
    }
    if (fresh) {
      // Like L-BFGS-B, keep the first step to unit length
      const len = norm2(d);
      if (len > 1) d = d.map((v) => v / len);
    }

    // Backtracking line search along the projected path
    let t = 1, xNew, fNew, gNew, accepted = false;
    for (let k = 0; k < 40; k++) {
      xNew = clip(x.map((v, i) => v + t * d[i]));
      ({ f: fNew, g: gNew } = fg(xNew));
      if (fNew <= f + 1e-4 * dot(g, sub(xNew, x))) {
        accepted = true;
        break;
      }
      t *= 0.5;
    }
    if (!accepted) break;

    const s = sub(xNew, x);
    const yv = sub(gNew, g);
    const fOld = f;
    x = xNew;
    f = fNew;
    g = gNew;
    if (fOld - f <= ftol * Math.max(Math.abs(fOld), Math.abs(f), 1)) break;

    // BFGS update of the inverse Hessian
    const sy = dot(s, yv);
    if (sy > 1e-10) {
      const rho = 1 / sy;
      const Hy = matVec(H, yv);
      const yHy = dot(yv, Hy);
      H = H.map((row, i) => row.map((h, j) =>
        h - rho * (s[i] * Hy[j] + Hy[i] * s[j]) + (rho * rho * yHy + rho) * s[i] * s[j]));
      fresh = false;
    }
  }
  return { x, fun: f };
}

// Forward-difference gradient, as scipy uses when no gradient is given
function withNumericGradient(f, bounds) {
  return (x) => {
    const fx = f(x);
    const g = x.map((v, i) => {
      let h = 1.4901161193847656e-8 * Math.max(1, Math.abs(v));
      if (v + h > bounds[i][1]) h = -h;
      const xh = x.slice();
      xh[i] = v + h;
      return (f(xh) - fx) / h;
    });
    return { f: fx, g };
  };
}

// ---------- Gaussian Process Regression (sklearn GaussianProcessRegressor) ----------

// Kernel: C(constant_value) * RBF(length_scale), hyperparameters optimized in log space
class GaussianProcessRegressor {
  constructor(rand, { constantValue = 1, lengthScale = 1, bounds = [1e-3, 1e3], nRestartsOptimizer = 10, alpha = 1e-4 } = {}) {
    this.rand = rand;
    this.initialTheta = [Math.log(constantValue), Math.log(lengthScale)];
    this.thetaBounds = [[Math.log(bounds[0]), Math.log(bounds[1])], [Math.log(bounds[0]), Math.log(bounds[1])]];
    this.nRestartsOptimizer = nRestartsOptimizer;
    this.alpha = alpha;
  }

  kernel(A, B, [c, l]) {
    return A.map((a) => B.map((b) => c * Math.exp(-0.5 * sqDist(a, b) / (l * l))));
  }

  // Log marginal likelihood and its gradient with respect to log(c), log(l)
  logMarginalLikelihood(theta) {
    const [c, l] = theta.map(Math.exp);
    const X = this.XTrain;
    const n = X.length;
    const K0 = this.kernel(X, X, [c, l]);
    const K = K0.map((row, i) => row.map((v, j) => (i === j ? v + this.alpha : v)));
    const L = cholesky(K);
    if (!L) return { lml: -Infinity, grad: [0, 0] };

    const alpha = choSolve(L, this.yTrain);
    let lml = -0.5 * dot(this.yTrain, alpha) - n / 2 * Math.log(2 * Math.PI);
    for (let i = 0; i < n; i++) lml -= Math.log(L[i][i]);

    // grad_j = 0.5 * trace((alpha alpha^T - K^-1) dK/dtheta_j)
    const Kinv = identity(n).map((e) => choSolve(L, e)); // symmetric, so rows = columns
    const grad = [0, 0];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const tmp = alpha[i] * alpha[j] - Kinv[i][j];
        grad[0] += tmp * K0[i][j];
        grad[1] += tmp * K0[i][j] * sqDist(X[i], X[j]) / (l * l);
      }
    }
    return { lml, grad: grad.map((v) => 0.5 * v) };
  }

  fit(X, y) {
    this.XTrain = X.map((r) => r.slice());
    this.yTrain = y.slice();

    const objective = (theta) => {
      const { lml, grad } = this.logMarginalLikelihood(theta);
      return { f: -lml, g: grad.map((v) => -v) };
    };

    // First run starts from the initial kernel, then random restarts in log space
    let best = minimizeBounded(objective, this.initialTheta, this.thetaBounds);
    for (let r = 0; r < this.nRestartsOptimizer; r++) {
      const theta0 = this.thetaBounds.map(([lo, hi]) => uniform(this.rand, lo, hi));
      const res = minimizeBounded(objective, theta0, this.thetaBounds);
      if (res.fun < best.fun) best = res;
    }
    this.kernelParams = best.x.map(Math.exp);

    const K = this.kernel(X, X, this.kernelParams).map((row, i) => row.map((v, j) => (i === j ? v + this.alpha : v)));
    this.L = cholesky(K);
    this.alphaVec = choSolve(this.L, this.yTrain);
    return this;
  }

  // Returns { mean, std } arrays for the rows of Xtest
  predict(Xtest) {
    const Ktrans = this.kernel(Xtest, this.XTrain, this.kernelParams);
    const mean = Ktrans.map((row) => dot(row, this.alphaVec));
    const std = Ktrans.map((row) => {
      const v = solveLower(this.L, row);
      return Math.sqrt(Math.max(0, this.kernelParams[0] - dot(v, v)));
    });
    return { mean, std };
  }
}

// ---------- The game ----------

class ColorGame {
  constructor(seed) {
    this.seed = seed ?? Math.floor(Math.random() * 10001); // Random seed for reproducibility
    this.rand = mulberry32(this.seed);

    // Target color in RGB format
    this.targetColor = [0, 1, 2].map(() => Math.floor(this.rand() * 255)); // Random target color
    this.targetColorNorm = this.targetColor.map((v) => v / 255);
    this.targetColorCMYK = this.RGBToCMYK(this.targetColor);
    this.targetString = this.targetColor.join(", ");

    // Optimization Parameters for Bayesian Optimization
    this.hyperparameter = 0.01;
    this.scalingFactor = 50;
    this.threshold = 5;

    // Generate a row of random CMYK samples
    this.nSamples = 8;
    this.maxWellVolume = 250;

    this.buildGame();
  }

  uniformSampling(numRows) {
    return Array.from({ length: numRows }, () => [0, 1, 2, 3].map(() => this.rand()));
  }

  normalSampling(numRows) {
    // Normal distribution samples are scaled to [0, 1] using the sigmoid function
    return Array.from({ length: numRows }, () => [0, 1, 2, 3].map(() => 1 / (1 + Math.exp(-standardNormal(this.rand)))));
  }

  stratifiedSampling(numRows) {
    const strata = Array.from({ length: 8 }, (_, i) => (i + 0.5) / 8);
    return Array.from({ length: numRows }, () => [0, 1, 2, 3].map(() => strata[Math.floor(this.rand() * strata.length)]));
  }

  // Classic Latin hypercube: one point per equal-width bin in every dimension
  latinHypercube(numSamples, dim = 4) {
    const columns = Array.from({ length: dim }, () =>
      permutation(this.rand, Array.from({ length: numSamples }, (_, i) => (i + this.rand()) / numSamples)));
    return Array.from({ length: numSamples }, (_, i) => columns.map((col) => col[i]));
  }

  latinHypercubeSampling(numSamples) {
    return this.latinHypercube(numSamples);
  }

  // pyDOE's criterion='maximin': of 5 designs, keep the one with the largest minimum distance
  latinHypercubeMaximin(numSamples, dim = 4, iterations = 5) {
    let best = null, bestDist = 0;
    for (let it = 0; it < iterations; it++) {
      const candidate = this.latinHypercube(numSamples, dim);
      let minDist = Infinity;
      for (let i = 0; i < numSamples; i++) {
        for (let j = i + 1; j < numSamples; j++) minDist = Math.min(minDist, Math.sqrt(sqDist(candidate[i], candidate[j])));
      }
      if (bestDist < minDist) {
        bestDist = minDist;
        best = candidate;
      }
    }
    return best;
  }

  latinHypercubeSumConstraint(numSamples, sumConstraint) {
    const lhsSamples = this.latinHypercubeMaximin(numSamples);

    // Adjust each sample to meet the sum constraint
    return lhsSamples.map((row) => {
      const currentSum = row.reduce((a, b) => a + b, 0);
      return row.map((v) => v * sumConstraint / currentSum);
    });
  }

  // sobol_sampling and halton_sampling are not ported: the game does not use them.

  // Function to convert CMYK to RGB
  CMYKToRGB(c, m, y, k) {
    return [(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)];
  }

  RGBToCMYK(RGBSample) {
    const [red, green, blue] = RGBSample.map((v) => v / 255);

    const black = Math.min(1 - red, 1 - green, 1 - blue);
    const cyan = (1 - red - black) / (1 - black);
    const magenta = (1 - green - black) / (1 - black);
    const yellow = (1 - blue - black) / (1 - black);

    return [cyan, magenta, yellow, black];
  }

  // Calculate the 2-norm distance for each sample color
  normError(normSamples, targetColor) {
    return normSamples.map((sample) => norm2(sub(sample, targetColor.map((v) => v / 255))));
  }

  // Expected Improvement maximizes implicitly
  expectedImprovement(X, XSample, ySample, model) {
    const { mean, std } = model.predict(X);

    const optimalSample = Math.min(...ySample);

    return mean.map((mu, i) => {
      const sigma = std[i];
      if (sigma === 0) return 0;
      const imp = mu - optimalSample - this.hyperparameter;
      const Z = imp / sigma;
      return imp * normCdf(Z) + sigma * normPdf(Z);
    });
  }

  // Function to propose the next sample point using Expected Improvement
  proposeLocation(acquisition, XSample, ySample, bounds, nRestarts = 25) {
    const minObj = (x) => -acquisition.call(this, [x], XSample, ySample, this.gp)[0];
    const fg = withNumericGradient(minObj, bounds);

    // Starting points for optimization
    let bestX = null, bestVal = Infinity;
    for (let r = 0; r < nRestarts; r++) {
      const x0 = bounds.map(() => uniform(this.rand, 0, 1));
      const res = minimizeBounded(fg, x0, bounds);
      if (res.fun < bestVal) {
        bestVal = res.fun;
        bestX = res.x;
      }
    }
    return bestX;
  }

  distanceToTarget(cmyk) {
    return norm2(sub(cmyk, this.targetColorCMYK));
  }

  bayesianOptimization(randomSamples, maxIterations, threshold = 5) {
    // Initialize variables
    let X = randomSamples.map((r) => r.slice());
    let y = X.map((x) => -this.distanceToTarget(x));
    const bounds = X[0].map(() => [0, 1]); // CMYK values are normalized between 0 and 1

    const XSamples = [];
    const ySamples = [];

    for (let i = 0; i < maxIterations; i++) {
      this.gp.fit(X, y); // Fit GP model

      // Propose the next sampling point
      const XNext = this.proposeLocation(this.expectedImprovement, X, y, bounds);

      // Evaluate the objective function for the new sample
      const yNext = -this.distanceToTarget(XNext);

      // Append the new sample to the dataset
      X.push(XNext);
      y.push(yNext);

      XSamples.push(XNext);
      ySamples.push(yNext);

      // Check if the error is below the threshold
      if (-yNext < threshold / this.scalingFactor) {
        console.log(`Optimization stopped early: error below ${threshold} at iteration ${i + 1}`);
        break;
      }
    }
    return [XSamples, ySamples];
  }

  buildGame() {
    // Initialize storage for attempts
    this.userAttempts = [];
    this.cmykAttempts = [];
    this.boGuesses = [];
    this.userDistances = [];
    this.boDistances = [];
    this.XSamples = [];
    this.ySamples = [];
    this.winner = null;

    // Initialize variables
    this.bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]; // CMYK values are normalized between 0 and 1
    this.randomCMYK = this.latinHypercubeSumConstraint(this.nSamples, this.maxWellVolume).map((r) => r.map((v) => v / this.maxWellVolume));
    this.randomRGB = this.randomCMYK.map((cmyk) => this.CMYKToRGB(...cmyk)); // Convert to RGB

    this.X = this.latinHypercubeSumConstraint(this.nSamples, this.maxWellVolume).map((r) => r.map((v) => v / this.maxWellVolume));
    this.y = this.X.map((x) => -this.distanceToTarget(x));

    // Gaussian Process Regression model
    this.gp = new GaussianProcessRegressor(this.rand, { constantValue: 1, lengthScale: 1, bounds: [1e-3, 1e3], nRestartsOptimizer: 10, alpha: 1e-4 });
  }

  // Called when the player presses Submit
  submitColor(c, m, y, k) {
    // Convert user CMYK input to RGB
    const cmykValues = [c, m, y, k];
    const userRGB = this.CMYKToRGB(...cmykValues).map((v) => Math.floor(v * 255));

    // Compute Euclidean distance from the target color
    let distance = this.distanceToTarget(cmykValues) * this.scalingFactor;
    if (!Number.isFinite(distance)) distance = 0.0;

    // Store attempt data
    this.userAttempts.push(userRGB);
    this.cmykAttempts.push(cmykValues);
    this.userDistances.push(distance);

    // Only run BO if the best distance is greater than the threshold
    if (!this.boDistances.length || Math.min(...this.boDistances) > 0.05) {
      // Update GP model
      this.gp.fit(this.X, this.y);

      // Propose the next sampling point based on Bayesian Optimization
      const XNext = this.proposeLocation(this.expectedImprovement, this.X, this.y, this.bounds);
      const yNext = -this.distanceToTarget(XNext);

      // Append the new sample to the dataset
      this.XSamples.push(XNext);
      this.ySamples.push(yNext);

      this.X.push(XNext);
      this.y.push(yNext);

      // Track the selected color
      this.boDistances.push(-yNext * this.scalingFactor);
      this.boGuesses.push(XNext);
    }

    // Determine the winner
    if (this.userDistances.some((d) => d < this.threshold)) this.winner = "user";
    else if (this.boDistances.some((d) => d < this.threshold)) this.winner = "ai";
    else this.winner = null;
  }

  // Everything the UI needs to draw the game (same shape as game.py's state())
  state() {
    const swatch = (cmyk, error = null) => ({
      cmyk,
      rgb: this.CMYKToRGB(...cmyk).map((v) => Math.floor(v * 255)),
      error,
    });
    return {
      target: { rgb: this.targetColor, cmyk: this.targetColorCMYK },
      random_samples: this.randomCMYK.map((s) => swatch(s)),
      user_attempts: this.cmykAttempts.map((s, i) => swatch(s, this.userDistances[i])),
      ai_attempts: this.boGuesses.map((s, i) => swatch(s, this.boDistances[i])),
      best_error: this.userDistances.length ? Math.min(...this.userDistances) : null,
      win_threshold: this.threshold,
      winner: this.winner,
    };
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { ColorGame, GaussianProcessRegressor, minimizeBounded, withNumericGradient, normCdf, normPdf };
}
