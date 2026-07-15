# Physics-Informed Neural Networks for Holographic Quantum Chromodynamics

A fully-differentiable Python pipeline applying Physics-Informed Neural Networks (PINNs) and parametric inverse fitting to the Dynamic Anti-de Sitter / Yang-Mills (AdS/YM) model of holographic Quantum Chromodynamics (QCD).

## Abstract

This project addresses two complementary questions in the Dynamic AdS/YM model with $N_c = 3$ colours and $N_f = 2$ light flavours:

1. **Forward problem (Task 1)**: Can a PINN recover the vacuum embedding $L(\rho)$ from the equation of motion (EOM)? **Yes**, to a max absolute error of $\sim 10^{-3}$ against an independent scipy shooting solution.
2. **Inverse problem (Task 2)**: Starting from the Task 1 baseline, can $L(\rho)$ be modified to reproduce the experimental $\rho$-meson tower from the Particle Data Group (PDG) together with the pion decay constant $f_\pi$? **No**, within the standard chiral limit boundary conditions. A reduction of $\chi^2$ by a factor of $\sim 1200$ is achievable, but the residual pattern reveals a structural mismatch: the model produces a near-linear-in-$n$ spectrum while experiment shows Regge-like behaviour. This is a known limitation of holographic QCD at finite $N_c$ that motivates Higher-Dimensional Operator (HDO) extensions via Witten's double-trace prescription.

## Physics background

### The Dynamic AdS/YM model

The Dynamic AdS/YM model (Evans, Jones, and Scott, arXiv:1508.06540) is a holographic description of QCD-like gauge theories. The bulk geometry is five-dimensional AdS, and the flavour degrees of freedom live on a probe Dirac-Born-Infeld (DBI) brane. The brane embedding is described by a single function $L(\rho)$ measuring the separation of the brane from the AdS centre.

In the **vacuum** the embedding satisfies a second-order Ordinary Differential Equation (ODE):

$$L''(\rho) = -\frac{3}{\rho}\,L'(\rho) + \frac{\Delta m^2\!\left(\ln\sqrt{\rho^2 + L^2}\right)}{\rho^2}\,L(\rho)$$

with $\Delta m^2(\mu) = -2\gamma(\mu)$ given by the anomalous dimension of $\bar{q}q$ in the two-loop running coupling $\alpha(\mu)$. The boundary conditions are $L(\rho_{\text{IR}}) = \rho_{\text{IR}}$ and $L'(\rho_{\text{IR}}) = 0$ (the Infrared (IR) mass condition), together with $L(\rho_{\text{max}}) = 0$ (the chiral limit at the Ultraviolet (UV) cutoff).

The vector meson spectrum is obtained from the eigenvalue problem:

$$\partial_\rho\!\left[\rho^3 \, \partial_\rho V\right] + M_V^2 \frac{\rho^3}{r^4} V = 0$$

with $r^2 = \rho^2 + L^2$. The pion decay constant is given by the axial source integral at zero external momentum.

### Why this is hard for standard methods

The full pipeline — solving for the vacuum embedding, computing the meson spectrum, computing $f_\pi$, and tuning the embedding to match experiment — has historically been done in Mathematica. The forward pieces are tractable. The inverse problem (tuning $L(\rho)$ to match PDG masses) is harder: it requires gradient information flowing through an eigenvalue problem and a Boundary Value Problem (BVP) solve, neither of which has a natural Mathematica differentiation chain.

## What this project does

The project implements the entire pipeline in differentiable PyTorch:

| Component | Method | Reference value | Achieved |
|---|---|---|---|
| Vacuum embedding $L(\rho)$ | PINN, 5×64 tanh, hard Dirichlet boundary conditions | scipy shooting | max abs error $\sim 10^{-3}$ |
| Vector meson eigensolve | Chebyshev spectral collocation, BC elimination | Mathematica masses | rel. err $\sim 10^{-3}$ |
| Pion decay constant $f_\pi$ | Differentiable linear solve + trapezoidal quadrature | $\tilde{f}_\pi = 0.095858$ | rel. err $\sim 10^{-2}$ |
| Inverse fit to PDG | Chebyshev parametric ansatz (6 modes), Adam + L-BFGS | PDG masses + $f_\pi$ | $\chi^2$ reduced by $\sim 1200\times$ |

### Key technical choices

- **Hard boundary conditions in the PINN**: $L(\rho) = L_{\text{IR}} \cdot \frac{\rho_{\text{max}} - \rho}{\rho_{\text{max}} - \rho_{\text{IR}}} + (\rho - \rho_{\text{IR}})(\rho_{\text{max}} - \rho) \cdot N(\rho)$ enforces both Dirichlet conditions exactly. Only the Neumann condition $L'(\rho_{\text{IR}}) = 0$ is enforced as a soft penalty.
- **Normalised PDE residual**: dividing the equation of motion by $\rho^2$ keeps the residual $\mathcal{O}(1)$ across the domain, preventing the optimiser from being dominated by trivial UV behaviour where $L \approx 0$.
- **Chebyshev over finite differences**: spectral collocation reaches machine precision at $N \approx 30$ points where second-order finite differences would need thousands. This matters because each training step solves a Generalised Eigenvalue Problem (GEVP) and a linear system on the discretised grid.
- **Differentiable eigensolve via `torch.linalg.eig`**: gradients flow through the eigenvalue computation, allowing $L(\rho)$ to be tuned to match the spectrum.
- **Parametric ansatz for the inverse problem**: a 6-coefficient Chebyshev expansion on top of the Task 1 baseline. The neural network with $\sim 16{,}800$ parameters severely over-fits 6 data points; the parametric form has exactly enough degrees of freedom to be informative without being pathological.

## Machine Learning engineering notes

This section surfaces the machine learning design decisions for readers approaching the project from the Machine Learning (ML) side rather than the physics side.

### Why a PINN at all?

The forward problem (Task 1) does not need a neural network — the boundary value problem can be solved to machine precision by classical shooting in milliseconds. The PINN is justified as **infrastructure for Task 2**: the inverse problem requires gradient information flowing back from observables (meson masses, $f_\pi$) to the function $L(\rho)$, and a differentiable parametrisation of $L(\rho)$ is the natural way to provide that. Building Task 1 with a PINN gives a clean, differentiable baseline that Task 2 can warm-start from.

This pattern — using ML for the inverse problem and validating against a classical solver on the forward problem — is the same one used in much of the physics-informed ML literature. The Task 1 validation against scipy is not a goal in itself; it is a sanity check on the differentiable infrastructure.

### Activation function choice

We use `tanh` rather than Rectified Linear Unit (ReLU). PINNs evaluate the equation of motion residual by computing second derivatives of the network output via automatic differentiation. ReLU has $\sigma''(x) = 0$ almost everywhere, so a ReLU-based PINN cannot represent the curvature $L''(\rho)$ that appears in the equation of motion. The residual becomes pathological (zero or undefined) regardless of training. Smooth activations (`tanh`, `sin`, `swish`) are essential for any PINN that uses second-order derivatives.

### Hard versus soft boundary conditions

The boundary conditions $L(\rho_{\text{IR}}) = \rho_{\text{IR}}$ and $L(\rho_{\text{max}}) = 0$ are enforced **structurally** in the network output rather than as loss terms. This is preferable to adding $\lambda \cdot (L(\rho_{\text{IR}}) - \rho_{\text{IR}})^2 + \lambda \cdot L(\rho_{\text{max}})^2$ to the loss function for two reasons:

1. **No tradeoff with the physics loss**: a soft penalty leaves a residual boundary error inversely proportional to $\lambda$. Increasing $\lambda$ reduces the boundary error but degrades the physics fit. The hard form eliminates this tradeoff.
2. **Cleaner optimisation landscape**: the boundary loss has its own attractors that can stall training. Removing it removes those minima entirely.

The Neumann condition $L'(\rho_{\text{IR}}) = 0$ is enforced softly because the natural hard-enforcement form would require explicit construction of a function with vanishing derivative at the endpoint, which is more invasive than the Dirichlet case.

### Loss normalisation

The original equation of motion has the form $\rho^3 L'' + 3\rho^2 L' = \Delta m^2 \rho L$. Dividing by $\rho^2$ gives the normalised form $\rho L'' + 3 L' - \Delta m^2 L / \rho = 0$, and we use the normalised form in the loss. Without it, the $\rho^3$ factor makes the residual dominated by the UV region ($\rho \to \rho_{\text{max}}$), where $L \approx 0$ is trivial to satisfy. The optimiser would over-fit the boring asymptotic region and under-fit the IR where the physics is non-trivial. Rescaling the residual to be $\mathcal{O}(1)$ across the domain is essential for balanced training.

### Two-phase optimisation: Adam then L-BFGS

We use Adam for 10,000 epochs (initial coarse convergence) followed by Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) for 500 epochs (precise refinement). The rationale:

- **Adam** is robust to bad initialisations and noisy loss landscapes. PINN losses are non-convex with many local minima; Adam's momentum and per-parameter learning rates handle this regime well.
- **L-BFGS** is a quasi-Newton method that uses approximate second-order information. Near convergence, the PINN loss landscape is effectively a least-squares problem, and L-BFGS converges much faster than first-order methods on such problems.

This combination is standard in the PINN literature for the same reason it is used here.

### Differentiable physics

The Task 2 pipeline requires gradients to flow through:

1. The neural network parametrisation of $L(\rho)$
2. The Chebyshev evaluation of $L$ on the collocation grid
3. The construction of the eigenvalue problem matrices $\mathcal{H}[L]$ and $\mathcal{W}[L]$
4. The generalised eigenvalue solve $\mathcal{H} V = M^2 \mathcal{W} V$ (via `torch.linalg.eig`)
5. The construction of the linear system for the axial source $\Pi(\rho)$
6. The linear solve (via `torch.linalg.solve`)
7. The trapezoidal integration for $f_\pi^2$
8. The construction of the $\chi^2$ loss against PDG data

All eight steps are implemented in PyTorch with autograd support. The result is end-to-end gradient flow from each PDG datum back to the parameters of $L(\rho)$ and to the energy scale $\Lambda$. This is **differentiable physics** in the modern sense: the entire forward simulation is treated as a differentiable function, and the inverse problem becomes a standard ML optimisation problem.

The technique generalises beyond holography. Any physics pipeline whose forward evaluation can be expressed in a differentiable framework can be inverted against experimental data this way.

### The capacity-vs-data lesson

The most important ML decision in this project is in Task 2: we **replaced the neural network with a parametric Chebyshev expansion**. The neural network from Task 1 has $\sim 16{,}800$ parameters. The PDG data set has 6 numbers (5 meson masses plus $f_\pi$). This is a $2800{:}1$ parameter-to-data ratio that no realistic regularisation can save.

Preliminary attempts using the full network with smoothness penalty, positivity penalty, and anchor penalty toward the Task 1 baseline all produced visibly unphysical $L(\rho)$ profiles: wild oscillations, negative excursions, sharp spikes. The eigenvalue problem is exquisitely sensitive to local features in $L(\rho)$, and the optimiser exploited this sensitivity to fit the data by introducing pathological features that no smoothness measure could catch.

The clean solution was to switch to a 6-coefficient Chebyshev expansion, giving a $1{:}1$ parameter-to-data ratio. This:

- Guarantees smoothness by construction (no oscillation control needed)
- Produces interpretable fitted parameters (the $a_k$ coefficients have physical meaning)
- Makes overfitting structurally impossible
- Warm-starts cleanly from the Task 1 baseline ($a_k = 0$ recovers Task 1 exactly)

The lesson is general: **when the data is sparse, the right answer is usually not a bigger neural network with more regularisation, but a smaller model with appropriate inductive bias**. The Chebyshev expansion encodes the prior "$L(\rho)$ should be smooth and close to the vacuum solution" in its functional form. A neural network with regularisation tries to encode the same prior in the loss function and fails because the network has too many ways to evade the penalty.

### Hyperparameter choices

| Hyperparameter | Value | Rationale |
|---|---|---|
| Network depth | 5 hidden layers | Standard for PINNs on smooth one-dimensional problems |
| Network width | 64 neurons | Sufficient capacity; over-parametrisation is not penalised by the physics loss |
| Activation | `tanh` | Smooth, required for second-order autograd (see above) |
| Adam learning rate | $10^{-3}$ | Standard initial value, halved on plateau |
| L-BFGS line search | Strong Wolfe | Robust against poor initial step sizes |
| BC penalty $\lambda_{\text{BC}}$ | 50 | Chosen so the residual BC loss is comparable in magnitude to the PDE residual at convergence |
| Chebyshev modes $K$ | 6 | Matches the data dimension (5 masses + $f_\pi$) |
| Chebyshev grid size | 51 | Spectral convergence reaches machine precision by $N \approx 30$; 51 gives margin |
| $f_\pi$ uncertainty | 1% of $f_\pi^{\text{PDG}}$ | The experimental value is known to $\sim 0.4$ MeV; the relaxation prevents a single data point from dominating the loss |

## Findings

### Task 1: forward problem succeeds

The PINN reproduces the scipy shooting solution to better than $10^{-3}$ absolute error across the entire domain, including the boundary conditions. Training takes 10,000 Adam epochs followed by 500 L-BFGS epochs.

### Task 2: inverse problem reveals a structural limitation

The Task 1 baseline, anchored by $f_\pi$ to convert dimensionless to physical units, gives mass predictions that overshoot every experimental value by 66% to 240%. The parametric inverse fit reduces $\chi^2$ by a factor of $\sim 1200$, but the reduced $\chi^2 \approx 640$ is far from a good fit.

The residual pattern is informative: the lowest two states and $f_\pi$ are fit to within a few MeV, while $\rho(1700)$ and above remain hundreds to thousands of MeV too heavy. This is the well-known signature of holographic QCD producing $M_n \propto n$ rather than the experimentally observed $M_n^2 \propto n$ (Regge behaviour). The mismatch is *structural*: no smooth deformation of $L(\rho)$ within the standard chiral limit can reorganise a linear-spacing spectrum into a Regge-like one.

This finding is consistent with the published analysis in Erdmenger, Evans, Porod, and Rigatos (arXiv:2010.10279, where one of the present authors is a co-author), which notes the same limitation and motivates the inclusion of Higher-Dimensional Operators via Witten's double-trace prescription (arXiv:hep-th/0112258).

## File structure

```
.
├── README.md                        # This file
└── PINN_holographic_QCD.ipynb       # Full pipeline notebook
```

The notebook is self-contained and runs end-to-end. It is organised into three parts:

- **Part 1** — Imports, physics constants, running coupling, scipy ground truth solution
- **Part 2** — Task 1: PINN architecture, loss, training, validation against ground truth
- **Part 3** — Task 2: Chebyshev infrastructure, eigensolve, $f_\pi$, PDG baseline, parametric inverse fit, results

## Reproducibility

### Dependencies

- Python 3.10 or later
- `torch >= 2.0` (Central Processing Unit (CPU) is sufficient; Metal Performance Shaders (MPS) or Compute Unified Device Architecture (CUDA) supported)
- `numpy`, `scipy`, `matplotlib`
- `jupyter` for the notebook interface

A minimal installation is sufficient — no GPU is required. Task 1 trains in about 2 minutes on a modern CPU, Task 2 in about 1 minute.

### Running

```bash
jupyter notebook PINN_holographic_QCD.ipynb
```

Run all cells from top to bottom. Random seeds are set in Part 1, so results are deterministic up to platform-specific floating-point ordering.

### Numerical verification

Several cells include verification against independent references:

- The two-loop running coupling against Mathematica via $\rho_{\text{IR}}$
- The vacuum embedding against scipy shooting
- The vector meson masses against Mathematica reference values
- The pion decay constant against Mathematica reference value

If any of these fail by more than the stated tolerances after a clean run, the most likely cause is a stale Python kernel or a torch version with different default dtype handling.

## References

1. Evans, Jones, and Scott, *Dynamic AdS/QCD and the spectrum of walking gauge theories*, arXiv:1508.06540
2. Erdmenger, Evans, Porod, and Rigatos, *Gauge/gravity dynamics for composite Higgs models and the top mass*, arXiv:2010.10279
3. Witten, *Multi-trace operators, boundary conditions, and AdS/CFT correspondence*, arXiv:hep-th/0112258
4. Trefethen, *Spectral Methods in MATLAB*, SIAM (2000) — Chebyshev differentiation matrices
5. Raissi, Perdikaris, and Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, Journal of Computational Physics 378, 686 (2019)

## Acknowledgments

Development was assisted by discussions with Claude (Anthropic).

## Author

Konstantinos Rigatos. The physics framework and modelling decisions are based on the author's prior work on holographic QCD (arXiv:2010.10279 and related).

## License

This project is intended for educational and research purposes. Please cite the referenced papers when using or extending this work.
