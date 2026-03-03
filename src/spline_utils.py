import numpy as np
from scipy.interpolate import BSpline

def get_bspline_matrix(x, knots, degree=3):
    """
    Constructs an optimized Design Matrix X for B-Splines.
    
    Technical Upgrades:
    - Degree 3 (Cubic): Guarantees C2 continuity (continuous second derivatives).
      This ensures that implied forward credit spreads are smooth.
    - Clamped Boundary Conditions: Repeats boundary knots 'degree' times to 
      force the model to respect the start and end points of the market data.
    - Vectorized Logic: Leverages Scipy's optimized backend.
    """
    
    # 1. Knot Padding (Requirement for de Boor's Algorithm)
    # We repeat the first and last knots 'degree' times to 'clamp' the spline.
    knots_padded = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
    
    # 2. Basis Dimension Calculation
    n_basis = len(knots) + degree - 1
    
    # 3. Initialize Design Matrix (X)
    X = np.zeros((len(x), n_basis))
    
    # 4. Generate and Evaluate Basis Functions
    for i in range(n_basis):
        # Create a unit coefficient vector to isolate the i-th basis function
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        
        # Instantiate the spline function for this specific basis
        spline_basis_func = BSpline(knots_padded, coeffs, degree)
        
        # Evaluate the function across all input tenors (x)
        X[:, i] = spline_basis_func(x)
        
    return X