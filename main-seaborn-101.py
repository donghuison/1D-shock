import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.io import FortranFile

# Directory management
DATA_DIR = './data/'
FIG_DIR = './figs/'
os.makedirs(FIG_DIR, exist_ok=True)

# Set Seaborn style for improved aesthetics
sns.set_theme(style="darkgrid", palette="muted")
sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})


# Function to read data from an unformatted Fortran binary file
def read_fortran_binary(filepath, num_elements):
    """Reads and returns 'num_elements' from a Fortran binary file."""
    with FortranFile(filepath, 'r') as file:
        data = file.read_reals(np.float64)  # Reading as double precision real numbers (float64)
    return data[:num_elements]  # Return only 'num_elements' elements

# Constants related to the physical properties
JMAX = 101  # Number of grid points

# Spatial domain
XMIN, XMID, XMAX = 0.0, 0.501, 1.0
x = np.linspace(XMIN, XMAX, JMAX)
DX = (XMAX - XMIN) / (JMAX - 1)

TIME_STEP = 0.002  # Time step size
GAMMA = 1.4  # Adiabatic index (specific heat ratio)
NUM_STEPS = 100
DT_DX_RATIO = TIME_STEP / DX

# Initial conditions for shock and pre-shock regions
PRESSURE_RATIO = 2.5  # Pressure ratio across the shock
GAMMA_RATIO = (GAMMA - 1.0) / (GAMMA + 1.0)
SHOCK_FACTOR = GAMMA_RATIO + PRESSURE_RATIO
INITIAL_DENSITY_SHOCKED = SHOCK_FACTOR / (1.0 + GAMMA_RATIO * PRESSURE_RATIO)
VELOCITY_FACTOR = np.sqrt(2.0 * GAMMA / (GAMMA + 1.0) / SHOCK_FACTOR)
INITIAL_VELOCITY_SHOCKED = (PRESSURE_RATIO - 1.0) * VELOCITY_FACTOR / GAMMA

# Initial states for shocked and unshocked regions
DENSITY_SHOCKED = INITIAL_DENSITY_SHOCKED
VELOCITY_SHOCKED = INITIAL_VELOCITY_SHOCKED
PRESSURE_SHOCKED = PRESSURE_RATIO

DENSITY_UNSHOCKED = 1.0
VELOCITY_UNSHOCKED = 0.0
PRESSURE_UNSHOCKED = 1.0


def initialize_state():
    """
    Initialize the state variable Q representing the shock conditions.

    Returns:
        Q (np.ndarray): Initialized state array with density, momentum, and energy.
    """
    Q = np.zeros([JMAX, 3])

    # Shocked region (left side of the domain)
    Q[x <= XMID, 0] = DENSITY_SHOCKED
    Q[x <= XMID, 1] = DENSITY_SHOCKED * VELOCITY_SHOCKED
    Q[x <= XMID, 2] = (PRESSURE_SHOCKED / GAMMA / (GAMMA - 1.0) + 
                       0.5 * DENSITY_SHOCKED * VELOCITY_SHOCKED**2)
    
    # Unshocked region (right side of the domain)
    Q[x > XMID, 0] = DENSITY_UNSHOCKED
    Q[x > XMID, 1] = DENSITY_UNSHOCKED * VELOCITY_UNSHOCKED
    Q[x > XMID, 2] = (PRESSURE_UNSHOCKED / GAMMA / (GAMMA - 1.0) + 
                      0.5 * DENSITY_UNSHOCKED * VELOCITY_UNSHOCKED**2)
    
    return Q

def calculate_cfl(Q):
    """
    Calculate the Courant-Friedrichs-Lewy (CFL) condition for numerical stability.

    Args:
        Q (np.ndarray): State array with density, momentum, and energy.

    Returns:
        float: Maximum CFL condition for the current state.
    """
    density, momentum, energy = Q[:, 0], Q[:, 1], Q[:, 2]
    
    velocity = momentum / density
    pressure = (GAMMA - 1.0) * (energy - 0.5 * density * velocity ** 2)
    
    sound_speed = np.sqrt(GAMMA * pressure / density)
    max_speed = sound_speed + np.abs(velocity)
    return np.max(max_speed) * DT_DX_RATIO

def Roe_FDS_flux(QL, QR, E):
    """
    Calculate Roe flux for the finite difference scheme.

    Args:
        QL, QR (np.ndarray): Left and right states at each interface.
        E (np.ndarray): Flux array to be filled with computed values.
    """
    for j in range(JMAX - 1):        
        rhoL, uL, pL = QL[j, 0], QL[j, 1], QL[j, 2]
        rhoR, uR, pR = QR[j + 1, 0], QR[j + 1, 1], QR[j + 1, 2]
        
        rhouL = rhoL * uL
        rhouR = rhoR * uR

        eL = pL / (GAMMA - 1.0) + 0.5 * rhoL * uL ** 2
        eR = pR / (GAMMA - 1.0) + 0.5 * rhoR * uR ** 2

        HL = (eL + pL) / rhoL
        HR = (eR + pR) / rhoR
        
        cL = np.sqrt((GAMMA - 1.0) * (HL - 0.5 * uL ** 2))
        cR = np.sqrt((GAMMA - 1.0) * (HR - 0.5 * uR ** 2))
                
        # Roe average
        sqrhoL = np.sqrt(rhoL)
        sqrhoR = np.sqrt(rhoR)

        rhoAVE = sqrhoL * sqrhoR
        uAVE = (sqrhoL * uL + sqrhoR * uR) / (sqrhoL + sqrhoR)
        HAVE = (sqrhoL * HL + sqrhoR * HR) / (sqrhoL + sqrhoR) 
        cAVE = np.sqrt((GAMMA - 1.0)* (HAVE - 0.5 * uAVE ** 2))
        eAVE = rhoAVE * (HAVE - cAVE ** 2 / GAMMA)
        
        dQ = np.array([rhoR - rhoL, rhoR * uR - rhoL * uL, eR - eL])
        
        Lambda = np.diag([np.abs(uAVE - cAVE), 
                          np.abs(uAVE), 
                          np.abs(uAVE + cAVE)])
        
        b1 = 0.5 * (GAMMA - 1.0) * uAVE ** 2 / cAVE ** 2
        b2 = (GAMMA - 1.0) / cAVE ** 2

        R = np.array([[1.0,              1.0,                1.0],
                      [uAVE - cAVE,       uAVE,        uAVE + cAVE],
                      [HAVE - uAVE * cAVE, 0.5 * uAVE ** 2, HAVE + uAVE * cAVE]])
        
        Rinv = np.array([[0.5 * (b1 + uAVE / cAVE), -0.5 * (b2 * uAVE + cAVE), 0.5 * b2],
                         [1.0 - b1,                 b2 * uAVE,      -b2],
                         [0.5 * (b1 - uAVE / cAVE), -0.5 * (b2 * uAVE - cAVE), 0.5 * b2]])
        
        AQ = R @ Lambda @ Rinv @ dQ
        
        EL = np.array([rhoL * uL, pL + rhouL * uL, (eL + pL) * uL])
        ER = np.array([rhoR * uR, pR + rhouR * uR, (eR + pR) * uR])
        
        E[j] = 0.5 * (ER + EL - AQ)

def minmod(x, y):
    """
    Minmod limiter function.

    Args:
        x, y (float): Input values for the limiter.

    Returns:
        float: Limited value.
    """
    sgn = np.sign(x)
    return sgn * np.maximum(np.minimum(np.abs(x), np.abs(y)), 0.0)

def MUSCL(Q, order, kappa):
    """
    Perform MUSCL reconstruction.

    Args:
        Q (np.ndarray): State array.
        order (int): Order of the scheme (1, 2, or 3).
        kappa (float): Kappa parameter for biasing.

    Returns:
        QL, QR (np.ndarray): Reconstructed left and right states.
    """
    rho, rhou, e = Q[:, 0], Q[:, 1], Q[:, 2]
    
    Q[:, 1] = rhou / rho  # u
    Q[:, 2] = (GAMMA - 1.0) * (e - 0.5 * rho * Q[:, 1] ** 2)  # p
    
    if order == 2 or order == 3:
        # 2nd / 3rd order & minmod limiter
        dQ = np.zeros([JMAX, 3])
        for j in range(JMAX - 1):
            dQ[j] = Q[j+1] - Q[j]
        
        b = (3.0 - kappa) / (1.0 - kappa) 
        
        Dp = np.zeros([JMAX, 3])
        Dm = np.zeros([JMAX, 3])
        for j in range(1, JMAX - 1):
            Dp[j] = minmod(dQ[j], b * dQ[j - 1])
            Dm[j] = minmod(dQ[j-1], b * dQ[j])    
        Dp[0] = Dp[1]
        Dm[0] = Dm[1]
        
        QL = Q.copy()
        QR = Q.copy()
        for j in range(1, JMAX - 1):
            QL[j] += 0.25 * ((1.0 - kappa) * Dp[j] + (1.0 + kappa) * Dm[j]) 
            QR[j] -= 0.25 * ((1.0 + kappa) * Dp[j] + (1.0 - kappa) * Dm[j])
        
    else:
        # 1st order
        QL = Q.copy()
        QR = Q.copy()

    return QL, QR

def compute_flux(Q, flux):
    """
    Compute the flux vector based on the current state Q.

    Args:
        Q (np.ndarray): State array with density, momentum, and energy.
        flux (np.ndarray): Array to store the computed flux values.
    """
    density, momentum, energy = Q[:, 0], Q[:, 1], Q[:, 2]
    
    velocity = momentum / density
    pressure = (GAMMA - 1.0) * (energy - 0.5 * momentum * velocity) * GAMMA

    flux[:, 0] = momentum
    flux[:, 1] = pressure / GAMMA + momentum * velocity
    flux[:, 2] = (pressure / (GAMMA - 1.0) + 0.5 * velocity * momentum) * velocity

def lax_wendroff(Q, num_steps, print_interval=2, use_central_diff=True):
    """
    Perform the Lax-Wendroff scheme for simulating shock propagation.

    Args:
        Q (np.ndarray): State array with density, momentum, and energy.
        num_steps (int): Number of time steps to run the simulation.
        print_interval (int): Interval at which to print the CFL condition.
        use_central_diff (bool): Whether to use central difference for artificial viscosity.
    """
    flux = np.zeros([JMAX, 3])

    for step in range(num_steps):
        if step % print_interval == 0:
            print(f'Step {step : 4d} : CFL = {calculate_cfl(Q) : .4f}')

        Qs = Q.copy()

        compute_flux(Q, flux)
        for j in range(1, JMAX - 1):
            Qs[j] = 0.5 * (Q[j-1] + Q[j]) - 0.5 * DT_DX_RATIO * (flux[j] - flux[j-1])

        compute_flux(Qs, flux)
        for j in range(1, JMAX - 2):
            Q[j] = Q[j] - DT_DX_RATIO * (flux[j + 1] - flux[j])

        if use_central_diff:
            apply_central_difference(Q)

def maccormack(Q, num_steps, print_interval=2, use_central_diff=True):
    """
    Perform the MacCormack scheme for simulating shock propagation.

    Args:
        Q (np.ndarray): State array with density, momentum, and energy.
        num_steps (int): Number of time steps to run the simulation.
        print_interval (int): Interval at which to print the CFL condition.
        use_central_diff (bool): Whether to use central difference for artificial viscosity.
    """
    flux = np.zeros([JMAX, 3])

    for step in range(num_steps):
        if step % print_interval == 0:
            print(f'Step {step : 4d} : CFL = {calculate_cfl(Q) : .4f}')

        Qs = Q.copy()

        compute_flux(Q, flux)
        for j in range(1, JMAX - 1):
            Qs[j] = Q[j] - DT_DX_RATIO * (flux[j] - flux[j-1])

        compute_flux(Qs, flux)
        for j in range(1, JMAX - 2):
            Q[j] = 0.5 * (Q[j] + Qs[j]) - 0.5 * DT_DX_RATIO * (flux[j + 1] - flux[j])

        if use_central_diff:
            apply_central_difference(Q)

def apply_central_difference(Q):
    """
    Apply central difference for artificial viscosity.

    Args:
        Q (np.ndarray): State array with density, momentum, and energy.
    """
    epsilon_c = 3.725  # Coefficient for central difference
    Qb = Q.copy()
    for j in range(1, JMAX - 1):
        diff1 = Qb[j - 1] - 2.0 * Qb[j] + Qb[j + 1]
        diff2 = Qb[j - 1] + 2.0 * Qb[j] + Qb[j + 1] 
        k = epsilon_c * np.linalg.norm(diff1) / np.linalg.norm(diff2)
        Q[j] += k * diff1

def Roe_FDS(Q, order, kappa, nmax, print_interval = 2):
    """
    Perform the Roe finite difference scheme for shock propagation.

    Args:
        Q (np.ndarray): State array with density, momentum, and energy.
        order (int): Order of the scheme (1, 2, or 3).
        kappa (float): Kappa parameter for biasing.
        nmax (int): Maximum number of time steps.
        print_interval (int): Interval at which to print the CFL condition.
    """
    E = np.zeros([JMAX, 3])

    for n in range(nmax):
        if n % print_interval == 0:
            print(f'n = {n : 4d} : CFL = {calculate_cfl(Q) : .4f}')

        Qold = Q.copy()
        
        coefs = [0.5, 1.0]
        for coef in coefs:
            QL, QR = MUSCL(Qold, order, kappa)

            Roe_FDS_flux(QL, QR, E)
            for j in range(1, JMAX - 1):
                Qold[j] = Q[j] - coef * DT_DX_RATIO * (E[j] - E[j-1])
            Qold[0] = Q[0]
            Qold[-1] = Q[-1]
            
        Q[:] = Qold[:]
        

def Roe_flux(wL, wR):
    """
    Use the Roe approximate Riemann solver to calculate fluxes.
    """
    uL = w2u(wL)
    uR = w2u(wR)
        
    # Primitive and other variables.
    # Left state
    rhoL = wL[0]
    vL = wL[1]
    pL = wL[2]
    eL = pL / (GAMMA - 1.0) + 0.5 * rhoL * vL ** 2

    # Right state
    rhoR = wR[0]
    vR = wR[1]
    pR = wR[2]
    eR = pR / (GAMMA - 1.0) + 0.5 * rhoR * vR ** 2

    HL = (eL + pL) / rhoL
    HR = (eR + pR) / rhoR

    aL = np.sqrt((GAMMA - 1.0) * (HL - 0.5 * vL ** 2))
    aR = np.sqrt((GAMMA - 1.0) * (HR - 0.5 * vR ** 2))
                
    # First compute the Roe Averages
    RT = np.sqrt(rhoR/rhoL)
    rho = RT*rhoL
    v = (vL+RT*vR)/(1.0+RT)
    H = (HL+RT*HR)/(1.0+RT)
    a = np.sqrt( (GAMMA-1.0)*(H-0.5*v*v) )

    # Differences in primitive variables.
    drho = rhoR - rhoL
    du = vR - vL
    dP = pR - pL

    # Wave strength (Characteristic Variables).
    dV = np.array([0.0,0.0,0.0])
    dV[0] = 0.5*(dP-rho*a*du)/(a*a)
    dV[1] = -( dP/(a*a) - drho )
    dV[2] = 0.5*(dP+rho*a*du)/(a*a)

    # Absolute values of the wave speeds (Eigenvalues)
    ws = np.array([0.0,0.0,0.0])
    ws[0] = abs(v-a)
    ws[1] = abs(v)
    ws[2] = abs(v+a)

    # Modified wave speeds for nonlinear fields (the so-called entropy fix, which
    # is often implemented to remove non-physical expansion shocks).
    # There are various ways to implement the entropy fix. This is just one
    # example. Try turn this off. The solution may be more accurate.
    Da = max(0.0, 4.0*((vR-aR)-(vL-aL)) )
    if (ws[0] < 0.5*Da):
        ws[0] = ws[0]*ws[0]/Da + 0.25*Da
    Da = max(0.0, 4.0*((vR+aR)-(vL+aL)) )
    if (ws[2] < 0.5*Da):
        ws[2] = ws[2]*ws[2]/Da + 0.25*Da

    # Right eigenvectors
    R = np.zeros((3,3))

    R[0][0] = 1.0
    R[1][0] = v - a
    R[2][0] = H - v*a

    R[0][1] = 1.0
    R[1][1] = v
    R[2][1] = 0.5*v*v

    R[0][2] = 1.0
    R[1][2] = v + a
    R[2][2] = H + v*a

    # Compute the average flux.
    flux = 0.5*( euler_flux(wL) + euler_flux(wR) )

    # Add the matrix dissipation term to complete the Roe flux.
    for i in range(0,3):
        for j in range(0,3):
            flux[i] = flux[i] - 0.5*ws[j]*dV[j]*R[i][j]
    return flux

def euler_flux(w):
    """
    Calculate the conservative Euler fluxes.
    """
    rho = w[0]
    u = w[1]
    p = w[2]
    e = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2

    f_1 = rho*u
    f_2 = p + f_1 * u
    # f_3 = (p / (GAMMA - 1.0) + 0.5 * u * f_1) * u
    f_3 = (e + p) * u
    
    return np.array([f_1, f_2, f_3])


def w2u(w):
    """
    Convert the primitive to conservative variables.
    """
    u = np.zeros(3)
    u[0] = w[0]
    u[1] = w[0]*w[1]
    u[2] = w[2]/(GAMMA-1.0)+0.5*w[0]*w[1]*w[1]
    return u
    
def u2w(u):
    """
    Convert the conservative to primitive variables.
    """

    w = np.zeros(3)
    w[0] = u[0]
    w[1] = u[1]/u[0]
    w[2] = (GAMMA-1.0)*( u[2] - 0.5*w[0]*w[1]*w[1] )
    return w

def Roe(U, dt, dx, JMAX):
    """
    Updates the solution of the equation
    via the Godunov procedure.
    """
    E = np.zeros([JMAX, 3])
    
    for n in range(0, NUM_STEPS):
        if n % PRINT_INTERVAL == 0:
            print(f'n = {n : 4d} : CFL = {calculate_cfl(U) : .4f}')
            
        U_old = U.copy()
        
        # Create fluxes
        for i in range(0, JMAX-1):
            wL = u2w(U_old[i])
            wR = u2w(U_old[i+1])
            E[i] = Roe_flux(wL, wR)

        # Update solution
        for i in range(1, JMAX-1):
            U_old[i] = U[i] - DT_DX_RATIO * (E[i]-E[i-1])
        U[:] = U_old[:]
        U[0] = U[0]
        U[-1] = U[-1]

def plot_results(x, exact_velocity, Q_LW, Q_LWAV, Q_Mac, Q_Roe_1st, Q_Roe_2nd, Q_Roe_3rd, Q_Roe):
    """
    Plot the results of the shock propagation simulations.

    Args:
        x (np.ndarray): Spatial domain array.
        exact_velocity (np.ndarray): Analytical solution for velocity.
        Q_LW (np.ndarray): Lax-Wendroff simulation results.
        Q_LWAV (np.ndarray): Lax-Wendroff with Artificial Viscosity results.
        Q_Mac (np.ndarray): MacCormack simulation results.
        Q_Roe_1st (np.ndarray): Roe 1st order upwind simulation results.
        Q_Roe-FDS_1st (np.ndarray): Roe-FDS 1st order upwind simulation results.
        Q_Roe-FDS_2nd (np.ndarray): MUSCL-Roe-FDS 2nd order upwind simulation results.
        Q_Roe-FDS_3rd (np.ndarray): MUSCL-Roe-FDS 3rd order upwind simulation results.
    """
    # Define file paths
    shock_data_file_FCT = os.path.join(DATA_DIR, 'shock_data_FCT_101.dat')

    # Read the data from the Fortran binary files
    shock_data_FCT = read_fortran_binary(shock_data_file_FCT, JMAX)

    plt.rcParams['font.weight'] = 'bold'  # Font weight (normal, bold, etc.)
    plt.rcParams['axes.titleweight'] = 'bold'  # Title font weight
    plt.rcParams['axes.labelweight'] = 'bold'  # Axis label font weight
    

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    palette = sns.color_palette("husl", 9)
    markers = ['o', 's', 'D', '^', 'v', 'x', 'P', '*', 'h']
    fig.suptitle('Shock Propagation', fontsize=24, weight='bold')

    ax = axs[0]
    sns.lineplot(x=x, y=exact_velocity, label='Analytic solution', color='black', linestyle='dashed', linewidth=2, ax=ax)
    sns.lineplot(x=x, y=Q_LW[:, 1] / Q_LW[:, 0], label='Lax-Wendroff (2nd-center)', color=palette[0], linewidth=2,
                    marker=markers[0], ax=ax)
    sns.lineplot(x=x, y=Q_LWAV[:, 1] / Q_LWAV[:, 0], label='Lax-Wendroff + Art. Viscosity (2nd-center)', color=palette[1], linewidth=2,
                marker=markers[1], ax=ax)
    sns.lineplot(x=x, y=shock_data_FCT, label='Lax-Wendroff + FCT (2nd-center)', color=palette[2], linestyle='-', marker=markers[7], 
                linewidth=2, ax=ax)
    sns.lineplot(x=x, y=Q_Mac[:, 1] / Q_Mac[:, 0], label='MacCormack (2nd-center)', color=palette[3], linewidth=2,
                marker=markers[2], ax=ax)
    
    ax.set_xlabel(r'$x$', fontsize=14, weight='bold')
    ax.set_ylabel(r'$u$', fontsize=14, weight='bold')
    ax.set_xlim(0.60, 0.88)
    ax.set_ylim(-0.05, 0.9)
    # ax.legend(fontsize=12, title_fontsize='16', title='Methods')
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax = axs[1]
    sns.lineplot(x=x, y=exact_velocity, label='Analytic solution', color='black', linestyle='dashed', linewidth=2, ax=ax)
    sns.lineplot(x=x, y=Q_Roe[:, 1] / Q_Roe[:, 0], label='Roe (1st-upwind)', color=palette[4], linewidth=2,
                marker=markers[6], ax=ax)
    sns.lineplot(x=x, y=Q_Roe_1st[:, 1] / Q_Roe_1st[:, 0], label='Roe-FDS (1st-upwind)', color=palette[5], linewidth=2,
                marker=markers[3], ax=ax)
    sns.lineplot(x=x, y=Q_Roe_2nd[:, 1] / Q_Roe_2nd[:, 0], label='MUSCL-Roe-FDS (2nd-upwind)', color=palette[6], linewidth=2,
                marker=markers[4], ax=ax)
    sns.lineplot(x=x, y=Q_Roe_3rd[:, 1] / Q_Roe_3rd[:, 0], label='MUSCL-Roe-FDS (3rd-upwind)', color=palette[7], linewidth=2,
                marker=markers[5], ax=ax)

    ax.set_xlabel(r'$x$', fontsize=14, weight='bold')
    ax.set_ylabel(r'$u$', fontsize=14, weight='bold')
    ax.set_xlim(0.60, 0.88)
    ax.set_ylim(-0.05, 0.9)
    # ax.legend(fontsize=12, title_fontsize='16', title='Methods')
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR+'Shock_Propagation_101.pdf', format='pdf', dpi=600)

 
           
# Main execution

PRINT_INTERVAL = 4

Q_LW = initialize_state()
lax_wendroff(Q_LW, NUM_STEPS, PRINT_INTERVAL, use_central_diff=False)

Q_LWAV = initialize_state()
lax_wendroff(Q_LWAV, NUM_STEPS, PRINT_INTERVAL, use_central_diff=True)

Q_Mac = initialize_state()
maccormack(Q_Mac, NUM_STEPS, PRINT_INTERVAL, use_central_diff=False)

order = 2
kappa = 0.
Q_Roe_MC_2nd_upwind = initialize_state()
Roe_FDS(Q_Roe_MC_2nd_upwind, order, kappa, NUM_STEPS, PRINT_INTERVAL)

order = 3
kappa = 1./3.
Q_Roe_MC_3rd_upwind = initialize_state()
Roe_FDS(Q_Roe_MC_3rd_upwind, order, kappa, NUM_STEPS, PRINT_INTERVAL)

order = 1
kappa = 0.
Q_Roe_1st_upwind = initialize_state()
Roe_FDS(Q_Roe_1st_upwind, order, kappa, NUM_STEPS, PRINT_INTERVAL)

Q_Roe = initialize_state()
Roe(Q_Roe, TIME_STEP, DX, JMAX)


# Exact solution calculation
time_elapsed = NUM_STEPS * TIME_STEP
shock_speed = 0.5 * ((GAMMA - 1.0) + (GAMMA + 1.0) * PRESSURE_RATIO) / GAMMA
shock_speed = np.sqrt(shock_speed)
shock_position = 0.501 + shock_speed * time_elapsed
shock_index = int(shock_position / DX + 1.0)

exact_velocity = np.zeros(JMAX)
for j in range(shock_index):
    exact_velocity[j] = VELOCITY_SHOCKED
for j in range(shock_index, JMAX):
    exact_velocity[j] = VELOCITY_UNSHOCKED

# Plot the results
plot_results(x, exact_velocity, Q_LW, Q_LWAV, Q_Mac, Q_Roe_1st_upwind, Q_Roe_MC_2nd_upwind, Q_Roe_MC_3rd_upwind, Q_Roe)