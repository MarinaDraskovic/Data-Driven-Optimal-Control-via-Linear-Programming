import torch
from torch import tensor
from deterministicAffineSystems import DeterministicAffineSystem
from stochasticAffineSystems import StochasticAffineSystem
from linearProgramStochastic import LinearProgramStochastic
from deterministicAffineSystems import Cost
import sys
import numpy
import matplotlib.pyplot as plt
from statistics import mean
from numpy import matlib

import random
##### Random seed given
random.seed(42)


#==============================================================================
#______Additional functions___________________________________________________
#==============================================================================
def calculate_objective(LP, Q, Q_L, e): # Calculates Ec = tr(Q*Sigma_c) + Q_L'*mu_c + e
    dimension = Q.size(0)               # Q dim (n+m)x(n+m), Q_L dim (n+m)x1, e dim 1x1 
                                                
                                        # c1*Q_flat= flat(sigma.T) * Q_flat = tr(Sigma*Q)
    Qflat = torch.reshape(Q, (dimension**2,1))
                                            
    c1 = torch.reshape(LP.sigma_c.T, (1, dimension**2)).double()
    c2 = LP.mu_c.T
    c3 = torch.tensor([[1]])
    
    objective = c1@Qflat + c2.double()@Q_L + c3.double()@e
    return objective.item()            # Returns double

#==============================================================================
#________Defining parameters for simple affine system_________________________
#==============================================================================
                                       # System dynamics: x^+ = A*x + B*u + b + xi
                                       # A dim n x n
A = tensor([[1, 0.1], [0.5, -0.5]]).double()         
B = tensor([[1], [0.5]]).double()      # dim n x m
b = tensor([[0.5], [-0.3]]).double()    # dim n x 1

                                       # Cost: [x;u]'*[Lxx, Lxu; Lxu', Luu]*[x;u]+
                                       #       + 2[x;u]'*[lx;lu] + l11
                                       # Lxx dim n x n
Lxx = tensor([[1, 0],[0, 1]]).double()
Lxu = tensor([[0.1], [0.1]]).double()  # dim n x m
Luu = tensor([[0.01]]).double()        # dim m x m
lx = tensor([[0.1],[0.1]]).double()    # dim n x 1
lu = tensor([[0.1]]).double()          # dim m x 1
l11 = tensor([[0.1]]).double()         # dim 1 x 1 

gamma = 0.95                           # Discount factor. 0.95->0.99

lb_x = -3                              # Lower and upper bounds on x and u (x and u uniform distribution)
ub_x = 3
lb_u = -1
ub_u = 1

n = A.size(0)                          # dim(x)=nx1, dim(u)=mx1
m = B.size(1)

mu_c = torch.zeros(n+m,1).double()     # Constant c for solving multidimensional integral
sigma_cn = torch.eye(n).double()
sigma_cm = 0.1*torch.eye(m).double()
sigma_c = torch.block_diag(sigma_cn, sigma_cm).double()
                                       # For stochastic system:
mu_xi = 0.05                             # Mean of the disturbance
sigma_xi =0.000001                    # Variance of the disturbance

no_xi_samples = 100                 
#==============================================================================
#________Check if parameters make sense_______________________________________
#==============================================================================
try:                                         
                                       # Checking if all dimensions are correct
    if A.size(0)!=A.size(1): raise Exception('A must be a square matrix')
    elif B.size(0)!=n: raise Exception('B must be of dimensions ' +n+' x ' +m)
    elif b.size(0)!=n or b.size(1)!=1: raise Exception('b must be of dimensions '+n+' x 1')                                  
    elif Lxx.size(0)!=n or Lxx.size(1)!=n: raise Exception('Lxx must be of dimensions '+n+' x '+n)
    elif Lxu.size(0)!=n or Lxu.size(1)!=m: raise Exception('Lxu must be of dimensions '+n+' x '+m)
    elif Luu.size(0)!=m or Luu.size(1)!=m: raise Exception('Luu must be of dimensions '+m+' x '+m)
    elif lx.size(0)!=n or lx.size(1)!=1: raise Exception('lx must be of dimensions '+n+' x 1')
    elif lu.size(0)!=m or lu.size(1)!=1: raise Exception('lu must be of dimensions '+m+' x 1')
    
                                       # Check if Lxx and Luu are symmetric (They need to be pd)
    if not torch.equal(Lxx,Lxx.T): raise Exception('Lxx needs to be symmetric.')
    if not torch.equal(Luu,Luu.T): raise Exception('Luu needs to be symmetric.')
    
                                       # Check if gamma is in wanted range
    if gamma<=0 or gamma>=1: raise Exception('gamma must be in (0,1)')
except:                                # If something is incorrect, exit the program
    sys.exit(1)
    
try:                                   # Check if Lxx and Luu are pd    
    torch.cholesky(Lxx.double())
except:
    print('Lxx needs to be positive definite.')
    sys.exit(1)
    
try:
    torch.cholesky(Luu)
except:
    print('Luu needs to be positive definite.')
    sys.exit(1)    
    
#==============================================================================
#________Simple stochastic affine system_______________________________________
#==============================================================================
                                    # Create an instance of AffineSystem class 
L = Cost(Lxx, Lxu, Luu, lx, lu, l11)
system = StochasticAffineSystem(A, B, b, mu_xi, sigma_xi, L, gamma, lb_x, ub_x, lb_u, ub_u)

LP = LinearProgramStochastic(system, mu_c, sigma_c)
no_constraints = [3000,5000,7500,10000,15000,17500,20000,30000,50000]

number_draws = 100                     # Number of draws of xi = 100
number_runs = 10                       # Number of identical runs = 10

flag_success = 1;

analytic_SQ, analytic_SQ_L, analytic_Se = system.relax_q_analytic_solution()
analytic_objective = calculate_objective(LP, analytic_SQ, analytic_SQ_L, analytic_Se)
optimal_policy = system.analytic_opt_policy_coefficients()

analytic_SQ_not_relaxed, analytic_SQ_L_not_relaxed, analytic_Se_not_relaxed = system.q_analytic_solution()
analytic_objective_not_relaxed = calculate_objective(LP, analytic_SQ_not_relaxed, analytic_SQ_L_not_relaxed, analytic_Se)

#______________________Plots for affine system_________________________________   
y_values = []                          # Initializing y axis on the plot
y_min = []
y_max = []
runtime = []
runtime_min = []
runtime_max = []
   
for c in no_constraints:               # Call solver with different number of constraints
    y = []
    run = []
    for r in range(number_runs):
        objective, LP_Q, LP_Q_L, LP_e, rt,_,_ = LP.solve_directly(c, number_draws)
        if objective is not None:      # Add results to an array that will be plotted
            difference = abs(analytic_objective - objective)
            y.append(difference)
            run.append(rt)
    y_values.append(mean(y))
    y_min.append(min(y))
    y_max.append(max(y))   
    runtime.append(mean(run))
    runtime_min.append(min(run))
    runtime_max.append(max(run))       
           
                                        # Check if solving LP was successful
if len(no_constraints) == len(y_values) and len(no_constraints) == len(runtime):    
    fig1 = plt.figure(1)
    plt.plot(no_constraints, y_values, '-o', color='#580F41')
    plt.fill_between(no_constraints, y_min, y_max, color= '#C79FEF')
    plt.grid(True)
    plt.xlabel('Number of constraints')
    plt.ylabel("$\mathbb{E}_{c}\hat{q} - \mathbb{E}_{c}q$")
    plt.yscale('log')
    plt.legend()
    
    fig2 = plt.figure(2)
    plt.plot(no_constraints, runtime, '-o', color='#580F41')
    plt.fill_between(no_constraints, runtime_min, runtime_max, color='#C79FEF')
    plt.axis([0, max(no_constraints)+2000, 0, max(runtime_max)+0.02])
    plt.grid(True)
    plt.xlabel('Number of constraints')
    plt.ylabel('Time [s]')

    plt.show()
    
    fig1.savefig('fig11.pdf')
    fig2.savefig('fig12.pdf')
else:
    print('Plotting performance and time vs constraints unsuccessful. LP not successfully solved for all #constrains.')


#==============================================================================
#________________________________Comparing PIs ________________________________
#==============================================================================
treshold = 0.0001
no_constraints = 1500
number_draws = 100                       # Number of draws of xi = 100
number_iterations = 9

LP = LinearProgramStochastic(system, mu_c, sigma_c)
y_values = []                            # Initializing y axis on the plot
y_min = []
y_max = []
runtime = []
runtime_min = []
runtime_max = []

#_______________Running PI for different iterations number_____________________
analytic_objectives = numpy.matlib.repmat(analytic_objective, 1,number_iterations)
analytic_objectives_not_relaxed = numpy.matlib.repmat(analytic_objective_not_relaxed, 1,number_iterations)

PI_objectives1,PI_Q,PI_Q_L,PI_e,rt1,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives2,PI_Q,PI_Q_L,PI_e,rt2,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives3,PI_Q,PI_Q_L,PI_e,rt3,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives4,PI_Q,PI_Q_L,PI_e,rt4,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives5,PI_Q,PI_Q_L,PI_e,rt5,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives6,PI_Q,PI_Q_L,PI_e,rt6,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives7,PI_Q,PI_Q_L,PI_e,rt7,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives8,PI_Q,PI_Q_L,PI_e,rt8,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives9,PI_Q,PI_Q_L,PI_e,rt9,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)
PI_objectives0,PI_Q,PI_Q_L,PI_e,rt0,PI_K1,PI_K2 = LP.PI_on_policy(LP,no_constraints,number_draws,number_iterations)

difference1 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives1))
difference2 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives2))
difference3 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives3))
difference4 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives4))
difference5 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives5))
difference6 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives6))
difference7 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives7))
difference8 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives8))
difference9 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives9))
difference0 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives0))


y=[difference1,difference2,difference3,difference4,difference5,difference6,difference7,difference8,difference9,difference0]
run = [rt1, rt2, rt3, rt4, rt5, rt6, rt7, rt8, rt9, rt0]

y_values=numpy.mean(y, axis=0)
y_min=numpy.min(y, axis=0)
y_max=numpy.max(y, axis=0)

run_values=numpy.mean(run, axis=0)
run_min=numpy.min(run, axis=0)
run_max=numpy.max(run, axis=0)  

# __________With other number of draws of w
y2=[]
run2=[]
PI_objectives1,PI_Q,PI_Q_L,PI_e,rt1,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives2,PI_Q,PI_Q_L,PI_e,rt2,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives3,PI_Q,PI_Q_L,PI_e,rt3,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives4,PI_Q,PI_Q_L,PI_e,rt4,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives5,PI_Q,PI_Q_L,PI_e,rt5,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives6,PI_Q,PI_Q_L,PI_e,rt6,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives7,PI_Q,PI_Q_L,PI_e,rt7,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives8,PI_Q,PI_Q_L,PI_e,rt8,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives9,PI_Q,PI_Q_L,PI_e,rt9,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)
PI_objectives0,PI_Q,PI_Q_L,PI_e,rt0,PI_K1,PI_K2 = LP.PI_on_policy_5w(LP,no_constraints,number_draws,number_iterations)

difference1 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives1))
difference2 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives2))
difference3 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives3))
difference4 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives4))
difference5 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives5))
difference6 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives6))
difference7 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives7))
difference8 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives8))
difference9 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives9))
difference0 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives0))


y2=[difference1,difference2,difference3,difference4,difference5,difference6,difference7,difference8,difference9,difference0]
run2 = [rt1, rt2, rt3, rt4, rt5, rt6, rt7, rt8, rt9, rt0]

y2_values=numpy.mean(y2, axis=0)
y2_min=numpy.min(y2, axis=0)
y2_max=numpy.max(y2, axis=0)

run2_values=numpy.mean(run2, axis=0)
run2_min=numpy.min(run2, axis=0)
run2_max=numpy.max(run2, axis=0) 


# For third line
y3=[]
run3=[]
PI_objectives1,PI_Q,PI_Q_L,PI_e,rt1,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives2,PI_Q,PI_Q_L,PI_e,rt2,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives3,PI_Q,PI_Q_L,PI_e,rt3,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives4,PI_Q,PI_Q_L,PI_e,rt4,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives5,PI_Q,PI_Q_L,PI_e,rt5,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives6,PI_Q,PI_Q_L,PI_e,rt6,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives7,PI_Q,PI_Q_L,PI_e,rt7,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives8,PI_Q,PI_Q_L,PI_e,rt8,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives9,PI_Q,PI_Q_L,PI_e,rt9,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)
PI_objectives0,PI_Q,PI_Q_L,PI_e,rt0,PI_K1,PI_K2 = LP.PI_on_policy_10w(LP,no_constraints,number_draws,number_iterations)

difference1 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives1))
difference2 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives2))
difference3 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives3))
difference4 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives4))
difference5 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives5))
difference6 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives6))
difference7 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives7))
difference8 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives8))
difference9 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives9))
difference0 = abs(analytic_objectives_not_relaxed - numpy.array(PI_objectives0))

y3=[difference1,difference2,difference3,difference4,difference5,difference6,difference7,difference8,difference9,difference0]
run3 = [rt1, rt2, rt3, rt4, rt5, rt6, rt7, rt8, rt9, rt0]

y3_values=numpy.mean(y3, axis=0)
y3_min=numpy.min(y3, axis=0)
y3_max=numpy.max(y3, axis=0)

run3_values=numpy.mean(run3, axis=0)
run3_min=numpy.min(run3, axis=0)
run3_max=numpy.max(run3, axis=0)     



#_________________Plotting results_____________________________________________    
fig1 = plt.figure(1)  # Plotting difference between objective functions analytic solution and the solution that PI returned
plt.plot(range(1,number_iterations+1), y_values[0,:], '-o',color='#006400', label='draws of $\omega$ = 1')  
plt.fill_between(range(1,number_iterations+1), y_min[0,:], y_max[0,:], color= '#00FF00', alpha=0.5)

plt.plot(range(1,number_iterations+1), y2_values[0,:], '-o',color='#F97306', label='draws of $\omega$ = 5') 
plt.fill_between(range(1,number_iterations+1), y2_min[0,:], y2_max[0,:], color= '#FFA500', alpha=0.5)

plt.plot(range(1,number_iterations+1), y3_values[0,:], '-o',color='#0343df', label='draws of $\omega$ = 10')
plt.fill_between(range(1,number_iterations+1), y3_min[0,:], y3_max[0,:], color= '#95d0fc', alpha=0.5)
     
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel("$\mathbb{E}_{c}q^* - \mathbb{E}_{c}q$")
plt.legend()
plt.yscale('log')

# PLOT TIME
fig2 = plt.figure(2)

plt.plot(range(1,number_iterations+1), run_values,'-o',color='#006400' , label='draws of $\omega$ = 1')  
plt.fill_between(range(1,number_iterations+1), run_min, run_max, color= '#00FF00', alpha=0.5)

plt.plot(range(1,number_iterations+1), run2_values, '-o',color='#F97306', label='draws of $\omega$ = 5') 
plt.fill_between(range(1,number_iterations+1), run2_min, run2_max, color= '#FFA500', alpha=0.5)

plt.plot(range(1,number_iterations+1), run3_values, '-o',color='#0343df', label='draws of $\omega$ = 10') 
plt.fill_between(range(1,number_iterations+1), run3_min, run3_max, color= '#95d0fc', alpha=0.5)

plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel("$Time [s]$")
plt.legend()

plt.show()

fig1.savefig('fig81.pdf')
fig2.savefig('fig82.pdf')


#==============================================================================
#__Stochastic sys. with increasing number of states generated with Erdos Renyi__
#==============================================================================
                                        # Parameters that dont change for all systems  
gamma = 0.95
lb_x = -0.5                            # Lower and upper bounds on x and u (x and u uniform distribution)
ub_x = 0.5
lb_u = -3
ub_u = 3
no_constraints = 10000
number_draws = 100
    
N = [2,3,4,5,6,7,8]#,9,10]             # Different number of states
m = 2

number_runs = 10                       # Number of identical runs
y_values = []                          # Initializing y axis on the plot
y_min = []
y_max = []
runtime = []
runtime_min = []
runtime_max = []

for n in N:                             
    A = torch.zeros(n, n)              # Creating A
    for i in range(n):      
        for j in range(n):
            if i==j:
                A[i,j] = 0.5
            else:
                A[i,j] = torch.tensor(numpy.random.choice([0,1], p=[0.1, 0.9])*numpy.random.uniform(-0.1, 0.1))
      
    B = torch.zeros(n,m)               # Creating B
    for i in range(n):
        for j in range(m):
            B[i,j] = torch.tensor(numpy.random.choice([0,1], p=[0.1, 0.9])*numpy.random.uniform(-0.1, 0.1))
    
    b = torch.zeros(n,1)               # Creating b
    for i in range(n):
        b[i] = numpy.random.choice([0,1], p=[0.1, 0.9])*numpy.random.uniform(-0.1, 0.1)
    
    A = A.double()
    B = B.double()
    b = b.double()
                                        # Creating cost:
    Lxx = torch.eye(n,n).double()*0.01     # dim n x n
    Lxu = torch.ones(n,m).double()*0.001         # dim n x m, m=2
    Luu = torch.eye(m,m).double()*0.01     # dim m x m
    lx = torch.ones(n,1).double()*0.001           # dim n x 1
    lu = torch.ones(m,1).double()*0.001           # dim m x 1
    l11 = torch.ones(1,1).double()*0.001          # dim 1 x 1 
                                        # Create system and LP
    L = Cost(Lxx, Lxu, Luu, lx, lu, l11)
    system = StochasticAffineSystem(A, B, b, mu_xi, sigma_xi, L, gamma, lb_x, ub_x, lb_u, ub_u)
    
    mu_c = torch.zeros(n+m,1)          # Constant c for solving multidimensional integral
    sigma_c = 0.8*torch.eye(n+m)
    LP = LinearProgramStochastic(system, mu_c, sigma_c)
                                        # Analytic solution
    analytic_Q, analytic_Q_L, analytic_e = system.relax_q_analytic_solution()
    analytic_objective = calculate_objective(LP, analytic_Q, analytic_Q_L, analytic_e)
        
    y = []                             # LP solution     
    run = []
    for r in range(number_runs):
        objective, LP_Q, LP_Q_L, LP_e, rt,_,_ = LP.solve_directly(no_constraints, number_draws)
        if objective is not None:      # Add results to an array that will be plotted
            difference = abs(analytic_objective - objective)
            y.append(difference)
            run.append(rt)
    y_values.append(mean(y))
    y_min.append(min(y))
    y_max.append(max(y))   
    runtime.append(mean(run))
    runtime_min.append(min(run))
    runtime_max.append(max(run))

#______________Plots for different dimensionalities____________________________   
if len(N) == len(y_values) and len(N) == len(runtime):
    fig3 = plt.figure(3)
    plt.plot(N, y_values, '-o', color='#580F41')
    plt.fill_between(N, y_min, y_max, color= '#C79FEF')
    plt.axis([2, max(N), 0, max(y_max) + 0.0005])
    plt.grid(True)
    plt.xlabel('Dimension $n_x$')
    plt.ylabel("$\mathbb{E}_{c}\hat{q} - \mathbb{E}_{c}q$")
    #plt.title('Stochastic system: Performance vs dimension')
       
    fig4 = plt.figure(4)
    plt.plot(N, runtime, '-o', color='#580F41')
    plt.fill_between(N, runtime_min, runtime_max, color='#C79FEF')
    plt.axis([2, max(N), 0, max(runtime_max) ])
    plt.grid(True)
    plt.xlabel('Dimension $n_x$')
    plt.ylabel('Time [s]')
    #plt.title('Stochastic system:Time vs dimension')
    
    plt.show()
else:
    print('Plotting performance and time vs dimension unsuccessful. LP not successfully solved for all #constrains.')

fig3.savefig('fig21.pdf')
fig4.savefig('fig22.pdf')