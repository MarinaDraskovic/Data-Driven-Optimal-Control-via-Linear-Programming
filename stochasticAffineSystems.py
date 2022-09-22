from dataclasses import dataclass
import torch
from torch import matmul
from scipy import linalg
import typing
import numpy as np
import sys
from torch import transpose
import random
random.seed(42)

@dataclass
class Cost:
    Lxx: 'typing.Any' = object()
    Lxu: 'typing.Any' = object()
    Luu: 'typing.Any' = object()
    lx: 'typing.Any' = object()
    lu: 'typing.Any' = object()
    l11: 'typing.Any' = object()

class StochasticAffineSystem:
    """--> Affine system dynamics: x_k+1 = A*x_k + B*u_k + b + xi_k.
    xi is (possibly non zero) mean noise. 
    b is a constant which makes the system affine.
    --> Cost: [x;u]'*[Lxx, Lxu; Lxu', Luu]*[x;u] + 2[x;u]'*[lx;lu] + l11.
    --> Function 'oracle' simulates the system; returns x^+ and cost l.
    --> Function V_analytic_solution returns theoretical optimal value function.
    --> Function q_analytic_solution returns theoretical optimal q function.
    --> Function relax_q_analytic_solution returns theoretical optimal relaxed q function.
    --> Function analytic_opt_control returns theoretical optimal control for certain state x.
    """

    def __init__(self, A, B, b, mu_xi, sigma_xi, cost, gamma, lb_x, ub_x, lb_u, ub_u):
        self.A = A
        self.B = B
        self.b = b
        self.mu_xi = mu_xi
        self.sigma_xi = sigma_xi
        self.cost = cost
        self.gamma = gamma
        self.lb_x = lb_x
        self.ub_x = ub_x
        self.lb_u = lb_u
        self.ub_u = ub_u

    def oracle(self, x, u, no_xi_samples):  
        # Simulates interaction with the real system.
        # Returns check, x+ and cost based on pair (x,u).
        # check is -1 if the dimensions dont match.
        
        no_constraints = x.size(0)
        n = self.A.size(0)
        m = self.B.size(1)                             
        x_plus = 0                           # Initializing return values                    
        l = 0
                                             # Check if dimensions match
        if x.size(1)!=n or x.size(2)!=1: 
            print('x should be of dimensions '+n+' x 1')
            return -1, x_plus, l         
        if u.size(1)!=m or u.size(2)!=1:
            print('u should be of dimensions '+m+' x 1')
            return -1, x_plus, l            
        
                                             # xi drawn from normal distributions with
                                             # mean and std are given. std=sqrt(variance)
        xi = torch.normal(self.mu_xi, np.sqrt(self.sigma_xi), size=(no_constraints, no_xi_samples, n, 1))
        xi = xi.double();
                                             # Calculate x in the next timestep without noise
        x_plus = matmul(self.A,x) + matmul(self.B,u) + self.b
        
                                             # Add one dimension s.t. samples of xi can be added
        x_plus = torch.reshape(x_plus, (no_constraints,1,n,1))
        x_plus = x_plus.repeat(1,no_xi_samples,1,1)
        
        #xi = torch.zeros(x_plus.shape)      # For deterministic system use this xi
        x_plus = x_plus + xi                 # Add noise. Many samples of noise per one x+ vlaue
                
                                             
        temp1 = matmul(self.cost.Lxx, x)     # Calculate cost 
        temp2 = matmul(self.cost.Lxu, u)
        temp3 = matmul(self.cost.Luu, u)
        quadratic_cost = matmul(transpose(x,1,2),temp1)+2*matmul(transpose(x,1,2),temp2)+matmul(transpose(u,1,2),temp3)
        linear_cost = 2*matmul(transpose(x,1,2),self.cost.lx) + 2*matmul(transpose(u,1,2),self.cost.lu)
        l = quadratic_cost + linear_cost + self.cost.l11
            
        return 0,x_plus,l                    # 0 is indicator that calculation was successful
    
#==============================================================================    
#___________Functions that return analytic values______________________________
#==============================================================================

    def calculate_parameters(self):          
        # Returns P, Qxx, Qxu, Quu, L, since they appear in multiple calculations
        
                                            # P is a solution of DARE
                                            # Adding sqrt(gamma) to A & B to add discount factor
        P = linalg.solve_discrete_are(np.sqrt(self.gamma)*self.A, np.sqrt(self.gamma)*self.B, self.cost.Lxx, self.cost.Luu, None, self.cost.Lxu)
        P = torch.from_numpy(P).double()
                                         
        Qxx = self.cost.Lxx + self.gamma*matmul(self.A.T, matmul(P, self.A))
        Qxu = self.cost.Lxu + self.gamma*matmul(self.A.T, matmul(P, self.B))
        Quu = self.cost.Luu + self.gamma*matmul(self.B.T, matmul(P, self.B))
        
        n = self.A.size(0)
                                             # Create a variable that deals with non zero mean noise.
        b_mean = self.b + torch.ones(n,1).double()*self.mu_xi 
        
                                             # L has closed-form solution
        L = 0                                # Initialization
        try:                                 # Try block bcs computing inverses may be impossible
            temp1 = torch.linalg.inv(Quu)
            temp2 = torch.eye(n) - self.gamma*self.A.T + self.gamma*matmul(Qxu, matmul(temp1, self.B.T))
            temp3 = self.gamma * matmul(self.B.T, matmul(P, b_mean)) + self.cost.lu
            temp4 = self.cost.lx + self.gamma*matmul(self.A.T, matmul(P,b_mean)) - matmul(Qxu, matmul(temp1,temp3))
            L = matmul(torch.linalg.inv(temp2), temp4)
        except Exception as exeption: 
            print('Computing L not successful.')
            print(exeption)
            
        return P,Qxx,Qxu,Quu,L                                         
                                                  
    def V_analytic_solution(self):           
        # Returns optimal value function that is a solution to BE of form V(x)=x'Px+2x'L+q.

                                             # Getting parameters P and L
        P,_,_,Quu,L = self.calculate_parameters() 
        n = self.A.size(0)        
                                             # Create a variable that deals with non zero mean noise.
        b_mean = self.b + torch.ones(n,1).double()*self.mu_xi
        
                                             # Constant term q in value function
        temp = self.gamma*matmul(self.B.T, matmul(P, b_mean)) + self.gamma*matmul(self.B.T,L) + self.cost.lu
        temp2 = self.cost.l11 + self.gamma*matmul(b_mean.T, matmul(P, b_mean)) + 2*self.gamma*matmul(L.T,b_mean) + self.gamma*torch.trace(P*self.sigma_xi)
        q = (temp2 - matmul(temp.T, matmul(torch.linalg.inv(Quu), temp)))/(1-self.gamma)

        return P,L,q
        
    def q_analytic_solution(self):           
        # Returns optimal q function of form  q(x)=[x;u]'Q[x;u]+2[x;u]'Q_L+e
        
        P, Qxx, Qxu, Quu, P_L = self.calculate_parameters()
        
                                             # Constant Q for quadratic term
        Q_temp1 = torch.cat((Qxx, Qxu), 1)
        Q_temp2 = torch.cat((Qxu.T, Quu), 1)
        Q = torch.cat((Q_temp1, Q_temp2), 0)
        
        n = self.A.size(0)                   # Create a variable that deals with non zero mean noise
        
        b_mean = self.b + torch.ones(n,1)*self.mu_xi
        
                                             # Constant QL for linear term has closed-form solution
        qx1 = 2*(self.cost.lx + self.gamma*matmul(self.A.T, matmul(P, b_mean)) + self.gamma*matmul(self.A.T, P_L))
        qu1 = 2*(self.cost.lu + self.gamma*matmul(self.B.T, matmul(P, b_mean)) + self.gamma*matmul(self.B.T, P_L))
        Q_L = torch.cat((qx1, qu1), 0)
        
                                             # Constant term q has closed-forom solution
        temp6 = self.gamma*matmul(self.B.T, matmul(P, b_mean)) + self.gamma*matmul(self.B.T, P_L) + self.cost.lu
        temp7 = self.cost.l11 + self.gamma*matmul(b_mean.T, matmul(P, b_mean)) + 2*self.gamma*matmul(b_mean.T, P_L)
        temp2 = P*self.sigma_xi
        trc2 = self.gamma*temp2.trace()/(1-self.gamma)
        
        e = (temp7-self.gamma*matmul(temp6.T, matmul(torch.linalg.inv(Quu), temp6)))/(1-self.gamma) + trc2
                                             # Note: Here we have inverse but we know Quu is pd hence it has an inverse
        
        return Q, Q_L, e
        
    def relax_q_analytic_solution(self):    
        # Differs from q function in a constant term delta e =  gamma*Tr(Qxu*Quu^-1*Qxu'Sigma)/(1-gamma) 
                                         
        P, Qxx, Qxu, Quu, P_L = self.calculate_parameters()
        
                                             # Constant Q for quadratic term
        Q_temp1 = torch.cat((Qxx, Qxu), 1)
        Q_temp2 = torch.cat((Qxu.T, Quu), 1)
        Q = torch.cat((Q_temp1, Q_temp2), 0)
        
        n = self.A.size(0)                   # Create a variable that deals with non zero mean noise.
        
        b_mean = self.b + torch.ones(n,1)*self.mu_xi
        
                                             # Constant QL for linear term has closed-form solution
        qx1 = 2*(self.cost.lx + self.gamma*matmul(self.A.T, matmul(P, b_mean)) + self.gamma*matmul(self.A.T, P_L))
        qu1 = 2*(self.cost.lu + self.gamma*matmul(self.B.T, matmul(P, b_mean)) + self.gamma*matmul(self.B.T, P_L))
        Q_L = torch.cat((qx1, qu1), 0)
        
                                             # Constant term q has closed-form solution
        temp6 = self.gamma*matmul(self.B.T, matmul(P, b_mean)) + self.gamma*matmul(self.B.T, P_L) + self.cost.lu
        temp7 = self.cost.l11 + self.gamma*matmul(b_mean.T, matmul(P, b_mean)) + 2*self.gamma*matmul(b_mean.T, P_L) #+ self.gamma*torch.trace(Qxx*self.sigma_xi)
        temp = Qxx*self.sigma_xi
        temp2 = P*self.sigma_xi
        trc =  self.gamma*temp.trace()/(1-self.gamma)

        e = (temp7-self.gamma*matmul(temp6.T, matmul(torch.linalg.inv(Quu), temp6)))/(1-self.gamma) + trc
        
        return Q, Q_L, e               
    
    def analytic_opt_policy_coefficients(self):
                                             # u = K1*x + K2. Function returns K1 and K2
                                             # K1 = -Quu^-1 * Qxu'
                                             # K2 = -Quu^-1 * (gamma*B'*P*b + gamma*B'*P_L + lu)
                                         
        n = self.A.size(0)                   # Create a variable that allows us to deal
                                             # with non zero mean noise.
        b_mean = self.b + torch.ones(n,1)*self.mu_xi
        
        P, _, Qxu, Quu, P_L = self.calculate_parameters()                                 
        temp = self.gamma*matmul(self.B.T, matmul(P, b_mean)) + self.gamma*matmul(self.B.T, P_L) + self.cost.lu
        K1 = -matmul(torch.linalg.inv(Quu), Qxu.T)
        K2 = -matmul(torch.linalg.inv(Quu), temp)
        return K1, K2
              
    def analytic_opt_policy(self, x):        # V, q, relaxed q all yield same control input.
                                             # Returns control input for specific state x
                                             # If dimensions of x dont match, returns -1
        n = self.A.size(0)
        u = 0                                # Initialize input
                                             # Check dimensions of x
        if x.size(0)!=n or x.size(1)!=1: 
            print('x should be of dimensions '+n+' x 1')
            return -1, u  
                                             # Get coefficients for calculating input
        K1, K2 = self.analytic_opt_policy_coefficients()
        u = matmul(K1, x) + K2               # u = K1*x + K2
        return 0, u                          # 0 is indicator that calculation was successful
    
#==============================================================================
#_______Creating data tuple____________________________________________________
#==============================================================================
                                             # Create a matrix with all x-s
    def draw_x(self, no_constraints):        # Draw one x (dim nx1) from normal distribution   
        n = self.A.size(0)          
        x = torch.DoubleTensor(no_constraints, n, 1).uniform_(self.lb_x, self.ub_x)
        return x

    def draw_u(self, no_constraints):        # Draw u (dim mx1) from normal distribution   
        m = self.B.size(1)
        u = torch.DoubleTensor(no_constraints, m, 1).uniform_(self.lb_u, self.ub_u)
        return u
    
    def create_data_tuples(self, no_constraints, no_xi_samples):             
                                             # Data tuple {(x,u,x+,l)}
                                             # Based on x and u drawn from distribution 
                                             # oracle spits out x+ and l for each touple
                
        x = self.draw_x(no_constraints)
        u = self.draw_u(no_constraints)
        check, x_plus, l = self.oracle(x, u, no_xi_samples)
        if check == -1:                      # Check if oracle was successful
            print('Dimensions of x or u do not match.')
            sys.exit(1)
        
        return x,u,x_plus,l                  # Return tuples
            
    