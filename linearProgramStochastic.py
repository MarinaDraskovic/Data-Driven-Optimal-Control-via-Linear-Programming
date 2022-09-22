import gurobipy as gp
from gurobipy import GRB
import torch
import numpy 
from torch import tensor
import time
from torch import matmul
import random
random.seed(42)


class LinearProgramStochastic:
    """sup_q integral{q(x,u)*c(x,u) dx du}
    such that q(x,u)<=l(x,u)+gamma*E_xi{q(x_u+, w)}.
    """
    
    def __init__(self, system, mu_c, sigma_c):
        self.system = system
        self.mu_c = mu_c
        self.sigma_c = sigma_c
        
    def solve_LP(self, x, u, x_plus, l, w, do_binding = 0, flag_1st_it=0, binding_A= None, binding_B= None):     
        # Solves an RLP directly using gurobi solver. General LP: max c'x (- objective)  s.t: A'x < B (- constraints)
                                            
        model = gp.Model("relaxed ALP")         # Create a new model
        model.setParam('dualReductions', 0)     # Checking if solution is infeasible or unbounded  
        model.setParam('FeasibilityTol', 1e-9)  # Setting accuracy
        model.setParam('OptimalityTol', 1e-9)
        model.setParam(GRB.Param.NumericFocus, 3)
                
        if do_binding!=0:                       # If we have binding constraints
            model.setParam('QCPDual', 1)        # Set this parameter for enabling dual value calculation                  
       
        runtime = 0                             # Initialize runtime
        n = self.system.A.size(0)           
        m = self.system.B.size(1)               # Q dim (n+m)x(n+m), Q_L dim (n+m)x1, e dim 1x1 
                                               
        # ----------------Objective-------------------
        # Objective transformed to sup{ tr(Q*Sigma_c) + Q_L'*mu_c + e }
                                                
                                                # c1*Q_flat= flat(sigma.T) * Q_flat = tr(Sigma*Q)
        upper_triag = torch.triu(self.sigma_c)  # Extract upper triangular matrix
        lower_triag = torch.tril(self.sigma_c)  # Lower triangular with zeros on main diagonal
        lower_triag.fill_diagonal_(0)
        lower_triag = lower_triag.T             # Transpose it so that we can add it to upper triag
        triag = upper_triag + lower_triag       # Sum the two
        triag = triag.numpy()
        
        indices = numpy.triu_indices_from(triag)# Transform c1 into an array
        c1 = numpy.asarray( triag[indices] )
        c1=c1.reshape((1,c1.shape[0]))
         
        c2 = self.mu_c.T
        c3 = tensor([[1]])
        c2=c2.numpy()
        c3=c3.numpy()
        c = numpy.concatenate((c1,c2,c3), 1)         
        
        X = model.addMVar(c.size, lb=-GRB.INFINITY,ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='X')
        
        model.setObjective(c@X, GRB.MAXIMIZE)   # Set objective        
        
        # ----------------Constraints-------------------        
        # Constraints in form: A'x<=b
        # q(x,u) <= l(x,u)+gamma*E_xi{q(x_u+, w)}
        # q(x,u) - gamma*E_xi{q(x_u+, w)} <= l(x,u)
        # q(x,u) = [x;u]'Q[x;u]+[x;u]'Q_L+e = A1x *Q_flat + A2x *Q_L + A3x *e
        # q(x+,w) = [x+;w]'Q[x+;w] + [x+;w]'Q_L + e = A1plus*Q_flat + A2plus*Q_L + A3plus*e
                                        
        A = numpy.empty((0,c.size))             
        b = numpy.empty((0,1))                  # Initialize A and b here
        
        xu = torch.cat((x, u), 1)               # Creating [x;u]        
        xw = torch.cat((x_plus, w), 2)          # Creating [x+;w]
       
                                                # Creating A1x:
        temp1x = torch.repeat_interleave(xu,n+m,dim=2)
                                                # temp1 = [x1,x1,x1; x2,x2,x2; u,u,u]
        temp2x = temp1x.transpose(1,2)          # temp2 = [x1,x2,u; x1,x2,u; x1,x2,u]    
                    
        temp = temp1x*temp2x                    # Elementwise multiplication: [x1x1, x1x2, x1u; x2x1...]
        upper_triag = torch.triu(temp)          # Extract upper triangular matrix
        no_zero = torch.triu(temp)              # no_zero will be upper triangular matrix with 0s on main diagonal    
                                                # Creat mask to get rid of main diagonal
        mask  = torch.eye(upper_triag.size(1),upper_triag.size(2))
                                                # Subtract from matrix main diagonal    
        no_zero = no_zero - mask.repeat(upper_triag.size(0),1,1)*no_zero
        A1x_matrix = upper_triag + no_zero      # A1 in matrix term. Upper diag mat. All elements not on diagonal are *2
        A1x_matrix = A1x_matrix.numpy()
                                               
        r,c = numpy.triu_indices(n+m)           # Transform A1 into an array
        A1x = A1x_matrix[:,r,c] 
                                                # Creating A1plus is similar but we have to deal with multiple draws of xi:
        temp1plus = torch.repeat_interleave(xw,n+m,dim=3)
        temp2plus = temp1plus.transpose(2,3)                 
        temp = temp1plus*temp2plus              
        upper_triag = torch.triu(temp)         
        no_zero = torch.triu(temp)             
        mask  = torch.eye(upper_triag.size(2),upper_triag.size(3))
        no_zero = no_zero - mask.repeat(upper_triag.size(0),1,1,1)*no_zero
        A1plus_matrix = upper_triag + no_zero  
        A1plus_matrix = A1plus_matrix.numpy()        
        r,c = numpy.triu_indices(n+m)
        A1plus = A1plus_matrix[:,:,r,c]
        
        A1plus = numpy.mean(A1plus, 1)          # Mean over samples xi
        
        A1 = A1x - self.system.gamma*A1plus      # A1 = A1x - gamma*A1plus -> contains info on [x;u]'*[x;u]
                                                
        xw = torch.mean(xw,1) 
        A2 = xu - self.system.gamma*xw          # A2 = A2x - gamma*A2plus = [x;u]'-gamma*[x+;w]'
        A2 = torch.reshape(A2, (x_plus.shape[0], n+m)).double()
        A2 = A2.numpy()
                                                # A3 = A3x - gamma*A3plus =1-gamma
        A3 = (1-self.system.gamma)*torch.ones((x_plus.shape[0],1,1))  
        A3 = torch.reshape(A3, (x_plus.shape[0], 1)).double()
        A3 = A3.numpy()     
                                          
        A = numpy.concatenate((A1,A2,A3), 1)  
        
        b = l.numpy() 
        b = numpy.squeeze(b)
        
        if do_binding!=0 and flag_1st_it==0:
                                                # If we have binding constraints from previous iteration, add them as constraints
            A = numpy.concatenate((A, binding_A), 0)
            b = numpy.concatenate((b, binding_B), 0)
            
        constraints = model.addConstr(A@X <= b, name='constr')
        
        model.optimize()                        # Optimize model
        runtime = model.Runtime            
                                                # Extracting Q, Q_L and e from X
        lengthQ = X.x.size - (n+m) - 1          # Extract Q, QL and e from variable X
        Qflat = X.x[0:lengthQ]
        indices = numpy.triu_indices(n+m)
        Qupper = numpy.zeros((n+m,n+m))
        Qupper[indices] = Qflat 
        Q = Qupper + Qupper.T - numpy.diag(Qupper.diagonal())
        
        Q_L = X.x[lengthQ:X.x.size-1]
        e = X.x[X.x.size-1]
        
        if model.Status == 2:                       # If LP was successfully solved
            if do_binding!=0:
                print(model.getAttr(GRB.Attr.Pi))   # Prints dual values for the constraints. 0->non binding constraint
                                                    # Get indeces of non zero values in the matrix of dual values
                nonzero_indeces = numpy.nonzero(model.getAttr(GRB.Attr.Pi)) 
                print(nonzero_indeces)
                print(constraints)
                binding_A = A[nonzero_indeces]      # Remember binding constraints for the next iteration
                binding_B = b[nonzero_indeces]
                print(binding_A.shape)
            else:                                   # If we are not using binding constraints, return none for this output
                binding_A = None       
                binding_B = None  
    
            return model.objVal,Q,Q_L,e,runtime, binding_A, binding_B 
        else:
            return None, None, None, None, None, None, None           
        
    def solve_linearizedLP(self, x, u, x_plus, l, w, do_binding = 0, flag_1st_it=0, binding_A= None, binding_B= None):     
        # Solves linearized LP
                                            
        model = gp.Model("relaxed ALP")         # Create a new model
        model.setParam('dualReductions', 0)     # Checking if solution is infeasible or unbounded          
        model.setParam('FeasibilityTol', 1e-9)  # Set accuracy
        model.setParam('OptimalityTol', 1e-9)
        model.setParam('NumericFocus', 3)
                
        if do_binding!=0:                       # If we have some binding constraints
            model.setParam('QCPDual', 1)        # Set this parameter for enabling dual value calculation                  
        runtime = 0                             # Initialize runtime
        n = self.system.A.size(0)           
        m = self.system.B.size(1)               # Q dim (n+m)x(n+m), Q_L dim (n+m)x1, e dim 1x1 
                 
        #-----------------Objective------------------                              
        # Multidimensional integral in objective transformed to 
        # sup{ tr(Q*Sigma_c) + Q_L'*mu_c + e }
                                                # c1*Q_flat= flat(sigma.T) * Q_flat = tr(Sigma*Q)
        upper_triag = torch.triu(self.sigma_c)  # Extract upper triangular matrix
        lower_triag = torch.tril(self.sigma_c)  # Lower triangular with zeros on main diagonal
        lower_triag.fill_diagonal_(0)
        lower_triag = lower_triag.T             # Transpose it so that we can add it to upper triag
        triag = upper_triag + lower_triag       # Sum the two
        triag = triag.numpy()
        
        indices = numpy.triu_indices_from(triag)# Transform c1 into an array
        c1 = numpy.asarray( triag[indices] )
        c1=c1.reshape((1,c1.shape[0]))
         
        c2 = self.mu_c.T
        c3 = tensor([[1]])
        c2=c2.numpy()
        c3=c3.numpy()
        cq = numpy.concatenate((c1,c2,c3), 1)         
        
        X = model.addMVar(cq.size+(n+m)*2, lb=-GRB.INFINITY,ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='X')
        
        cv = numpy.zeros(((n+m)*2,1)).T
        c =  numpy.concatenate((cq, cv), 1)
       
        model.setObjective(c@X, GRB.MAXIMIZE)   # Set objective        
                
        #------------Constraints---------------
                      
        # Related to q(x,u)     - A                     
        A = numpy.empty((0,c.size))             # Constraints in form: A'x<=b
        b = numpy.empty((0,1))                  # Initialize A and b here
        
        xu = torch.cat((x, u), 1)               # Creating [x;u]
       
                                                # Creating A1x:
        temp1x = torch.repeat_interleave(xu,n+m,dim=2)
                                                # temp1 = [x1,x1,x1; x2,x2,x2; u,u,u]
        temp2x = temp1x.transpose(1,2)          # temp2 = [x1,x2,u; x1,x2,u; x1,x2,u]    
                    
        temp = temp1x*temp2x                    # Elementwise multiplication: [x1x1, x1x2, x1u; x2x1...]
        upper_triag = torch.triu(temp)          # Extract upper triangular matrix
        no_zero = torch.triu(temp)              # no_zero will be upper triangular matrix with 0s on main diagonal    
                                                # Creat mask to get rid of main diagonal
        mask  = torch.eye(upper_triag.size(1),upper_triag.size(2))
                                                # Subtract from matrix main diagonal    
        no_zero = no_zero - mask.repeat(upper_triag.size(0),1,1)*no_zero
        A1x_matrix = upper_triag + no_zero      # A1 in matrix term. Upper diag mat. All elements not on diagonal are *2
        A1x_matrix = A1x_matrix.numpy()
                                                # Transform A1 into an array
        
        r,c = numpy.triu_indices(n+m)
        A1 = A1x_matrix[:,r,c] 
        A2 = xu                                 # A2 multiplies Q_L
        A2 = torch.reshape(A2, (x_plus.shape[0], n+m)).double()
        A2 = A2.numpy()
                                                # A3 multiplies e
        A3 = (1)*torch.ones((x_plus.shape[0],1,1))  
        A3 = torch.reshape(A3, (x_plus.shape[0], 1)).double()
        A3 = A3.numpy()            
        
        # Related to V(x)   - Bx
        temp1x = torch.repeat_interleave(x,n,dim=2)
                                                # temp1 = [x1,x1; x2,x2]
        temp2x = temp1x.transpose(1,2)          # temp2 = [x1,x2; x1,x2]    
                    
        temp = temp1x*temp2x                    # Elementwise multiplication: [x1x1, x1x2; x2x1, x2x2]
        upper_triag = torch.triu(temp)          # Extract upper triangular matrix
        no_zero = torch.triu(temp)              # no_zero will be upper triangular matrix with 0s on main diagonal    
                                                # Creat mask to get rid of main diagonal
        mask  = torch.eye(upper_triag.size(1),upper_triag.size(2))
                                                # Subtract from matrix main diagonal    
        no_zero = no_zero - mask.repeat(upper_triag.size(0),1,1)*no_zero
        B1x_matrix = upper_triag + no_zero      # B1x in matrix term. Upper diag mat. All elements not on diagonal are *2
        B1x_matrix = B1x_matrix.numpy()
                                                # Transform into an array       
        r,c = numpy.triu_indices(n)
        B1x = B1x_matrix[:,r,c] 
        
        B2x = 2*x
        B2x = torch.reshape(B2x, (x.shape[0], n)).double()
        B2x = B2x.numpy()
        
        B3x = torch.ones((x.shape[0],1,1))  
        B3x = torch.reshape(B3x, (x.shape[0], 1)).double()
        B3x = B3x.numpy() 
        
        
        # Related to V(x+)  - B
        temp1plus = torch.repeat_interleave(x_plus,n,dim=3)
        temp2plus = temp1plus.transpose(2,3)                   
        temp = temp1plus*temp2plus             
        upper_triag = torch.triu(temp)          
        no_zero = torch.triu(temp)            
        mask  = torch.eye(upper_triag.size(2),upper_triag.size(3)) 
        no_zero = no_zero - mask.repeat(upper_triag.size(0),1,1,1)*no_zero
        B1plus_matrix = upper_triag + no_zero   
        B1plus_matrix = B1plus_matrix.numpy()
        r,c = numpy.triu_indices(n)
        B1plus = B1plus_matrix[:,:,r,c]
        B1plus = numpy.mean(B1plus, 1)
        
        B1 = - self.system.gamma*B1plus
                                                
        xplus = torch.mean(x_plus,1) 
        B2 = - self.system.gamma*xplus         
        B2 = torch.reshape(B2, (x_plus.shape[0], n)).double()
        B2 = B2.numpy()
        
        B3 = (-self.system.gamma)*torch.ones((x_plus.shape[0],1,1))  
        B3 = torch.reshape(B3, (x_plus.shape[0], 1)).double()
        B3 = B3.numpy() 

                                   
        AA = numpy.concatenate((A1,A2,A3), 1)
        B = numpy.concatenate((B1,B2,B3), 1)
        Bx = numpy.concatenate((B1x,B2x,B3x), 1)
                                               
        A = numpy.concatenate((numpy.concatenate((AA, -B), 1), numpy.concatenate((Bx,-AA), 1)), 0)
     
        b = l.numpy() 
        b = numpy.squeeze(b)
        b = numpy.concatenate((b, numpy.zeros(AA.shape[0], dtype=type)))
        
        if do_binding!=0 and flag_1st_it==0:
                                                # If we have binding constraints from previous iteration, add them as constraints
            A = numpy.concatenate((A, binding_A), 0)
            b = numpy.concatenate((b, binding_B), 0)

        constraints = model.addConstr(A@X <= b, name='constr')
        
        model.optimize()                        # Optimize model
        runtime = model.Runtime            
                                                # Extracting Q, Q_L and e from X
        lengthQ = X.x.size - ((n+m)*2) - (n+m) - 1      # Extract Q, QL and e from variable X
        Qflat = X.x[0:lengthQ]
        indices = numpy.triu_indices(n+m)
        Qupper = numpy.zeros((n+m,n+m))
        Qupper[indices] = Qflat 
        Q = Qupper + Qupper.T - numpy.diag(Qupper.diagonal())
        
        Q_L = X.x[lengthQ:X.x.size-((n+m)*2)-1]
        e = X.x[X.x.size-((n+m)*2)-1]
        
        if model.Status == 2:                       # If LP was successfully solved
            if do_binding!=0:
                print(model.getAttr(GRB.Attr.Pi))   # Prints dual values for the constraints. 0->non binding constraint
                                                    # Get indeces of non zero values in the matrix of dual values
                nonzero_indeces = numpy.nonzero(model.getAttr(GRB.Attr.Pi)) 
                print(nonzero_indeces)
                print(constraints)
                binding_A = A[nonzero_indeces]      # Remember binding constraints for the next iteration
                binding_B = b[nonzero_indeces]
                print(binding_A.shape)
            else:                                   # If we are not using binding constraints, return none for this output
                binding_A = None       
                binding_B = None  
    
            return model.objVal,Q,Q_L,e,runtime, binding_A, binding_B 
        else:
            return None, None, None, None, None, None, None           
        
        
    def solve_directly(self, no_constraints, no_xi_samples): 
                                                   # Get data tuples
        x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)
        
        w = self.system.draw_u(no_constraints)     # Draw w from the distribution
        m = self.system.B.size(1) 
        w = torch.reshape(w, (no_constraints,1,m,1)) 
        w = w.repeat(1,no_xi_samples,1,1)
                                                   # Call solver 
        return self.solve_LP(x, u, x_plus, l, w)
    
    def solve_linearizedLP_directly(self, no_constraints, no_xi_samples): 
                                                   # Get data tuples
        x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)
        
        w = self.system.draw_u(no_constraints)     # Draw w from the distribution
        m = self.system.B.size(1) 
        w = torch.reshape(w, (no_constraints,1,m,1)) 
        w = w.repeat(1,no_xi_samples,1,1)
                                                   # Call solver 
        return self.solve_linearizedLP(x, u, x_plus, l, w)
                                                         
    def greedy_policy_coefficients(self, Q, Q_L):  # pi*(x)=-Quu^(-1)*Qxu'*x - Quu^(-1)*qu1 = K1*x + K2
                                                   # Function returns K1 and K2
         n = self.system.A.size(0) 
         
         Quu = Q[n:,n:]                            # Extract Quu, Qxu' and qu1 from Q and Q_L
         QxuT = Q[n:,0:n]
         qu1 = 0.5*Q_L[n:]
         
         try:
             K1 = -matmul(torch.linalg.inv(Quu), QxuT)
             K2 = -matmul(torch.linalg.inv(Quu), qu1)
         except:
             print('Calculating pseudo inverses for getting coefficients of greedy policy.')
             K1 = -numpy.matmul(numpy.linalg.pinv(Quu), QxuT)
             K2 = -numpy.matmul(numpy.linalg.pinv(Quu), qu1)
         return K1, K2
     
    def greedy_policy(self, Q, Q_L, x):            # pi*(x) = K1*x + K2
                                             # Returns greedy policy for specific x 
        n = self.A.size(0)
        pi = 0                               # Initialize policy
                                             # Check dimensions of x
        if x.size(0)!=n or x.size(1)!=1: 
            print('x should be of dimensions '+n+' x 1')
            return -1, pi  
                                             # Get coefficients for calculating input
        K1, K2 = self.greedy_policy_coefficients(Q, Q_L)
        pi = matmul(K1, x) + K2              #pi = K1*x + K2
        return 0, pi                         # 0 is indicator that calculation was successful
    
    def policy_iteration(self, LP, no_constraints, no_xi_samples, number_iterations):      
        n = self.system.A.size(0)                  # Get dimensions
        m = self.system.B.size(1) 
        
        K1 = [-1, 1]  # numpy.ones((m,n))          # Initialize policy
        K2 = [0] # numpy.zeros((1,1))                                    
        value_difference = float('inf')            # Initialize difference between q functions in 2 steps
        iteration = 0                              # For counting iterations 
        runtime = []
        objective_old = 0
        objective_new = 0
        prev_runtime = 0
        objectives = []                            # Initialize array which contains all objectives
        
        while(iteration < number_iterations):      # or value_difference > treshold
            
            start_time = time.time()
          # Policy evaluation                      # Get tuples
            x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)
            KK1 = torch.tensor(K1).double()        # Calculate w = mu(x+)
            KK1 = torch.reshape(KK1, (1,1,m,n))    
            KK1 = KK1.repeat(no_constraints, no_xi_samples,1,1)
            w = matmul(KK1,x_plus) + torch.tensor(K2).double() 
            
            objective_old = objective_new          # objective_old is now objective from previous iteration
            objective_new, Q, Q_L, e, rt,_ ,_  = self.solve_LP(x, u, x_plus, l, w)
            objectives.append(objective_new)
            
          # Polici improvement
            K1 , K2 = self.greedy_policy_coefficients(Q, Q_L)
            
          # Calculate the difference to determine if algorithm should continue
            if iteration>0:
               value_difference = abs(objective_old - objective_new)
  
            print("Iteration: ", iteration, "; Difference", value_difference)
            iteration += 1 
            end_time = time.time()
            runtime.append(prev_runtime+ end_time-start_time)
            prev_runtime = prev_runtime + end_time-start_time
            
        return objectives, Q, Q_L, e, runtime, K1, K2
    
    def policy_iteration2(self, LP, no_constraints, no_xi_samples, number_iterations):     
        # PI that draws omega from distribution and allows exploration of new policies
        
        n = self.system.A.size(0)                  # Get dimensions
        m = self.system.B.size(1) 
        
        K1 = [-1, 1]  # numpy.ones((m,n))          # Initialize policy
        K2 = [0] # numpy.zeros((1,1))                                    
        value_difference = float('inf')            # Initialize difference between q functions in 2 steps
        iteration = 0                              # For counting iterations 
        runtime = []
        objective_old = 0
        objective_new = 0        
        prev_runtime=0        
        objectives = []                            # Initialize array which contains all objectives
        
        while(iteration < number_iterations):      # or value_difference > treshold
            
            start_time = time.time()
          # Policy evaluation                      # Get tuples
            x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)
            w =self.system.draw_u(no_constraints)  # Draw w from the distribution
            m = self.system.B.size(1) 
            w = torch.reshape(w, (no_constraints,1,m,1)) 
            w = w.repeat(1,no_xi_samples,1,1)
            
            objective_old = objective_new          # objective_old is now objective from previous iteration
            objective_new, Q, Q_L, e, rt,_ ,_  = self.solve_LP(x, u, x_plus, l, w)
            objectives.append(objective_new)
            
          # Polici improvement
            K1 , K2 = self.greedy_policy_coefficients(Q, Q_L)
            
          # Calculate the difference to determine if algorithm should continue
            if iteration>0:
               value_difference = abs(objective_old - objective_new)
  
            print("Iteration: ", iteration, "; Difference", value_difference)
            iteration += 1 
            end_time = time.time()
            runtime.append(prev_runtime+end_time-start_time)
            prev_runtime=prev_runtime + end_time - start_time
            
        return objectives, Q, Q_L, e, runtime, K1, K2
            
    def PI_on_policy(self, LP, no_constraints, no_xi_samples, number_iterations):      
        n = self.system.A.size(0)                  # Get dimensions
        m = self.system.B.size(1) 
        
        K1 = [-1, 1]  # numpy.ones((m,n))          # Initialize policy
        K2 = [0] # numpy.zeros((1,1))                                    
        value_difference = float('inf')            # Initialize difference between q functions in 2 steps
        iteration = 0                              # For counting iterations 
        runtime = []
        objective_old = 0
        objective_new = 0
        binding_A = []
        binding_B = []
        flag_1st_iteration = 1                     # To know if there are binding constraints yet or no
        
        old_runtime = 0
        
        objectives = []                            # Initialize array which contains all objectives
        
        while(iteration < number_iterations):      # or value_difference > treshold
            start_time = time.time()
          # Policy evaluation                      # Get tuples
            x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)
            
            w =self.system.draw_u(no_constraints)  # Draw w from the distribution
            m = self.system.B.size(1) 
            w = torch.reshape(w, (no_constraints,1,m,1)) 
            w = w.repeat(1,no_xi_samples,1,1)
            
            # KK1 = torch.tensor(K1).double()        # Calculate w = mu(x+)
            # KK1 = torch.reshape(KK1, (1,1,m,n))    
            # KK1 = KK1.repeat(no_constraints, no_xi_samples,1,1)
            # w = matmul(KK1,x_plus) + torch.tensor(K2).double() 
            
            objective_old = objective_new          # objective_old is now objective from previous iteration

            if flag_1st_iteration == 1:            # In the 1st iteration call LP normally
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1,flag_1st_iteration)
                flag_1st_iteration = 0
            else:                                  # In further iterations call LP w/ binding constraints
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1, flag_1st_iteration ,binding_A, binding_B)
            
            objectives.append(objective_new)
            
          # Polici improvement
            K1 , K2 = self.greedy_policy_coefficients(Q, Q_L)
            
          # Calculate the difference to determine if algorithm should continue
            if iteration>0:
               value_difference = abs(objective_old - objective_new)
  
            print("Iteration: ", iteration, "; Difference", value_difference)
            iteration += 1   
            end_time = time.time()
            runtime.append(old_runtime + end_time - start_time)
            old_runtime = old_runtime + end_time-start_time
            
        return objectives, Q, Q_L, e, runtime, K1, K2
    
    def PI_on_policy2(self, LP, no_constraints, no_xi_samples, number_iterations):    
        # On-policy setting that uses greedy policy
        
        n = self.system.A.size(0)                  # Get dimensions
        m = self.system.B.size(1) 
        
        K1 = [-1, 1]  # numpy.ones((m,n))          # Initialize policy
        K2 = [0] # numpy.zeros((1,1))                                    
        value_difference = float('inf')            # Initialize difference between q functions in 2 steps
        iteration = 0                              # For counting iterations 
        runtime = []
        objective_old = 0
        objective_new = 0
        binding_A = []
        binding_B = []
        flag_1st_iteration = 1                     # To know if there are binding constraints yet or no
        
        old_runtime=0
        
        objectives = []                            # Initialize array which contains all objectives
        
        while(iteration < number_iterations):      # or value_difference > treshold
            start_time = time.time()
          # Policy evaluation                      # Get tuples
            x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)
            
            # Repeat tuples 5 times
            x = x.repeat(5,1,1)
            u = u.repeat(5,1,1)
            x_plus = x_plus.repeat(5,1,1,1)
            l = l.repeat(5,1,1)
            
            
            KK1 = torch.tensor(K1).double()        # Calculate w = mu(x+)
            KK1 = torch.reshape(KK1, (1,1,m,n))    
            KK1 = KK1.repeat(no_constraints*5, no_xi_samples,1,1)
            w = matmul(KK1,x_plus) + torch.tensor(K2).double() 
            
            objective_old = objective_new          # objective_old is now objective from previous iteration

            if flag_1st_iteration == 1:            # In the 1st iteration call LP normally
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1,flag_1st_iteration)
                flag_1st_iteration = 0
            else:                                  # In further iterations call LP w/ binding constraints
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1, flag_1st_iteration ,binding_A, binding_B)
            
            objectives.append(objective_new)
            
          # Polici improvement
            K1 , K2 = self.greedy_policy_coefficients(Q, Q_L)
            
          # Calculate the difference to determine if algorithm should continue
            if iteration>0:
               value_difference = abs(objective_old - objective_new)
  
            print("Iteration: ", iteration, "; Difference", value_difference)
            iteration += 1   
            end_time = time.time()
            runtime.append(old_runtime+ end_time-start_time)
            old_runtime = old_runtime + end_time-start_time
        return objectives, Q, Q_L, e, runtime, K1, K2
    
    def PI_off_policy(self, LP, no_constraints, no_xi_samples, number_iterations):      
        n = self.system.A.size(0)                  # Get dimensions
        m = self.system.B.size(1) 
        
        K1 = [-1, 1]  # numpy.ones((m,n))          # Initialize policy
        K2 = [0] # numpy.zeros((1,1))                                    
        value_difference = float('inf')            # Initialize difference between q functions in 2 steps
        iteration = 0                              # For counting iterations 
        runtime = []
        objective_old = 0
        objective_new = 0
        old_runtime = 0
        objectives = []                            # Initialize array which contains all objectives
        
        # Construct buffer
        x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)
        
        while(iteration < number_iterations):      # or value_difference > treshold
            start_time = time.time()
          # Policy evaluation                      
            KK1 = torch.tensor(K1).double()        # Calculate w = mu(x+)
            KK1 = torch.reshape(KK1, (1,1,m,n))    
            KK1 = KK1.repeat(no_constraints, no_xi_samples,1,1)
            w = matmul(KK1,x_plus) + torch.tensor(K2).double() 
            
            objective_old = objective_new          # objective_old is now objective from previous iteration
            objective_new, Q, Q_L, e, rt,_ ,_  = self.solve_LP(x, u, x_plus, l, w)
            objectives.append(objective_new)
            
          # Polici improvement
            K1 , K2 = self.greedy_policy_coefficients(Q, Q_L)
            
          # Calculate the difference to determine if algorithm should continue
            if iteration>0:
               value_difference = abs(objective_old - objective_new)
  
            print("Iteration: ", iteration, "; Difference", value_difference)
            iteration += 1    
            end_time = time.time()
            runtime.append(old_runtime+ end_time-start_time)
            old_runtime= old_runtime + end_time-start_time
        return objectives, Q, Q_L, e, runtime, K1, K2
    
    
    def PI_off_policy2(self, LP, no_constraints, no_xi_samples, number_iterations):      
        # Off-policy setting that draws omega from distribution; allows exploration of policies
        
        n = self.system.A.size(0)                  # Get dimensions
        m = self.system.B.size(1) 
        
        K1 = [-1, 1]  # numpy.ones((m,n))          # Initialize policy
        K2 = [0] # numpy.zeros((1,1))                                    
        value_difference = float('inf')            # Initialize difference between q functions in 2 steps
        iteration = 0                              # For counting iterations 
        runtime = []
        objective_old = 0
        objective_new = 0
        old_runtime = 0
        objectives = []                            # Initialize array which contains all objectives
        binding_A = []
        binding_B = []
        flag_1st_iteration = 1    
        
        # Construct buffer
        x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)
        
        while(iteration < number_iterations):      # or value_difference > treshold
            start_time = time.time()
          # Policy evaluation                      
            w =self.system.draw_u(no_constraints)  # Draw w from the distribution
            m = self.system.B.size(1) 
            w = torch.reshape(w, (no_constraints,1,m,1)) 
            w = w.repeat(1,no_xi_samples,1,1)
            
            objective_old = objective_new          # objective_old is now objective from previous iteration
            
            if flag_1st_iteration == 1:            # In the 1st iteration call LP normally
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1,flag_1st_iteration)
                flag_1st_iteration = 0
            else:                                  # In further iterations call LP w/ binding constraints
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1, flag_1st_iteration ,binding_A, binding_B)
            
            # objective_new, Q, Q_L, e, rt,_ ,_  = self.solve_LP(x, u, x_plus, l, w)
            
            objectives.append(objective_new)
            
          # Polici improvement
            K1 , K2 = self.greedy_policy_coefficients(Q, Q_L)
            
          # Calculate the difference to determine if algorithm should continue
            if iteration>0:
               value_difference = abs(objective_old - objective_new)
  
            print("Iteration: ", iteration, "; Difference", value_difference)
            iteration += 1    
            end_time = time.time()
            runtime.append(old_runtime+ end_time-start_time)
            old_runtime = old_runtime + end_time-start_time
        return objectives, Q, Q_L, e, runtime, K1, K2
    
    def PI_on_policy_5w(self, LP, no_constraints, no_xi_samples, number_iterations):  
        # On-policy setting with 5 draws of omega for each data tuble
        
        n = self.system.A.size(0)                  # Get dimensions
        m = self.system.B.size(1) 
        
        K1 = [-1, 1]  # numpy.ones((m,n))          # Initialize policy
        K2 = [0] # numpy.zeros((1,1))                                    
        value_difference = float('inf')            # Initialize difference between q functions in 2 steps
        iteration = 0                              # For counting iterations 
        runtime = []
        objective_old = 0
        objective_new = 0
        binding_A = []
        binding_B = []
        flag_1st_iteration = 1                     # To know if there are binding constraints yet or no
        
        old_runtime = 0
        
        objectives = []                            # Initialize array which contains all objectives
        
        while(iteration < number_iterations):      # or value_difference > treshold
            start_time = time.time()
          # Policy evaluation                      # Get tuples
            x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)

            # Repeat tuples 5 times
            x = x.repeat(4,1,1)
            u = u.repeat(4,1,1)
            x_plus = x_plus.repeat(4,1,1,1)
            l = l.repeat(4,1,1)
            
            w =self.system.draw_u(no_constraints*4)  # Draw w from the distribution
            m = self.system.B.size(1) 
            w = torch.reshape(w, (no_constraints*4,1,m,1)) 
            w = w.repeat(1,no_xi_samples,1,1)
            

            
            # KK1 = torch.tensor(K1).double()        # Calculate w = mu(x+)
            # KK1 = torch.reshape(KK1, (1,1,m,n))    
            # KK1 = KK1.repeat(no_constraints, no_xi_samples,1,1)
            # w = matmul(KK1,x_plus) + torch.tensor(K2).double() 
            
            objective_old = objective_new          # objective_old is now objective from previous iteration

            if flag_1st_iteration == 1:            # In the 1st iteration call LP normally
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1,flag_1st_iteration)
                flag_1st_iteration = 0
            else:                                  # In further iterations call LP w/ binding constraints
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1, flag_1st_iteration ,binding_A, binding_B)
            
            objectives.append(objective_new)
            
          # Polici improvement
            K1 , K2 = self.greedy_policy_coefficients(Q, Q_L)
            
          # Calculate the difference to determine if algorithm should continue
            if iteration>0:
               value_difference = abs(objective_old - objective_new)
  
            print("Iteration: ", iteration, "; Difference", value_difference)
            iteration += 1   
            end_time = time.time()
            runtime.append(old_runtime + end_time - start_time)
            old_runtime = old_runtime + end_time-start_time
            
        return objectives, Q, Q_L, e, runtime, K1, K2
    
    def PI_on_policy_10w(self, LP, no_constraints, no_xi_samples, number_iterations):  
        # On-policy setting with 10 draws of omega for each data tuble
        
        n = self.system.A.size(0)                  # Get dimensions
        m = self.system.B.size(1) 
        
        K1 = [-1, 1]  # numpy.ones((m,n))          # Initialize policy
        K2 = [0] # numpy.zeros((1,1))                                    
        value_difference = float('inf')            # Initialize difference between q functions in 2 steps
        iteration = 0                              # For counting iterations 
        runtime = []
        objective_old = 0
        objective_new = 0
        binding_A = []
        binding_B = []
        flag_1st_iteration = 1                     # To know if there are binding constraints yet or no
        
        old_runtime = 0
        
        objectives = []                            # Initialize array which contains all objectives
        
        while(iteration < number_iterations):      # or value_difference > treshold
            start_time = time.time()
          # Policy evaluation                      # Get tuples
            x, u, x_plus, l = self.system.create_data_tuples(no_constraints, no_xi_samples)

            # Repeat tuples 5 times
            x = x.repeat(10,1,1)
            u = u.repeat(10,1,1)
            x_plus = x_plus.repeat(10,1,1,1)
            l = l.repeat(10,1,1)
            
            w =self.system.draw_u(no_constraints*10)  # Draw w from the distribution
            m = self.system.B.size(1) 
            w = torch.reshape(w, (no_constraints*10,1,m,1)) 
            w = w.repeat(1,no_xi_samples,1,1)
            

            
            # KK1 = torch.tensor(K1).double()        # Calculate w = mu(x+)
            # KK1 = torch.reshape(KK1, (1,1,m,n))    
            # KK1 = KK1.repeat(no_constraints, no_xi_samples,1,1)
            # w = matmul(KK1,x_plus) + torch.tensor(K2).double() 
            
            objective_old = objective_new          # objective_old is now objective from previous iteration

            if flag_1st_iteration == 1:            # In the 1st iteration call LP normally
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1,flag_1st_iteration)
                flag_1st_iteration = 0
            else:                                  # In further iterations call LP w/ binding constraints
                objective_new, Q, Q_L, e, rt, binding_A, binding_B = self.solve_LP(x, u, x_plus, l, w, 1, flag_1st_iteration ,binding_A, binding_B)
            
            objectives.append(objective_new)
            
          # Polici improvement
            K1 , K2 = self.greedy_policy_coefficients(Q, Q_L)
            
          # Calculate the difference to determine if algorithm should continue
            if iteration>0:
               value_difference = abs(objective_old - objective_new)
  
            print("Iteration: ", iteration, "; Difference", value_difference)
            iteration += 1   
            end_time = time.time()
            runtime.append(old_runtime + end_time - start_time)
            old_runtime = old_runtime + end_time-start_time
            
        return objectives, Q, Q_L, e, runtime, K1, K2