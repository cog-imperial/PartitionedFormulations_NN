"""
This file contains implementations of the Optimal Adversary problem described in
Section 4.1 of the manuscript. The file can be run directly, with the following parameters.
Parameters can be changed on lines 16-18.
    
    Parameters:
        problemNo (int): index of problem to be solved, can be in range(100)
        N (int): number of partitions to use
        epsilon (real): maximum l1-norm defining perturbations
"""

import numpy as np
import gurobipy as gb
import partitioningStrategies

problemNo = 0
N = 2
epsilon = 10

nLayers = 3
instances = np.load('mnist2x50instances.npz')
inputimage = instances['images'][problemNo]
labels = instances['labels'][problemNo]

def getPartitions(m, N):
    """ 
    Get partitions from partitioningStrategies.py
    Options are: {'getEqualSize', 'getEqualRange', 'getRandom', 'getUnevenMagnitudes'}
    """
    return partitioningStrategies.getEqualSize(m, N)

#BEGIN OPTIMIZATION MODEL
model = gb.Model()
x = {}; y = {}; z2 = {}; sig = {};

# Create input nodes; layer '0' is the input
x[0] = {}; z2[0] = {}; sig[0] = {}
for i in range(instances['w1'].shape[1]):
    x[0][i] = model.addVar(max(inputimage[i] - epsilon, 0), min(1, inputimage[i] + epsilon), name='x_' + str(0) + '_' + str(i))
    y[i] = model.addVar(0, epsilon)
    model.addConstr(y[i] >= x[0][i] - inputimage[i])
    model.addConstr(y[i] >= inputimage[i] - x[0][i])

model.update()
# l1-norm constraint
model.addConstr(sum(y[i] for i in range(instances['w1'].shape[1])) <= epsilon)

for ind in range(nLayers):
    # vars of 'ind+1' correspond to the output of layer 'ind'
    x[ind + 1] = {}; z2[ind + 1] = {}; sig[ind + 1] = {}
    # get weights for layer ind
    w_layer = instances['w' + str(ind+1)]
    b_layer = instances['b' + str(ind+1)]
    
    for i in range(w_layer.shape[0]):
        # get weights and biases for node i of layer ind
        w = w_layer[i]; b = b_layer[i]
        partitions = getPartitions(w, N)
        
        # compute upper and lower bounds for output using interval arithmetic
        ub = sum(x[ind][j].UB*max(0,w[j]) + x[ind][j].LB*min(0,w[j]) for j in range(len(w))) + b
        lb = sum(x[ind][j].UB*min(0,w[j]) + x[ind][j].LB*max(0,w[j]) for j in range(len(w))) + b

        if ind == nLayers - 1:
            # no ReLU activation in final layer
            x[ind + 1][i] = model.addVar(lb, ub, name='x_' + str(ind + 1) + '_' + str(i))
            model.addConstr(x[ind + 1][i] == sum(x[ind][j] * w[j] for j in range(len(w))) + b)

        else:
            # define variables for partition-based formulation
            x[ind+1][i] = model.addVar(0, max(0, ub), name='x_' + str(ind + 1) + '_' + str(i))
            sig[ind+1][i] = model.addVar(0, 1, vtype=gb.GRB.BINARY, name='sigma_' + str(ind + 1) + '_' + str(i))
            z2[ind+1][i] = {}

            for j in range(len(partitions)):
                # compute upper and lower bounds for partition using interval arithmetic
                ub = sum(x[ind][n].UB*max(0,w[n]) + x[ind][n].LB*min(0,w[n]) for n in partitions[j])
                lb = sum(x[ind][n].UB*min(0,w[n]) + x[ind][n].LB*max(0,w[n]) for n in partitions[j])

                # create auxiliary partition variables
                z2[ind+1][i][j] = model.addVar(min(0, lb), max(0, ub), name='z2_' + str(ind + 1) + '_' + str(i) + '_' + str(j))

                # auxiliary variable bounds--equations (22)-(23)
                model.addConstr(sum(x[ind][n]*w[n] for n in partitions[j]) - z2[ind+1][i][j] >= sig[ind+1][i]*lb)
                model.addConstr(sum(x[ind][n]*w[n] for n in partitions[j]) - z2[ind+1][i][j] <= sig[ind+1][i]*ub)
                model.addConstr(z2[ind+1][i][j] >= (1 - sig[ind+1][i]) * lb)
                model.addConstr(z2[ind+1][i][j] <= (1 - sig[ind+1][i]) * ub)

            # partition-based formulation--equations (19)-(21)
            model.addConstr(sum(sum(x[ind][n] * w[n] for n in partitions[j]) - z2[ind+1][i][j] for j in range(len(z2[ind+1][i]))) +
                            sig[ind + 1][i] * b <= 0)
            model.addConstr(sum(z2[ind + 1][i][j] for j in range(len(z2[ind + 1][i]))) + (1 - sig[ind+1][i]) * b >= 0)
            model.addConstr(x[ind+1][i] == sum(z2[ind+1][i][j] for j in range(len(z2[ind+1][i]))) + b * (1 - sig[ind+1][i]))
            
    model.update()

model.setObjective(-(x[ind+1][labels[1]] - x[ind+1][labels[0]]))

model.setParam('MIPFocus',3)
model.setParam('Cuts',1)
model.setParam('Method', 1)
model.setParam('TimeLimit',3600)
#model.setParam('DisplayInterval', 50)
           
model.optimize()
    



        
        
    

