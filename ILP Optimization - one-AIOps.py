#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries

import pandas as pd
import os
import time

from ortools.linear_solver import pywraplp


# In[ ]:


# INPUT PARAMETERS FOR THE MODEL

# Host CPU usage threshold (The CPU usage sould no exceed alpha x 100%)
alpha = 1.0

# General limits 
core_limit = -1
distance_limit = -1

# Select cost function
# cost_function = 1 ==> Optimize number of cores in use (within each cluster)
# cost_function = 2 ==> Optimize load balance (within each cluster)
# cost_function = 3 ==> Optimize allocation distance (within each cluster)
cost_function = 1

# Sets a time limit for running the solver
max_seconds = 120


# In[ ]:


# WORKING directory
work_dir="/Users/rafa/Documents/Investigacion/VM-Traces/aux/"

# INPUT dir: Directory with CPU usage predictions
# One csv file for every VM (vm_id.csv)
# Format of every file: time,cpu
pred_dir = work_dir + "predictions/" 

# INPUT file: VM configuration file (csv)
# Format: vm_id,cores,mem
vm_file = work_dir + "vm_config.csv" 

# INPUT file: Host configuration file (csv)
# Format: cluster_id,host_id,cores,mem
host_file = work_dir + "host_config.csv" 

# INPUT file: current day allocation file (csv)
# Format: vm_id,cluster_id,host_id
alloc_file = work_dir + "current_alloc.csv" 

# OUTPUT file: Next day allocation file (csv)
# Format: vm_id,cluster_id,host_id
new_alloc_file = work_dir + "next_day_alloc.csv" 


# In[ ]:


# Global variables 

# Percentage of CPU used by each VM at each time interval (predicted value)
# vm_cpu{vm_id: [u0, u1, ...,u23]} is the predicted CPU usage of VM vm_id at hour period t
vm_cpu = {} 

# VM config dictionary
# vm_dict{vm_id: [cores, mem]}
vm_config = {}

# VM allocation dictionary (previous day)
# vm_alloc{vm_id: [cluster_id, host_id]}
vm_alloc = {}

# Next day VM allocation list 
# [[vm_id,cluster_id, host_id]]
new_vm_alloc = []

# Host config dictionary
# host_dict{host_id: [cluster_id, cores, mem]}
host_config = {}

# VMs per cluster dictionary
# cluster_vms{cluster_id: [vm_0, vm_1, ...,vm_n]} is the list of VMs running on  cluster_id
cluster_vms = {}  

# Hosts per cluster (organized in a dictionary)
# cluster_hosts{cluster_id: [host_0, host_1, ...,host_n]} is the list of hosts available in  cluster_id
cluster_hosts = {}  

# VM solution dictionary --> It indicates if we have obtained a valid solution (new allocation) for this VM
# vm_solved{vm_id: [True/False]}
vm_solved = {}

# Auxiliar variables

cores_per_vm = []  
mem_per_vm = []  
cores_per_host = []  
mem_per_host = []  
cores_per_host = []  
vm_cpu_usage = []  
alloc = []  
alloc_matrix = []  
host_index = {} 
vm_index = {} 
host_id_list = []  
vm_id_list = []  


# In[ ]:


# Read prediction files and build dictionary vm_cpu
# Predition files contains, for each VM, the predicted cpu usage values (%) for next 24 hours period  

def read_prediction_file():

    global cpu_usage
    
    input_dir = pred_dir

    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            # vm_id = file name without extension .csv
            vm_id = os.path.splitext(file)[0]
            file_path = input_dir + file
            # Create a dataframe from file
            df = pd.read_csv(file_path, delimiter=",", engine='python')
            # convert dataframe to a list 
            forecast = df["cpu"].values.tolist()
            # Create vm_cpu dictionary 
            # vm_cpu{vm_id: [forecast0, forecast1, ..., forecast23]}
            vm_cpu[vm_id] = forecast

    # Check if prediction file of some VM is missing
    # If this case, set the forecast value to 1.0 (100% of CPU usage) for this VM
    for vm_id in vm_config:
        if not vm_id in vm_cpu.keys(): 
            vm_cpu[vm_id] = [1.0] * 24


# In[ ]:


# Read VM config file and build vm_config list
def read_vm_config_file():

    global vm_config
    #global vm_solved
    
    # Read csv file and convert to dataframe.
    # Each entry contains: vm_id, cores, mem
    df = pd.read_csv(vm_file)
    
    # convert dataframe to a list    
    vm_list = df.values.tolist()
    
    # Create VM config dictionary
    # vm_config{vm_id: [cores, mem]}
    vm_config = {} 

    # Create VM solution dictionary --> It indicates if we have obtained a valid solution (new allocation) for this VM
    # vm_solved{vm_id: [True/False]}
    #vm_solved = {}
    
    for item in vm_list:
        vm_id = item[0]
        cores = item[1]
        mem = item[2]
        vm_config[vm_id] = [cores,mem]
        #vm_solved[vm_id] = False
    
    # We compute the total cores and mem used by all the VMs
    #total_cores = 0
    #total_mem = 0
    #for vm in vm_list:
    #    total_cores += vm[1]
    #    total_mem += vm[2]
        
    #print("total cores used: ", total_cores)
    #print("total mem used: ", total_mem)
    


# In[ ]:


# Read Host config file and build host_config and cluster_host dictionaries
def read_host_config_file():

    global host_config
    global cluster_hosts
    
    # Read csv file and convert to dataframe.
    # Each entry contains: cluster_id, host_id, cores, mem
    df = pd.read_csv(host_file)
    
    # convert dataframe to a list
    host_list = df.values.tolist()
    
    # Order host_list by cluster_id index (index 0)
    host_list.sort(key=lambda x: x[0])
    
    # Create cluster_hosts dictionary
    # cluster_hosts{cluster_id: [host_0, host_1, ...,host_n]} is the list of hosts available in cluster_id
    
    # Create host config dictionary
    # host_config{host_id: [cluster_id, cores, mem]}
    
    cluster_hosts = {} 
    host_config = {}

    last_cluster = ""
    for item in host_list:
        cluster_id = item[0]
        host_id = item[1]
        cores = item[2]
        mem = item[3]
        if cluster_id != last_cluster:
            last_cluster = cluster_id
            cluster_hosts[cluster_id] = []
        cluster_hosts[cluster_id].append(host_id)
        host_config[host_id] = [cluster_id, cores, mem]
    
    # We compute the total cores and mem  available in all the physical hosts
    #total_cores = 0  
    #total_mem = 0
    #for host in host_list:
    #    total_cores += host[2]
    #    total_mem += host[3]
        
    #print("total cores available: ", total_cores)
    #print("total mem available: ", total_mem)    


# In[ ]:


# Read previous day allocation file and build allocation list
def read_allocation_file():

    global vm_alloc
    global cluster_vms
    
    # Read csv file and convert to dataframe.
    # Each entry contains: vm_id, cluster_id, host_id
    df = pd.read_csv(alloc_file)
    
    # Convert dataframe to list
    alloc_list = df.values.tolist()
    
    #print(allocation)
    
    # Create VM allocation dictionary (previous day)
    # vm_alloc{vm_id: [cluster_id, host_id]}
    vm_alloc = {}
    for item in alloc_list:
        vm_id = item[0]
        cluster_id = item[1]
        host_id = item[2]
        vm_alloc[vm_id] = [cluster_id,host_id]
    #print(vm_alloc)
    
    
    # Create cluster_vms dictionary
    # cluster_vms{cluster_id: [vm_0, vm_1, ...,vm_n]} is the list of VMs running on cluster_id
    
    # reorder alloc_list by cluster_id index (index 1)
    alloc_list.sort(key=lambda x: x[1])
    
    last_cluster = ""
    cluster_vms = {}
    for item in alloc_list:
        vm_id = item[0]
        cluster_id = item[1]
        if cluster_id != last_cluster:
            last_cluster = cluster_id
            cluster_vms[cluster_id] = []
        cluster_vms[cluster_id].append(vm_id)


# In[ ]:


# Initilize vm_solved dictionary to False
def initialize_vm_solved():

    global vm_solved
    
    # Create VM solution dictionary --> It indicates if we have obtained a valid solution (new allocation) for this VM
    # vm_solved{vm_id: [True/False]}
    #vm_solved = {False for vm_id in vm_config}
    vm_solved = {}
    
    for vm_id in vm_config:
        vm_solved[vm_id] = False
    
    


# In[ ]:


def create_lists_from_dicts(cluster):
    
    # We use the following auxiliary lists (new lists are used for each cluster)     
    # Cores assigned to each VM
    global cores_per_vm
    # Mem assigned to each VM
    global mem_per_vm
    # Cores available in each host
    global cores_per_host
    # Mem available in each host
    global mem_per_host
    # Percentage of cpu usage (predicted, hourly value) for each VM (from hour 0 to hour 23)
    global vm_cpu_usage
    # Physical hosts where each VM is allocated 
    global alloc
    # Allocation matrix (see explanation above)
    global alloc_matrix
    # Dictionary to translate from host_id to host list index 
    global host_index
    # Dictionary to translate from vm_id to vm list index 
    global vm_index
    # List to translate from host list index to host_id 
    global host_id_list
    # List to translate from vm list index to vm_id 
    global vm_id_list
    
    max_vms = len(cluster_vms[cluster])
    max_hosts = len(cluster_hosts[cluster])
    
    # First we define a set of lists obtained from the data of this cluster
    cores_per_vm = [vm_config[vm][0] for vm in cluster_vms[cluster]] 
    mem_per_vm = [vm_config[vm][1] for vm in cluster_vms[cluster]] 
    
    cores_per_host = [host_config[host][1] for host in cluster_hosts[cluster]] 
    mem_per_host = [host_config[host][2] for host in cluster_hosts[cluster]] 
    
    vm_cpu_usage = [vm_cpu[vm] for vm in cluster_vms[cluster]]
    
    host_id_list = cluster_hosts[cluster]
    vm_id_list = cluster_vms[cluster]
    
    # We create a dictionary that assign an index (0..m-1) to each host_id
    # And we create a list with the host_id assiged to each host index

    h = 0
    host_index = {}
    for item in cluster_hosts[cluster]:
        host_index[item] = h
        h += 1
    
    # We assign an index ( 0..n-1) to each vm_id --> Don't need this here
    # And we create a list with the vm_id assiged to each vm index
    v = 0
    vm_index = {}
    for item in cluster_vms[cluster]:
        vm_index[item] = v
        v += 1
    
    # Previous day allocation list (convert host_id to index)
    alloc = []
    for vm in cluster_vms[cluster]:
        host_id = vm_alloc[vm][1]
        alloc.append(host_index[host_id])
    
    #Define previous day allocation matrix
    #alloc_matrix[i, j] = 1 if VM i was mapped to host j
    #                    = 0 otherwise
    alloc_matrix = []
    for i in range(max_vms):
        alloc_matrix.append([])
        for j in range(max_hosts):
            if alloc[i] == j:
                alloc_matrix[i].append(1)
            else:
                alloc_matrix[i].append(0)
                


# In[ ]:


# define model unknowns: x, y

def define_unknowns():

    
    max_vms = len(cores_per_vm)
    max_hosts = len(cores_per_host)
    
    # General Variables

    # x[i, j] = 1 if VM i is mapped to host j.
    x = {}
    for i in range(max_vms):
        for j in range(max_hosts):
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # y[j] = 1 if host j is used.
    y = {}
    for j in range(max_hosts):
        y[j] = solver.IntVar(0, 1, 'y_%i' % j)
        
    return(x, y)


    


# In[ ]:


# We define the three hosts functions used in the model:
# cost_function = 1 ==> total_cores ==> to optimize number of cores in use, within each cluster)
# cost_function = 2 ==> max_load ==> to optimize load balance (within each cluster)
# cost_function = 3 ==> distance ==> to optimize the allocation distance (within each cluster)

def define_cost_functions():

    max_vms = len(cores_per_vm)
    max_hosts = len(cores_per_host)
    
    infinity = solver.infinity()
    
    # Cost funcion 1: total_cores
    total_cores = solver.IntVar(0, infinity, 'total_cores')
    total_cores = solver.Sum([y[j] * cores_per_host[j] for j in range(max_hosts)])

    
    # Cost funcion 2: max_load
    #
    # The average load of every machine is:
    # load_per_core(j) = sum(x[(i, j)] * cores_per_vm[i] * vm_cpu_usage[i][t] for i in range(max_vms) for t in range(24))) / (24 * cores_per_host[j])
    #
    # max_load = Max{load_per_core(j) for j in range(max_hosts)}

    # We define the variable load_per_core

    load_per_core = {}
    for j in range(max_hosts):
        load_per_core[j] = solver.NumVar(0, infinity, 'load_per_core_%i' % j)

    for j in range(max_hosts):
        load_per_core[j] = (solver.Sum([(x[(i, j)] * cores_per_vm[i] * vm_cpu_usage[i][t]) / (24 * cores_per_host[j]) for i in range(max_vms) for t in range(24)]))

    # We transform the function Maximum into a linear funcion
    max_load = solver.NumVar(0, infinity, 'max_load')

    for j in range(max_hosts):
        if y[j]:
            solver.Add(max_load >= load_per_core[j])
       
    # Cost funcion 3: distance
    #
    # The allocation "distance" between the allocation of the previous day and the allocation 
    # of the nex day can be defined as follows:
    # d(i,j) = |x(i,j) - prev(i,j)| 
    # distance = sum(d(i,j) for i in range(max_vms) for j in range(max_hosts)) / 2
    #
    # First we define the variable "d", the variable "d_abs", to transform the absolute value into a linear function
    # d(i,j) = x(i,j) - prev(i,j)
    # d_abs(i,j) = |d(i,j)|

    d = {}
    for i in range(max_vms):
        for j in range(max_hosts):
            d[(i, j)] = solver.IntVar(-1, 1, 'd_%i_%i' % (i, j))

    for i in range(max_vms):
        for j in range(max_hosts):
            d[(i,j)]  = x[(i, j)] - alloc_matrix[i][j]

    d_abs = {}
    for i in range(max_vms):
        for j in range(max_hosts):
            d_abs[(i, j)] = solver.IntVar(0, 1, 'd_abs_%i_%i' % (i, j))

    # Define the constrains that relate d and d_abs
    for i in range(max_vms):
        for j in range(max_hosts):
            solver.Add(d_abs[(i,j)] >= 0)
            solver.Add(d_abs[(i,j)] >= d[(i,j)])
            solver.Add(d_abs[(i,j)] >= -d[(i,j)])

    # Define the variable (allocation) distance 
    distance = solver.IntVar(0, infinity, 'distance')
    distance = solver.Sum([d_abs[(i,j)] for i in range(max_vms) for j in range(max_hosts)]) / 2
    

    return(total_cores, max_load, distance)


# In[ ]:


# We define the constarins for the optimization model

def define_constrains():

    max_vms = len(cores_per_vm)
    max_hosts = len(cores_per_host)
    
    infinity = solver.infinity()
    
    # Each VM must be mapped exactly to one host.
    for i in range(max_vms):
        solver.Add(sum(x[i, j] for j in range(max_hosts)) == 1)
        
    
    # For each host, and each hourly interval, the host CPU usage cannot exceed alpha*cores_per_host 
    for j in range(max_hosts):
        for t in range(24):
            solver.Add(
                 sum(x[(i, j)] * cores_per_vm[i] * vm_cpu_usage[i][t] for i in range(max_vms)) 
                <= y[j] * cores_per_host[j] * alpha)
    
    # For each host, the VM memory usage cannot exceed the memory available
    for j in range(max_hosts):
        solver.Add(
            sum(x[(i, j)] * mem_per_vm[i] for i in range(max_vms)) <= y[j] * mem_per_host[j])
        
    
    # Core limit constrain
    if core_limit >= 0:
        solver.Add(total_cores <= core_limit)

    # Distance limit constrain
    if distance_limit >= 0:
        solver.Add(distance <= distance_limit)


# In[ ]:


# Create next day allocation list 
def create_new_alloc(cluster):
    
    max_vms = len(cores_per_vm)
    max_hosts = len(cores_per_host)
    
    cluster_id = cluster
    
    for j in range(max_hosts):
        if y[j].solution_value() == 1:
            host_id = host_id_list[j]
            for i in range(max_vms):
                if x[i, j].solution_value() > 0:
                    vm_id = vm_id_list[i] 
                    new_vm_alloc.append([vm_id, cluster_id, host_id])
                    vm_solved[vm_id] = True
    


# In[ ]:


# Write next day allocation list to file
def write_new_alloc_file():
    
    # If a given VM is not solved, we use the same allocation than the previous day for this VM
    for vm_id in vm_solved:
        if not vm_solved[vm_id]:
            cluster_id = vm_alloc[vm_id][0]
            host_id = vm_alloc[vm_id][1]
            new_vm_alloc.append([vm_id, cluster_id, host_id])
        
    # Order new_vm_alloc by vm_id index (index 0)
    new_vm_alloc.sort(key=lambda x: x[0])
    
    # Convert list to dataframe
    df = pd.DataFrame(new_vm_alloc)
    # Add headers to dataframe
    df.rename(columns = {0:'vm_id', 1:'cluster_id', 2:'host_id'}, inplace = True)
    # Save dataframe to csv file
    df.to_csv(new_alloc_file, index=False)


# In[ ]:


# Print results
def print_results(cluster):
    
    print()
    print("***********************************")
    print("****", cluster, " allocation ****")
    print("***********************************")
    print()
    
    max_vms = len(cores_per_vm)
    max_hosts = len(cores_per_host)
    
    total_hosts = 0
    total_cores = 0
    total_mem = 0
    alloc_dist = 0
    current_day_alloc = [0] * max_vms
    previous_day_alloc = alloc
    
    for j in range(max_hosts):
        if y[j].solution_value() == 1:
            vms_mapped_to_host = []
            # Host CPU usage using forecasted values of vm_cpu_usage
            host_cpu_usage = [0.0] * 24
            # Host CPU usage values of vm_cpu_usage
            average_load_per_core = 0
            host_mem_used = 0
            for i in range(max_vms):
                if x[i, j].solution_value() > 0:
                    vms_mapped_to_host.append(vm_id_list[i])
                    host_mem_used += mem_per_vm[i]
                    for t in range(24):
                        host_cpu_usage[t] += vm_cpu_usage[i][t] * cores_per_vm[i]
                    current_day_alloc[i] = j
                    if previous_day_alloc[i] != j:
                        alloc_dist += 1
            average_load_per_core = sum(host_cpu_usage[t] for t in range(24)) / (24 * cores_per_host[j])
            for t in range(24):
                host_cpu_usage[t] = round(100*host_cpu_usage[t],1)
            if vms_mapped_to_host:
                total_hosts += 1
                total_cores += cores_per_host[j]
                total_mem += mem_per_host[j]
                print('Host ID', host_id_list[j] , "(" , cores_per_host[j], "vCPUs )")
                print('  VMs mapped:', vms_mapped_to_host)
                print('  % CPU usage per hour period (0-23) - Forecasted:', host_cpu_usage)
                print('  Average load per core - Forecasted:', round(average_load_per_core * 100,1), "%")
                print('  Mem used (GB) / Mem available (GB):', round(host_mem_used/1024,1), " / ", round(mem_per_host[j]/1024,1))
            
                # print(host_cpu_usage)
                print()
    #alloc_dist = alloc_dist / 2
    print()
    print('Total hosts used:', total_hosts)
    print('Total cores allocated:', total_cores)
    print('Total mem allocated (GB):', round(total_mem/1024,1))
    print()
    print('Previous day allocation: ', previous_day_alloc)
    print('Current day allocation:  ', current_day_alloc)        
    print('Allocation distance: ', alloc_dist)
    print()        
    #print('Time = ', solver.WallTime(), ' milliseconds')
    


# In[ ]:


# Call solver
def call_solver():
        
     # Sets a time limit for running the solver
    solver.SetTimeLimit(max_seconds*1000)

    # set a minimum gap limit for the integer solution during branch and cut
    gap = 0.05
    solverParams = pywraplp.MPSolverParameters()
    solverParams.SetDoubleParam(solverParams.RELATIVE_MIP_GAP, gap)
    
    now = time.time()
    
    # Call solver
    status = solver.Solve(solverParams)
    
    elapsed_time = time.time() - now
    print("Solver execution time = ","{:.2f}".format(elapsed_time), "s.")
    
    if status == pywraplp.Solver.INFEASIBLE:
        print("Solution INFEASIBLE for this cluster")
    elif status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution reached for this cluster")
    else:
        print("Sub-optimal solution reached for this cluster")


# In[ ]:


def optimize(cost_function):
    
# cost_function = 1 ==> Optimize number of cores in use (within each cluster)
# cost_function = 2 ==> Optimize load balance (within each cluster)
# cost_function = 3 ==> Optimize allocation distance (within each cluster)

    if cost_function == 1:
        solver.Minimize(total_cores)
        call_solver()
        
    if cost_function == 2:
        solver.Minimize(max_load)
        call_solver()
    
    if cost_function == 3:
        solver.Minimize(distance)
        call_solver()
    
    


# In[ ]:


# Main program

# Read all input files and create dictionaries
read_vm_config_file()
read_host_config_file()
read_allocation_file()
read_prediction_file()


# In[ ]:


# Initialize vm_solved dictitonary to False
initialize_vm_solved()

# Process clusters
for cluster in cluster_vms:
    
    print()
    print("****** Processing ", cluster, " ... " )
    
    # Create auxiliar lists from dictionaries for this cluster
    create_lists_from_dicts(cluster)

    # Create solver instance
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Define unknowns x,y for the model
    x, y = define_unknowns()

    # Define cost functions for the model
    total_cores, max_load, distance = define_cost_functions()
    
    # Define constrains for the model
    define_constrains()
    
    # Call optimization function for this cluster
    optimize(cost_function)
    
    # print results for this cluster
    # print_results(cluster)   
    
    # create next day allocation list for this cluster
    create_new_alloc(cluster)

# Save new allocation to file
write_new_alloc_file()

