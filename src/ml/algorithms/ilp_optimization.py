#!/usr/bin/env python
# coding: utf-8

# Import libraries

import pandas as pd
import time

from ortools.linear_solver import pywraplp

class ILPOptimization:
    """ILP Optimization class to solve the VM allocation problem"""    

    def __init__(
            self,
            vm_info: dict, 
            cluster_info: dict, 
            predictions: dict, 
            allocations: dict,
            config: dict,
        ):
        # Solver
        self.solver = pywraplp.Solver.CreateSolver('SCIP')

        # Source data
        # VMs indexed by cluster_id
        self.vm_info      = vm_info
        # Host indexed by cluster_id
        self.cluster_info = cluster_info
        # CPU forecasting
        self.cpu_predictions  = predictions
        # Current allocations
        self.vm_allocations  = allocations

        # Input paramaters for the model
        self.core_limit = config.get('core_limit')
        self.distance_limit = config.get('distance_limit')
        # Host CPU usage threshold (The CPU usage sould no exceed alpha x 100%)
        self.alpha = config.get('alpha_value')
        # Sets a time limit for running the solver
        self.max_seconds = config.get('max_seconds')
        self.cost_function = config.get('id')

        # Init attrs
        self._initialize_vm_solved()


    def _initialize_vm_solved(self) -> None:
        """
        Create VM solution dictionary. It indicates if we have obtained 
        a valid solution (new allocation) for an specific VM.

        vm_solved{vm_id: [True/False]}
        """
        self.vm_solved = dict()

        for vm_id in self.vm_info:
            self.vm_solved[vm_id] = False

    def _initialize_attrs(self, cluster_id: int) -> None:
        """
        Generate stats for a specific cluster

        Args:
            cluster_id (int): Cluster ID
        """
        # Remove VMs without CPU usage predictions
        for vm in self.vm_info[cluster_id].copy().keys():
            if vm not in self.cpu_predictions:
                del self.vm_allocations[vm]
                del self.vm_info[cluster_id][vm]

        self.max_vms   = len(self.vm_info[cluster_id])
        self.max_hosts = len(self.cluster_info[cluster_id])

        self.cores_per_vm = [self.vm_info[cluster_id][vm]['cores'] for vm in self.vm_info[cluster_id]] 
        self.mem_per_vm   = [self.vm_info[cluster_id][vm]['memory'] for vm in self.vm_info[cluster_id]]

        self.cores_per_host = [
            self.cluster_info[cluster_id][host]['cores'] for host in self.cluster_info[cluster_id]
        ]
        self.mem_per_host = [
            self.cluster_info[cluster_id][host]['memory'] for host in self.cluster_info[cluster_id]    
        ]

        self.vm_cpu_usage = [self.cpu_predictions[vm] for vm in self.vm_info[cluster_id]]
        self.host_ids = list(self.cluster_info[cluster_id].keys())
        self.vm_ids  = list(self.vm_info[cluster_id].keys())
        self.allocation_matrix = self._generate_alloc_matrix(cluster_id=cluster_id)


    def _initialize_model(self) -> None:
        # Define unknowns x,y for the model
        self.x, self.y = self._define_unknows()

        # Define cost functions for the model
        self.total_cores, self.max_load, self.distance = self._define_cost_functions()

        # Define constrains for the model
        self._define_constrains()

    def _generate_alloc_matrix(self, cluster_id: int) -> list:
        """
        Generate allocation matrix for a specific cluster

        Args:
            cluster_id (int): Cluster ID

        Returns:
            list: Allocation matrix
        """        
        # Creates a dictionary that assign an index (0..m-1) to each host_id
        # And we create a list with the host_id assiged to each host index
        host_index = self._index_hash(self.cluster_info[cluster_id])

        # Assign an index ( 0..n-1) to each vm_id --> Don't need this here
        # And we create a list with the vm_id assiged to each vm index
        vm_index = self._index_hash(self.vm_info[cluster_id])
        
        # Previous day allocation list (convert host_id to index)
        allocation = list()

        for vm in self.vm_info[cluster_id]:
            host_id = self.vm_allocations[vm][1]
            allocation.append(host_index[host_id])
        
        #Define previous day allocation matrix
        #alloc_matrix[i, j] = 1 if VM i was mapped to host j
        #                    = 0 otherwise
        alloc_matrix = list()

        for i in range(self.max_vms):
            alloc_matrix.append(list())
            for j in range(self.max_hosts):
                if allocation[i] == j:
                    alloc_matrix[i].append(1)
                else:
                    alloc_matrix[i].append(0)

        return alloc_matrix

    def _index_hash(self, hash_to_index: dict) -> dict:
        """
        Assign an index (0..m-1) to each host_id

        Args:
            hash_to_index (dict): Dictionary to index

        Returns:
            dict: Dictionary indexed
        """        
        index = 0
        hash_indexed = dict()
    
        for item in hash_to_index:
            hash_indexed[item] = index
            index = index + 1

        return hash_indexed

    def _define_unknows(self) -> list:
        """
        Defines the unknows of the model:
            - x[i, j] = 1 if VM i is mapped to host j.
            - y[j] = 1 if host j is used.
        Returns:
            list: x, y
        """         
        # x[i, j] = 1 if VM i is mapped to host j.
        x = {}
        for i in range(self.max_vms):
            for j in range(self.max_hosts):
                x[(i, j)] = self.solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

        # y[j] = 1 if host j is used.
        y = {}
        for j in range(self.max_hosts):
            y[j] = self.solver.IntVar(0, 1, 'y_%i' % j)
            
        return(x, y)

    def _define_cost_functions(self) -> list:
        """
        Defines the three hosts functions used in the model:
            - cost_function = 0 ==> total_cores ==> to optimize number of cores in use
                                    (within each cluster)
            - cost_function = 1 ==> max_load ==> to optimize load balance (within each cluster)
            - cost_function = 2 ==> distance ==> to optimize the allocation distance 
                                    (within each cluster)

        Returns:
            list: [total_cores: int, max_load: int, distance: int]
        """
        infinity = self.solver.infinity()

        # Cost function 1: total_cores
        total_cores = self.solver.IntVar(0, infinity, 'total_cores')
        total_cores = self.solver.Sum([self.y[j] * self.cores_per_host[j] for j in range(self.max_hosts)])

        # Cost funcion 2: max_load
        #
        # The average load of every machine is:
        # load_per_core(j) = sum(x[(i, j)] * cores_per_vm[i] * vm_cpu_usage[i][t] for i 
        # in range(max_vms) for t in range(24))) / (24 * cores_per_host[j])
        #
        # max_load = Max{load_per_core(j) for j in range(max_hosts)}
        # We define the variable load_per_core

        load_per_core = dict()

        for j in range(self.max_hosts):
            load_per_core[j] = self.solver.NumVar(0, infinity, 'load_per_core_%i' % j)

        for j in range(self.max_hosts):
            load_per_core[j] = (
                self.solver.Sum(
                [(self.x[(i, j)] * self.cores_per_vm[i] * self.vm_cpu_usage[i][t]) / (24 * self.cores_per_host[j]) 
                 for i in range(self.max_vms) for t in range(24)]
                )
            )

        # We transform the function Maximum into a linear funcion
        max_load = self.solver.NumVar(0, infinity, 'max_load')

        for j in range(self.max_hosts):
            if self.y[j]:
                self.solver.Add(max_load >= load_per_core[j])
        
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

        d = dict()

        for i in range(self.max_vms):
            for j in range(self.max_hosts):
                d[(i, j)] = self.solver.IntVar(-1, 1, 'd_%i_%i' % (i, j))

        for i in range(self.max_vms):
            for j in range(self.max_hosts):
                d[(i,j)]  = self.x[(i, j)] - self.allocation_matrix[i][j]

        d_abs = dict()

        for i in range(self.max_vms):
            for j in range(self.max_hosts):
                d_abs[(i, j)] = self.solver.IntVar(0, 1, 'd_abs_%i_%i' % (i, j))

        # Define the constrains that relate d and d_abs
        for i in range(self.max_vms):
            for j in range(self.max_hosts):
                self.solver.Add(d_abs[(i,j)] >= 0)
                self.solver.Add(d_abs[(i,j)] >= d[(i,j)])
                self.solver.Add(d_abs[(i,j)] >= -d[(i,j)])

        # Define the variable (allocation) distance 
        distance = self.solver.IntVar(0, infinity, 'distance')
        distance = self.solver.Sum(
            [d_abs[(i,j)] for i in range(self.max_vms) for j in range(self.max_hosts)]
        ) / 2

        return(total_cores, max_load, distance)

    def _define_constrains(self) -> None:
        """
        Defines the constrains for the optimization model
        """
        infinity = self.solver.infinity()
    
        # Each VM must be mapped exactly to one host.
        for i in range(self.max_vms):
            self.solver.Add(sum(self.x[i, j] for j in range(self.max_hosts)) == 1)
            
        
        # For each host, and each hourly interval, the host CPU usage cannot exceed alpha*cores_per_host 
        for j in range(self.max_hosts):
            for t in range(24):
                self.solver.Add(
                    sum(self.x[(i, j)] * self.cores_per_vm[i] * self.vm_cpu_usage[i][t] 
                        for i in range(self.max_vms)
                    ) <= self.y[j] * self.cores_per_host[j] * self.alpha)
        
        # For each host, the VM memory usage cannot exceed the memory available
        for j in range(self.max_hosts):
            self.solver.Add(
                sum(self.x[(i, j)] * self.mem_per_vm[i] 
                    for i in range(self.max_vms)) 
                <= self.y[j] * self.mem_per_host[j])
        
        # Core limit constrain
        if self.core_limit >= 0:
            self.solver.Add(self.total_cores <= self.core_limit)

        # Distance limit constrain
        if self.distance_limit >= 0:
            self.solver.Add(self.distance <= self.distance_limit)

    def _create_new_alloc(self, cluster_id: int) -> list:
        """
        Creates a new allocation for the given cluster

        Args:
            cluster_id (int): Cluster ID

        Returns:
            list: New allocation for the given cluster
        """        

        new_vm_alloc = list()

        # If a given VM is solved, we add the new allocation to the new_vm_alloc list
        for j in range(self.max_hosts):
            if self.y[j].solution_value() == 1:
                host_id = self.host_ids[j]
                for i in range(self.max_vms):
                    if self.x[i, j].solution_value() > 0:
                        vm_id = self.vm_ids[i] 
                        new_vm_alloc.append([host_id, vm_id])
                        self.vm_solved[vm_id] = True

        # If a given VM is not solved, we use the same allocation than the previous day for this VM
        for vm_id in self.vm_solved:
            if not self.vm_solved[vm_id]:
                cluster_id = self.vm_allocations[vm_id][0]
                host_id = -1
                new_vm_alloc.append([host_id, vm_id])

        # Order new_vm_alloc by vm_id index (index 1)
        new_vm_alloc.sort(key=lambda x: x[1])
        
        return new_vm_alloc

    def _call_solver(self):
        """
        Calls the solver
        """        
        # Sets a time limit for running the solver
        self.solver.SetTimeLimit(self.max_seconds*1000)

        # set a minimum gap limit for the integer solution during branch and cut
        gap = 0.05
        solverParams = pywraplp.MPSolverParameters()
        solverParams.SetDoubleParam(solverParams.RELATIVE_MIP_GAP, gap)
        
        now = time.time()
        
        # Call solver
        status = self.solver.Solve(solverParams)
        
        elapsed_time = time.time() - now
        print('\n--------------------------------------')
        print('\nCost function used = ', self.cost_function)
        print("\nSolver execution time = ","{:.2f}".format(elapsed_time), "s.")
        
        if status == pywraplp.Solver.INFEASIBLE:
            print("Solution INFEASIBLE for this cluster")
        elif status == pywraplp.Solver.OPTIMAL:
            print("Optimal solution reached for this cluster")
        else:
            print("Sub-optimal solution reached for this cluster")

    def optimize(self, cluster_id: int = 0) -> list:
        """
        Optimize the allocation for the given cluster
            - cost_function = 0 ==> Optimize number of cores in use (within each cluster)
            - cost_function = 1 ==> Optimize load balance (within each cluster)
            - cost_function = 2 ==> Optimize allocation distance (within each cluster)

        Args:
            cluster_id (int, optional): Cluster ID. Defaults to 0.

        Returns:
            list: New allocation for the given cluster
        """        
        self._initialize_attrs(cluster_id=cluster_id)
        self._initialize_model()

        if self.cost_function == 0:
            self.solver.Minimize(self.total_cores)
            self._call_solver()
            
        if self.cost_function == 1:
            self.solver.Minimize(self.max_load)
            self._call_solver()
        
        if self.cost_function == 2:
            self.solver.Minimize(self.distance)
            self._call_solver()

        return self._create_new_alloc(cluster_id)

