#!/usr/bin/env python
# coding: utf-8

# Import libraries

import pandas as pd
import os
import time

from ortools.linear_solver import pywraplp
from helper import print_values

class ILPOptimization:

    def __init__(
            self,
            vm_info: dict, 
            cluster_info: dict, 
            predictions: dict, 
            allocations: dict,
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
        self.core_limit = -1
        self.distance_limit = -1
        # Host CPU usage threshold (The CPU usage sould no exceed alpha x 100%)
        self.alpha = 1.0
        # Sets a time limit for running the solver
        self.max_seconds = 120

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

        self.vm_cpu_usage = [self.cpu_predictions[str(vm)] for vm in self.vm_info[cluster_id]]
        self.host_ids = list(self.cluster_info[cluster_id].keys())
        self.vm_ids  = list(self.vm_info[cluster_id].keys())
        self.allocation_matrix = self._generate_alloc_matrix(cluster_id=cluster_id)

        # Debug 
        print_values('max_vms', self.max_vms )
        print_values('max_hosts', self.max_hosts)
        print_values('cores_per_vm', self.cores_per_vm)
        print_values('mem_per_vm', self.mem_per_vm)
        print_values('cores_per_host', self.cores_per_host)
        print_values('mem_per_host', self.mem_per_host)
        print_values('vm_cpu_usage', self.vm_cpu_usage)
        print_values('host_ids', self.host_ids)
        print_values('vm_ids', self.vm_ids)
        print_values('allocation_matrix', self.allocation_matrix)

    def _initialize_model(self) -> None:
        # Define unknowns x,y for the model
        self.x, self.y = self._define_unknows()

        # Define cost functions for the model
        self.total_cores, self.max_load, self.distance = self._define_cost_functions()

        # Define constrains for the model
        self._define_constrains()

    def _generate_alloc_matrix(self, cluster_id: int) -> None:
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

    def _index_hash(self, hash_to_index: hash) -> None:
        index = 0
        hash_indexed = dict()
    
        for item in hash_to_index:
            hash_indexed[item] = index
            index = index + 1

        return hash_indexed

    def _define_unknows(self):
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
            - cost_function = 1 ==> total_cores ==> to optimize number of cores in use
                                    (within each cluster)
            - cost_function = 2 ==> max_load ==> to optimize load balance (within each cluster)
            - cost_function = 3 ==> distance ==> to optimize the allocation distance 
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
        Create next day allocation list
        """
        new_vm_alloc = list()

        for j in range(self.max_hosts):
            if self.y[j].solution_value() == 1:
                host_id = self.host_ids[j]
                for i in range(self.max_vms):
                    if self.x[i, j].solution_value() > 0:
                        vm_id = self.vm_ids[i] 
                        new_vm_alloc.append([vm_id, cluster_id, host_id])
                        self.vm_solved[vm_id] = True

        # If a given VM is not solved, we use the same allocation than the previous day for this VM
        for vm_id in self.vm_solved:
            if not self.vm_solved[vm_id]:
                cluster_id = self.vm_allocations[vm_id][0]
                host_id = self.vm_allocations[vm_id][1]
                new_vm_alloc.append([vm_id, cluster_id, host_id])
            
        # Order new_vm_alloc by vm_id index (index 0)
        new_vm_alloc.sort(key=lambda x: x[0])

        self.print_results(cluster_id)
        
        return new_vm_alloc

    def _call_solver(self):
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
        print("\nSolver execution time = ","{:.2f}".format(elapsed_time), "s.")
        
        if status == pywraplp.Solver.INFEASIBLE:
            print("Solution INFEASIBLE for this cluster")
        elif status == pywraplp.Solver.OPTIMAL:
            print("Optimal solution reached for this cluster")
        else:
            print("Sub-optimal solution reached for this cluster")

    def optimize(self, cluster_id: int = 0, cost_function: int = 1) -> None:
        """
        - cost_function = 1 ==> Optimize number of cores in use (within each cluster)
        - cost_function = 2 ==> Optimize load balance (within each cluster)
        - cost_function = 3 ==> Optimize allocation distance (within each cluster)
        """
        self._initialize_attrs(cluster_id=cluster_id)
        self._initialize_model()

        if cost_function == 1:
            self.solver.Minimize(self.total_cores)
            self._call_solver()
            
        if cost_function == 2:
            self.solver.Minimize(self.max_load)
            self._call_solver()
        
        if cost_function == 3:
            self.solver.Minimize(self.distance)
            self._call_solver()

        return self._create_new_alloc(cluster_id)

    def print_results(self, cluster_id: int) -> None:       
        print()
        print("***********************************")
        print("****", cluster_id, " allocation ***")
        print("***********************************")
        print()

        total_hosts = 0
        total_cores = 0
        total_mem = 0
        alloc_dist = 0
        current_day_alloc = [0] * self.max_vms
        previous_day_alloc = self.vm_allocations
        
        for j in range(self.max_hosts):
            if self.y[j].solution_value() == 1:
                vms_mapped_to_host = []
                # Host CPU usage using forecasted values of vm_cpu_usage
                host_cpu_usage = [0.0] * 24
                # Host CPU usage values of vm_cpu_usage
                average_load_per_core = 0
                host_mem_used = 0

                for i in range(self.max_vms):
                    if self.x[i, j].solution_value() > 0:
                        vms_mapped_to_host.append(self.vm_ids[i])
                        host_mem_used += self.mem_per_vm[i]
                        for t in range(24):
                            host_cpu_usage[t] += self.vm_cpu_usage[i][t] * self.cores_per_vm[i]
                        current_day_alloc[i] = j
                        if previous_day_alloc[i] != j:
                            alloc_dist += 1
                average_load_per_core = sum(host_cpu_usage[t] for t in range(24)) / (24 * self.cores_per_host[j])
                for t in range(24):
                    host_cpu_usage[t] = round(100*host_cpu_usage[t],1)
                if vms_mapped_to_host:
                    total_hosts += 1
                    total_cores += self.cores_per_host[j]
                    total_mem += self.mem_per_host[j]
                    print('Host ID', self.host_ids[j] , "(" , self.cores_per_host[j], "vCPUs )")
                    print('  VMs mapped:', vms_mapped_to_host)
                    print('  % CPU usage per hour period (0-23) - Forecasted:', host_cpu_usage)
                    print('  Average load per core - Forecasted:', round(average_load_per_core * 100,1), "%")
                    print('  Mem used (GB) / Mem available (GB):', round(host_mem_used/1024,1), " / ", round(self.mem_per_host[j]/1024,1))
                
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
