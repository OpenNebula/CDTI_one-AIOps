from .algorithms.cpu_forecaster import CpuForecaster
from .algorithms.ilp_optimization import ILPOptimization

class Optimizer:
    """Predictor class to predict CPU usage and optimize VM allocation"""    
    
    def __init__(self, cost_functions: list):
        self.forecaster = CpuForecaster()
        self.cost_functions = cost_functions

    def predict_cpu_usage(self, cpu_usage: dict) -> dict:
        """
        Predict CPU usage for the next 24 hours

        Args:
            cpu_usage (dict): Dictionary with VM ID as key and historic CPU usage as value

        Returns:
            dict: Dictionary with VM ID as key and predicted CPU usage as value
        """        
        return self.forecaster.forecast(cpu_usage)
    
    def optimice_allocation(
            self,
            vm_info: dict,
            cluster_info: dict,
            cpu_prediction: dict,
            current_alloc: dict,
        ) -> list:
        """
        Optimize VM allocation

        Args:
            vm_info (dict): VM information (cpu, memory) in each cluster
            cluster_info (dict): Host information (cpu, memory) in each cluster
            cpu_prediction (dict): CPU usage prediction for the next 24 hours
            current_alloc (dict): Current VM allocation

        Returns:
            list: List of dictionaries with the optimized allocation for each cost function
        """ 
        results = list()
  
        for cost_function in self.cost_functions:     
            # Create the optimizer by cluster
            optimizer = ILPOptimization(
                    vm_info=vm_info,
                    cluster_info=cluster_info,
                    predictions=cpu_prediction,
                    allocations=current_alloc,
                    config=cost_function
            )
            func_result = dict()

            # Process clusters
            for cluster_id in cluster_info.keys():
                func_result[cluster_id] = optimizer.optimize(
                    cluster_id=cluster_id
                )

            cost_function['results'] = func_result
            results.append(cost_function)

        return results
