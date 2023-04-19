from .algorithms.cpu_forecaster import CpuForecaster
from .algorithms.ilp_optimization import ILPOptimization

class Predictor:
    """Predictor class to predict CPU usage and optimize VM allocation"""    
    
    def __init__(self, config: dict):
        self.forecaster = CpuForecaster()
        self.config = config

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
        ) -> dict:
        """
        Optimize VM allocation

        Args:
            vm_info (dict): VM information (cpu, memory) in each cluster
            cluster_info (dict): Host information (cpu, memory) in each cluster
            cpu_prediction (dict): CPU usage prediction for the next 24 hours
            current_alloc (dict): Current VM allocation

        Returns:
            dict: Dictionary with cluster ID as key and optimized VM allocation as value
        """        
        # Create the optimizer by cluster
        optimizer = ILPOptimization(
                vm_info=vm_info,
                cluster_info=cluster_info,
                predictions=cpu_prediction,
                allocations=current_alloc,
                config=self.config
        )

        results = dict()

        # Process clusters
        for cluster_id in cluster_info.keys():
            results[cluster_id] = optimizer.optimize(
                cluster_id=cluster_id
            )

        return results


