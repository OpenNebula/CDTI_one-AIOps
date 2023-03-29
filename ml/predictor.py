from .algorithms.cpu_forecaster import CpuForecaster
from .algorithms.ilp_optimization import ILPOptimization
from ortools.linear_solver import pywraplp

class Predictor:
    
    def __init__(self):
        self.forecaster = CpuForecaster()
        self.cost_function = 1

    def predict_cpu_usage(self, cpu_usage):
        return self.forecaster.forecast(cpu_usage)
    
    def optimice_allocation(
            self,
            vm_info: dict,
            cluster_info: dict,
            cpu_prediction: dict,
            current_alloc: dict,
        ) -> None:
        # Create the optimizer by cluster
        optimizer = ILPOptimization(
                vm_info=vm_info,
                cluster_info=cluster_info,
                predictions=cpu_prediction,
                allocations=current_alloc,
        )

        results = dict()

        # Process clusters
        for cluster_id in cluster_info.keys():
            print("****** Processing ", cluster_id, " ... " )
            results[cluster_id] = optimizer.optimize(
                cluster_id=cluster_id, 
                cost_function=self.cost_function,
            )

        return results


