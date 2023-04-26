from .collectors.collector import DataCollector
from .ml.optimizer import Optimizer

class OneAI:
    """OneAI class to predict CPU usage and optimize VM allocation"""

    def __init__(self, config: dict) -> None:
        self.dataCollector = DataCollector(config=config.get('collectors'))
        self.optimizer     = Optimizer(cost_functions=config.get('cost_functions'))

    def get_cpu_prediction(self) -> dict:
        """
        Get CPU usage prediction

        Returns:
            dict: Dictionary with VM ID as key and predicted CPU usage as value
        """        
        return self.cpu_predicted
    
    def is_reallocated(self, suggested_alloc: list) -> bool:
        """
        Check if VM allocation has changed

        Args:
            suggested_alloc (list): List with the suggested allocation [host_id, vm_id]

        Returns:
            bool: True if VM allocation has changed, False otherwise
        """        
        host_id    = suggested_alloc[0]
        vm_id      = suggested_alloc[1]

        if self.vm_alloc[vm_id][1] != host_id:
            return True
        
        return False

    def suggest_allocation(self) -> list:
        """
        Suggest VM allocation based on CPU usage prediction

        Returns:
            list: List of dictionaries with the optimized allocation for each cost function
        """        
        # Get data from sources
        self.cpu_usage    = self.dataCollector.cpu_usage_info()
        self.vm_info      = self.dataCollector.vm_info()
        self.cluster_info = self.dataCollector.cluster_info()
        self.vm_alloc     = self.dataCollector.vm_allocation()

        # Predict cpu usage based on history
        self.cpu_predicted = self.optimizer.predict_cpu_usage(self.cpu_usage)

        # Based in cpu usage for next day, 
        # optimize the cpu usage and allocation
        allocations = self.optimizer.optimice_allocation(
            self.vm_info, self.cluster_info, self.cpu_predicted, self.vm_alloc
        )

        return allocations
