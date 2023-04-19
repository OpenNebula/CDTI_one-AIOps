from .collectors.collector import DataCollector
from .ml.predictor import Predictor

class OneAI:
    """OneAI class to predict CPU usage and optimize VM allocation"""

    def __init__(self, config: dict) -> None:
        self.dataCollector = DataCollector(config=config['OPENNEBULA'])
        self.predictor     = Predictor(config=config['ONEAI'])

    def predict_usage(self) -> dict:
        """
        Predict the next day cpu usage

        Returns:
            dict: Dictionary with cluster ID as key and optimized VM allocation as value
        """        
        # Get data from sources
        cpu_usage    = self.dataCollector.cpu_usage_info()
        vm_info      = self.dataCollector.vm_info()
        cluster_info = self.dataCollector.cluster_info()
        vm_alloc     = self.dataCollector.vm_allocation()

        # Predict cpu usage based on history
        cpu_predicted = self.predictor.predict_cpu_usage(cpu_usage)

        # Based in cpu usage for next day, 
        # optimize the cpu usage and allocation
        allocations = self.predictor.optimice_allocation(
            vm_info, cluster_info, cpu_predicted, vm_alloc
        )

        return allocations
