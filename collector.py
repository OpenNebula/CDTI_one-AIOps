import time
import json
import datetime
from datetime import timezone
from prometheus_client.core import GaugeMetricFamily, REGISTRY
from prometheus_client import start_http_server
from src.oneai import OneAI

# Read configuration
with open('config.json') as config_file:
    config         = json.load(config_file)

class Metric:
    """Prometheus Metric class to store metric information"""
    
    def __init__(self, name, type, docstr, value, labels):
        self.name = name
        self.type = type
        self.docstr = docstr
        self.value = value
        self.labels = labels    


class OneAICollector:
    """OneAI Collector class to collect metrics from OneAI and export them to Prometheus"""    

    def __init__(self, exporter_port: int):
        # Define metrics
        self.allocation_metrics = [
            Metric(
                name='opennebula_vm_suggested_allocation',
                type='gauge',
                docstr='Suggested Host allocation for Virtual Machines',
                value=lambda v: str(v[0]),
                labels={'one_vm_id': lambda v: str(v[1]), 'func': lambda v: str(v), 'is_change': lambda v: str(v)}
            )
        ]
        self.cpu_metrics = [
            Metric(
                name='opennebula_vm_predicted_cpu_usage',
                type='gauge',
                docstr='Predicted CPU usage for Virtual Machines',
                value=lambda v: str(v),
                labels={'one_vm_id': lambda v: str(v), 'timestamp': lambda v: str(v)}
            )
        ]
        # Start exporter
        start_http_server(exporter_port)

    def collect(self) -> GaugeMetricFamily:
        """
        Collect metrics from OneAI and export them to Prometheus

        Yields:
            Iterator[GaugeMetricFamily]: Prometheus metric
        """     
        print('Collecting metrics...')

        oneai = OneAI(config)
        alloc_results = oneai.suggest_allocation()
        cpu_predictions = oneai.get_cpu_prediction()

        # CPU metrics
        for vm_id, vm_usage in cpu_predictions.items():
            # Iterate over cpu predictions
            for index, cpu_value in enumerate(vm_usage):                
                # Generate metrics for each cpu prediction
                for metric in self.cpu_metrics:
                    gauge = GaugeMetricFamily(metric.name, metric.docstr, labels=metric.labels)
                    gauge.add_metric(
                        [metric.labels['one_vm_id'](vm_id)], 
                        metric.value(cpu_value)
                    )
                    yield gauge

        # Allocation metrics
        for cost_function in alloc_results:
            func_results = cost_function.get('results')
            func_id      = cost_function.get('id')
            # Iterate over clusters
            for cluster_id in func_results:
                # Iterate over allocations in each cluster
                for alloc in func_results[cluster_id]:
                    # Generate metrics for each allocation
                    for metric in self.allocation_metrics:
                        gauge = GaugeMetricFamily(metric.name, metric.docstr, labels=metric.labels)
                        gauge.add_metric(
                            [
                                metric.labels['one_vm_id'](alloc),
                                metric.labels['func'](func_id),
                                metric.labels['is_change'](oneai.is_reallocated([alloc[0], alloc[1]]))
                            ],
                            metric.value(alloc)
                        )
                        yield gauge

def main():
    """Main entry point"""
    exporter = OneAICollector(exporter_port=config.get('exporter_port'))
    REGISTRY.register(exporter)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()