import time
import configparser
from prometheus_client.core import GaugeMetricFamily, REGISTRY
from prometheus_client import start_http_server
from src.oneai import OneAI

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')


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
        self.metrics = [
            Metric(
                name='opennebula_vm_suggested_allocation',
                type='gauge',
                docstr='Suggested Host allocation for Virtual Machines',
                value=lambda v: str(v[0]),
                labels={'one_vm_id': lambda v: str(v[1])}
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
        oneai = OneAI(config)
        alloc_results = oneai.predict_usage()

        # Iterate over clusters
        for cluster_id in alloc_results:
            # Iterate over allocations in each cluster
            for alloc in alloc_results[cluster_id]:
                # Generate metrics for each allocation
                for metric in self.metrics:
                    gauge = GaugeMetricFamily(metric.name, metric.docstr, labels=metric.labels)
                    gauge.add_metric([metric.labels['one_vm_id'](alloc)], metric.value(alloc))
                    yield gauge


def main():
    """Main entry point"""
    exporter = OneAICollector(exporter_port=config.getint('EXPORTER', 'PORT'))
    REGISTRY.register(exporter)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()