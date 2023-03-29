import configparser

from collectors.collector import DataCollector
from ml.predictor import Predictor
from exporters.exporter import OneAIExporter
from helper import print_values

CONF_FILE = 'oneai.conf'

# Create objects
dataCollector = DataCollector()
predictor     = Predictor()
exporter      = OneAIExporter()

# Get data from sources
cpu_usage    = dataCollector.cpu_usage_info()
vm_info      = dataCollector.vm_info()
cluster_info = dataCollector.cluster_info()
vm_alloc     = dataCollector.vm_allocation()

# Debug (print values)
print_values('CPU_USAGE', cpu_usage)
print_values('VM_INFO', vm_info)
print_values('CLUSTER_INFO', cluster_info)
print_values('VM_ALLOCATION', vm_alloc)

# Predict cpu usage based on history
cpu_predicted = predictor.predict_cpu_usage(cpu_usage)

# Debug (print values)
print_values('CPU_PREDICTED', cpu_predicted)

# Based in cpu usage for next day, 
# optimize the cpu usage and allocation
allocations = predictor.optimice_allocation(
    vm_info, cluster_info, cpu_predicted, vm_alloc
)

# Debug (print values)
print_values('ALLOCATIONS', allocations)

# Export the data to Prometheus
# exporter.export_metrics(allocations)
