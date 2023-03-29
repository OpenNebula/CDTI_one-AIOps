
from prometheus_api_client.prometheus_connect import PrometheusConnect

class PrometheusClient:
    def __init__(self, uri):
        self.prom = PrometheusConnect(url=uri)
        
    def get_metric_value(self, metric_name):
        """Returns the value of a given metric name"""
        return self.prom.get_current_metric_value(metric_name)
    
    def get_metric_range(self, metric_name, start_time, end_time):
        """Returns a list of metric values within a given time range"""
        return self.prom.custom_query_range(
            query=metric_name,
            start_time=start_time,
            end_time=end_time,
            step='1h'
        )
