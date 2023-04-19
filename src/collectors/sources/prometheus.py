
from prometheus_api_client.prometheus_connect import PrometheusConnect
from pyparsing import Any
import datetime

class PrometheusClient:
    """Prometheus Client class to connect to Prometheus and get metrics"""

    def __init__(self, uri):
        self.prom = PrometheusConnect(url=uri)
        
    def get_metric_value(self, metric_name: str) -> Any:
        """
        Returns the value of a given metric name

        Args:
            metric_name (str): name of the metric

        Returns:
            Any: value of the metric
        """        
        return self.prom.get_current_metric_value(metric_name)
    
    def get_metric_range(self, metric_name: str, start_time: datetime, end_time: datetime) -> list:
        """
        Returns a list of values for a given metric name between two dates

        Args:
            metric_name (str): name of the metric
            start_time (datetime): start date
            end_time (datetime): end date

        Returns:
            list: list of values
        """        
        return self.prom.custom_query_range(
            query=metric_name,
            start_time=start_time,
            end_time=end_time,
            step='1h'
        )
