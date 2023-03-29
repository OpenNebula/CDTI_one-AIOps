from collectors.sources.one import OneClient
from collectors.sources.prometheus import PrometheusClient
from datetime import date
from datetime import datetime
from datetime import timedelta

class DataCollector:
    
    def __init__(self):
        self.oneClient = OneClient(
            uri="http://localhost:2633/RPC2",
            session="oneadmin:opennebula"
        )
        self.promClient = PrometheusClient(
            uri="http://localhost:9090"
        )

    # Returns host info
    # Format: {'cluster_id': {'host_id': {'cores', 'mem}}}
    def vm_info(self, vm_id=None):
        vm_info = self.oneClient.vm_info()

        if vm_id:
            host_info = host_info.get(vm_id)
        
        return vm_info

    # Returns host info
    # Format: {'cluster_id': {'host_id': {'cores', 'mem}}}
    def host_info(self, host_id=None):
        host_info = self.oneClient.host_info()

        if host_id:
            host_info = host_info.get(host_id)
        
        return host_info
    
    # Returns cluster info
    # Format: {'cluster_id': {'host_id': {'cores', 'mem}}}
    def cluster_info(self, cluster_id=None):
        cluster_info = self.oneClient.cluster_info()

        if cluster_id:
            cluster_info = cluster_info.get(cluster_id)
        
        return cluster_info
    
    #Format: time,cpu_usage
    def cpu_usage_info(self, start_date = None, end_date = None):
        previous_day = date.today() - timedelta(days = 1)
        response = self.promClient.get_metric_range(
            metric_name='opennebula_vm_cpu_ratio',
            start_time= datetime(2023,3,27),
            end_time= datetime(2023,3,28)
        )
        return self._parse_prom_response(
            response=response,
            index_by='one_vm_id'
        )

    # Previous day allocation file
    # Format: vm_id,cluster_id,host_id
    def vm_allocation(self):
        return self.oneClient.vm_allocation()
        
    def _parse_prom_response(
            self,
            response: list, 
            index_by: str, 
            time_format: str = "%m/%d/%Y %H:%M:%S"
        ) -> dict:
        """
        Parse API response from prometheus to dictionary indexed by resource ID

        Args:
            response (list): prom client raw response
            index_by (str): key ID name of the resource
            time_format (_type_, optional): format for the date. Defaults to "%m/%d/%Y %H:%M:%S".

        Returns:
            dict: parse response indexed by resource ID
        """
        parsed_responde = dict()

        for item in response:
            index_id = item['metric'][index_by]
            values   = item['values']

            # Parse timestamp to str datetime
            for value in values:
                value[0] = datetime.fromtimestamp(value[0]).strftime(time_format)

            parsed_responde[index_id] = values
        
        return parsed_responde