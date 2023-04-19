from .sources.one import OneClient
from .sources.prometheus import PrometheusClient
from datetime import datetime
from datetime import timedelta

class DataCollector:
    """Data Collector class to get data from sources"""    
    
    def __init__(self, config: dict):
        # Initialize clients
        self.oneClient = OneClient(
            uri=config.get('ONE_XMLRCP_ENDPOINT'),
            session=f"{config.get('ONE_XMLRCP_USER')}:{config.get('ONE_XMLRCP_PWD')}"
        )
        self.promClient = PrometheusClient(
            uri=config.get('ONE_PROMETHEUS_ENDPOINT'),
        )


    def vm_info(self, vm_id: int =None) -> dict:
        """
        Get VM info from OpenNebula

        Args:
            vm_id (int, optional): VM ID to get only information about a specific VM.
            Defaults to None.

        Returns:
            dict: VM info
        """        
        vm_info = self.oneClient.vm_info()

        if vm_id:
            host_info = host_info.get(vm_id)
        
        return vm_info

    def host_info(self, host_id: int =None) -> dict:
        """
        Get host info from OpenNebula

        Args:
            host_id (int, optional): Host ID to get only information about a specific Host.
            Defaults to None.

        Returns:
            dict: Host info
        """        
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
    def cpu_usage_info(self, start_date: datetime =None, end_date: datetime =None) -> dict:
        """
        Get CPU usage info from prometheus

        Args:
            start_date (datetime, optional): start date. Previous day otherwise.
            end_date (datetime, optional): end date. Today otherwise.

        Returns:
            dict: CPU usage info
        """
        # Generate dates
        start_date = start_date if start_date else datetime.today() - timedelta(days = 2)
        end_date   = end_date if end_date else datetime.today() - timedelta(days = 1)

        # Get data from prometheus
        response = self.promClient.get_metric_range(
            metric_name='opennebula_vm_cpu_ratio',
            start_time=start_date,
            end_time=end_date,
        )

        return self._parse_prom_response(
            response=response,
            index_by='one_vm_id',
        )

    def vm_allocation(self) -> dict:
        """
        Get VM allocation info from OpenNebula

        Returns:
            dict: VM allocation info
        """        
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
            time_format (str, optional): format for the date. Defaults to "%m/%d/%Y %H:%M:%S".

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

            parsed_responde[int(index_id)] = values
        
        return parsed_responde