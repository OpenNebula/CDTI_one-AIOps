import pyone

class OneClient:
    """OpenNebula client"""    

    def __init__(self, uri: str, session: str):
      self.client = pyone.OneServer(
          uri=uri,
          session=session
      )

    def vm_allocation(self) -> dict:
        """
        Generate a dictionary with all the VMs allocated

        Returns:
            dict: VMs allocated indexed by Cluster ID
        """        
        vmpool = self.client.vmpool.info(-1, -1, -1, -1)
        vm_alloc = dict()

        # Iterate over vms and index by cluster
        for vm in vmpool.VM:
            vm_id      = int(vm.ID)
            cluster_id = int(vm.HISTORY_RECORDS.HISTORY[-1].CID)
            host_id    = int(vm.HISTORY_RECORDS.HISTORY[-1].HID)
            
            vm_alloc[vm_id] = [cluster_id, host_id]

        return vm_alloc


    def vm_info(self) -> dict:
        """
        Generate a dictionary with all the VMs allocated

        Returns:
            dict: VMs allocated indexed by Cluster ID
        """    
        vmpool = self.client.vmpool.info(-1, -1, -1, -1)
        vm_info = dict()

        # Iterate over vms and index by cluster
        for vm in vmpool.VM:
            vm_id      = int(vm.ID)
            cluster_id = int(vm.HISTORY_RECORDS.HISTORY[-1].CID)

            output = {
                "cores": float(vm.TEMPLATE.get('CPU')),
                "memory": int(vm.TEMPLATE.get('MEMORY')),
            }

            if cluster_id in vm_info:
                vm_info[cluster_id][vm_id] = output
            else:
                vm_info[cluster_id] = dict()
                vm_info[cluster_id][vm_id] = output
        return vm_info

    def host_info(self) -> dict:
        """
        Generate a dictionary with all the VMs allocated in each host

        Returns:
            dict: VMs indexed by Host
        """
        vmpool = self.client.vmpool.info(-1, -1, -1, -1)
        host_info = dict()

        # Iterate over vms and index by host
        for vm in vmpool.VM:
            vm_id   = int(vm.ID)
            host_id = int(vm.HISTORY_RECORDS.HISTORY[-1].HID)

            output = {
                "cores": float(vm.TEMPLATE.get('CPU')),
                "memory": int(vm.TEMPLATE.get('MEMORY')),
            }
     
            if host_id in host_info:
                host_info[host_id][vm_id] = output
            else:
                host_info[host_id] = dict()
                host_info[host_id][vm_id] = output

        return host_info

    def cluster_info(self) -> dict:
        """
        Generate a dictionary with all the hosts inside a cluster

        Returns:
            dict: Hosts indexed by Cluster
        """
        hostpool = self.client.hostpool.info()
        cluster_info = dict()

        # Iterate over host and index by cluster
        for host in hostpool.HOST:
            # Get output information
            cluster_id = int(host.CLUSTER_ID)
            host_id    = int(host.ID)

            output = {
                "id": host_id,
                "cores": int(host.HOST_SHARE.VMS_THREAD),
                "memory": int(host.HOST_SHARE.MAX_MEM)
            }

            if host.CLUSTER_ID in cluster_info:
                cluster_info[cluster_id][host_id] = output
            else:
                cluster_info[cluster_id] = dict()
                cluster_info[cluster_id][host_id] = output

        return cluster_info
