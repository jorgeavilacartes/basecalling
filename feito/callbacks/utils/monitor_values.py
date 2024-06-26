import time # For timestamp
import inspect # To access local values of variables
from typing import List
from collections import namedtuple # To collect monitored values

class MonitorValues:
    """Monitor values of variables
        Returns a list of namedtuples with a timestamp and all values
        of the variables monitored. 
        Usage: 
        _________________________________________________
        # 1. Instantiate class with the desired variables
        mv = MonitorValues(["var1","var2"])
        # Call the instance
        mv()
        _________________________________________________
        At this point, the instance 'mv' should have a row with a namedtuple
        and the corresponding values of the variables 'var1' and 'var2' and a timestamp
        automatically generated when this is called.
        """
    def __init__(self, list_vars: List[str]):
        """Initialize class with list of variables to monitor. 
        If a variable does not exists when the monitor is called, 
        it will be set to None at that time.
        Args:
            list_vars (list): list with the name of variables to monitor as strings.
                                Eg: list_vars = ["x","y,"z"]
        """         
        # List of variables to monitor
        self.list_vars = list_vars
        self.list_vars.insert(0,"timestamp") if "timestamp" not in self.list_vars else None

        # namedtuple to save variables
        self.values = []
        self.Vars = namedtuple("Vars", self.list_vars)
        
    def __call__(self,):   
        """Collect values of desired variables""" 
        # Get current local values in the namespace
        frame = inspect.currentframe()
        local_values = frame.f_back.f_locals
        
        # Monitor values
        list_values_vars = [local_values.get(var) for var in self.list_vars[1:]] # monitor desired vars
        list_values_vars.insert(0,self._get_localtime()) # add timestamp
        # consolidate timestamp and monitored vars
        self.values.append(self.Vars._make(list_values_vars))

    def __repr__(self,):
        """Print how class was initialized"""
        return f"MonitorValues({self.list_vars!r})"

    def __len__(self,):
        """Number of monitored calls"""
        return len(self.values)

    def __getitem__(self, n: int):
        """Get the n-esim monitored value"""
        return self.values[int(n)]

    def _get_localtime(self,):
        """Get a string representation of current time 
        <DayName> <Month> <NumberDay> hh:mm:ss <Year>"""
        return time.asctime(time.localtime(time.time()))

    def get_values(self,):
        """Return the list of monitored values"""
        return self.values