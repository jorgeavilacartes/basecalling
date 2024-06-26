import inspect
from pathlib import Path
from typing import Optional
from .utils import MonitorValues

class CSVLogger(MonitorValues):

    def __init__(self, list_vars, out_file=Optional[str], overwrite: Optional[bool]=False):
        super().__init__(list_vars)
        self.sep = None # will be set depending on the out file
        self.out_file = out_file
        self.overwrite = overwrite
        
        if out_file:

            if Path(out_file).exists() and overwrite is False:
                assert self._read_colnames(out_file) == list_vars,f"{out_file} exists and has different variables"
            self._create_file(out_file)
        
    def __call__(self,):
        """Collect values of desired variables""" 
        # Get current local values in the namespace
        frame = inspect.currentframe()
        local_values = frame.f_back.f_locals
        
        # Monitor values
        list_values_vars = [local_values.get(var) for var in self.list_vars[1:]] # monitor desired vars
        list_values_vars.insert(0,self._get_localtime()) # add timestamp

        if self.out_file: 
            
            with open(self.out_file, "a") as fp:
                for value in list_values_vars:     
                    fp.write(str(value))
                    fp.write(self.sep)
                fp.write("\n")

        # consolidate timestamp and monitored vars
        self.values.append(self.Vars._make(list_values_vars))

    def _get_sep(self,):
        "Get the right separator for values depending on the out_file (tsv or csv)"
        if self.out_file.endswith(".csv"):
            return str(",")
        elif self.out_file.endswith(".tsv"):
            return str("\t")

    def _create_file(self, out_file):
        """Create the output file to save values
        It can be either a tsv or csv
        """
        if type(out_file) is str and any(out_file.endswith(ext) for ext in [".csv",".tsv"]):
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)

            self.sep=self._get_sep()
            cols = f"{self.sep}".join(self.list_vars)

            # create the file with colnames
            if self.overwrite is True or not Path(out_file).exists():
                with open(Path(out_file),"w") as fp:
                    fp.write(cols)
                    fp.write("\n")

    def _read_colnames(self, out_file):
        with open(out_file) as fp:
            first_line = fp.readline().replace("\n","").split(self._get_sep())
        return first_line