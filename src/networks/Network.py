import os
import subprocess

from dataclasses import dataclass, field , asdict
from typing import Optional, Dict, Any, Tuple , Union , Literal
import json
import pandas as pd
# from pyhlicorn.post_inference import _adjacency_list,_coregulators


class AdjList(dict):
    def __init__(self, inferenceResult: pd.DataFrame, precompute: bool = True,):
        self.__inferenceResult = inferenceResult
        self._bygene: Optional[Dict[str, Any]] = None
        self._bytf: Optional[Dict[str, Any]] = None

        if precompute:
            # self._bygene, self._bytf = _adjacency_list(self.__inferenceResult)
            super().__init__({"bygene": self._bygene, "bytf" : self._bytf})
        else:
            super().__init__()

    def _ensure_computed(self):
        if self._bygene is None or self._bytf is None:
            # self._bygene, self._bytf = _adjacency_list(self.__inferenceResult)
            self.update({"bygene": self._bygene, "bytf": self._bytf})

    @property
    def bygene(self) -> Dict[str, Any]:
        self._ensure_computed()
        return self._bygene

    @property
    def bytf(self) -> Dict[str, Any]:
        self._ensure_computed()
        return self._bytf

    def __getitem__(self, key):
        self._ensure_computed()
        return super().__getitem__(key)

    def __repr__(self):
        self._ensure_computed()
        return super().__repr__()
   
    @classmethod
    def _set_adjacency_lists(cls, bygene: Dict[str, Any], bytf: Dict[str, Any]):
        instance = cls.__new__(cls)  # Create an instance without calling __init__
        instance._bygene = bygene
        instance._bytf = bytf
        instance.update({"bygene": instance._bygene, "bytf": instance._bytf})
        return instance
        


@dataclass
class Network:
    """
    A class to represent a gene regulatory network (GRN) with associated metadata, 
    inference parameters, adjacency lists, and coregulators.
    
    GRN : pd.DataFrame
        A DataFrame representing the Gene Regulatory Network. Each row corresponds to 
        a target gene with information about its regulators and associated coefficients.
    metadata : dict, optional
        A dictionary containing metadata about the network.
    inferenceParameters : dict, optional
        A dictionary containing parameters used during the inference process of the GRN.
    """
    
    def __init__(
        self,
        GRN : pd.DataFrame,
        metadata : dict = None,
        inferenceParameters : dict = None
    ):
        
        self._GRN = GRN
        self._metadata = metadata
        self._inferenceParameters = inferenceParameters
        
        self._adjlist = AdjList(self._GRN, precompute=True)
        
        self._coregs = None
        self.coregsinfo = None
        
        self.__interactions()  
        self._GRN.set_index("Target",inplace=True)
        print(self)
        

        
    
    def __interactions(self : 'Network') ->  'Network':
        
        if self._metadata is None:
            self._metadata = {'gene_list' : None, 'tf_list': None}
        
        interactions = []
        for targets,regs in self.adjlist.bygene.items():
            interactions.append(len(regs['act']) + len(regs['rep']))
        
        self._metadata['interactions'] = sum(interactions)
        self._metadata['transcription_factors'] = len(self.adjlist.bytf.keys())
        self._metadata['target_genes'] = len(self.adjlist.bygene.keys())
        
        
    def __str__(self):
        return f"{self._metadata['transcription_factors']} Transcription Factors.  {self._metadata['target_genes']} Target Genes.  {self._metadata['interactions']} Regulatory interactions."
    
    def __repr__(self):
        return f"{self._metadata['transcription_factors']} Transcription Factors.  {self._metadata['target_genes']} Target Genes.  {self._metadata['interactions']} Regulatory interactions."
    
    @property
    def inferenceParameters(self):
        """
        A dictionary containing parameters used during the inference process. 
        
        Returns
        -------
        dict
            A dictionary of inference parameters used during the GRN inference process.
        
        Examples
        --------
        
        >>> network = inference.fit(numerical_expression, tf_list)
        >>> net.inferenceParameters
        {
            'min_gene_support': 0.1, 
            'min_coreg_support': 0.1,
            'max_coreg': None,
            'search_thresh': 0.3333333333333333,
            'nGRN': 100
        }
        """
        return self._inferenceParameters
        
    @property
    def metadata(self):
        """
        Metadata for the Gene Regulatory Network (GRN).

        This dictionary contains detailed information about the GRN, including:
            - The number of target genes, transcription factors, and regulatory interactions.

        Returns
        -------
        dict
            A dictionary containing metadata about the GRN.

        Examples
        --------
        >>> network = inference.fit(numerical_expression, tf_list)
        >>> network.metadata
        {
            'gene_list': 1,
            'tf_list': 65,
            'transcription_factors': 3,
            'target_genes': 1,
            'interactions': 3,
        }
        """
        
        
        return self._metadata
        
    @property
    def adjlist(self):
        """
        Returns the adjacency list representation of the Gene Regulatory Network (GRN). 

        The adjacency list is a dictionary containing two components:
            - 'bygene': A dictionary of gene-to-gene interactions.
            - 'bytf': A dictionary of transcription factor (TF)-to-gene interactions.

        **The structure of the adjacency list is as follows:**

        adjlist : dict of dicts
            A dictionary representing the adjacency list of the GRN. Each key in the outer dictionary 
            corresponds to a gene or TF, and the corresponding value is another dictionary describing its 
            interactions with other genes or TFs. The components are:
        
        
            bygene : dict of dict
                A dictionary where each key is a target gene, and its corresponding value is another
                dictionary with two possible keys: 'act' and 'rep'. The 'act' key maps to a set of strings
                representing activators (transcription factors or other genes) that activate the target gene.
                The 'rep' key maps to a set of strings representing repressors that repress the target gene.

            bytf : dict of dict
                A dictionary where each key is a transcription factor (TF), and its corresponding value is another
                dictionary with two possible keys: 'act' and 'rep'. The 'act' key maps to a set of strings
                representing genes activated by this transcription factor. The 'rep' key maps to a set of strings
                representing genes repressed by this transcription factor.

        Returns
        -------
        adjlist : dict
            A dictionary containing the adjacency list of the GRN, structured as described above.
        
        Examples
        --------
        Acessing bygene and bytf separately. 

        >>> network = inference.fit(numerical_expression, tf_list, gene_list = ["VANGL2"], collect = True)
        >>> net.adjlist.bygene
        {'VANGL2': {'act': ['IRX3', 'HOXC6'], 'rep': ['VGLL1']}}
        >>> net.adjlist.bytf
        {'IRX3': {'act': ['VANGL2'], 'rep': []}, 
        'VGLL1': {'act': [], 'rep': ['VANGL2']}, 
        'HOXC6': {'act': ['VANGL2'], 'rep': []}}
        >>> net.adjlist
        {
            'bygene': {'VANGL2': {'act': ['HOXC6', 'IRX3'], 'rep': ['VGLL1']}},
            'bytf': {
                    'HOXC6': {'act': ['VANGL2'], 'rep': []},
                    'VGLL1': {'act': [], 'rep': ['VANGL2']},
                    'IRX3': {'act': ['VANGL2'], 'rep': []}
                    }
        }
        """
        
        if self._adjlist is None :
            self._adjlist = AdjList(self._GRN, precompute=True)
        return self._adjlist
        
    @property
    def coregs(self):
        """
        A dataframe specifying inferred co-regulators for each pair of genes or transcription factors. This dataframe can include:

        +----------------+----------------+-------------------------------------------+
        | Column         | Type           | Description                               |
        +================+================+===========================================+
        | Reg1           | str            | Regulator One                             |
        +----------------+----------------+-------------------------------------------+
        | Reg2           | str            | Regulator Two                             |
        +----------------+----------------+-------------------------------------------+
        | support        | float          | The support value for both co-regulators. |
        +----------------+----------------+-------------------------------------------+
        | nGRN           | int            | The number of gene regulatory networks.   |
        +----------------+----------------+-------------------------------------------+
        | fisherTest     | float          | Fisher's exact test p-value calculated    |
        |                |                | using the `fisher_exact,                  |
        |                |                | alternative='greater'`                    |
        |                |                | function from `scipy.stats`.              |
        +----------------+----------------+-------------------------------------------+
        | adjustedPvalue | float          | Adjusted p-values calculated using the    |
        |                |                | Holm method from                          |
        |                |                | `statsmodels.stats.multitest`.            |
        +----------------+----------------+-------------------------------------------+

        Examples
        --------

        +--------+--------+--------+---------+-------------------+----------------+
        | Reg1   | Reg2   | support| nGRN    | fisherTest        | adjustedPvalue |
        +--------+--------+--------+---------+-------------------+----------------+
        | TF1    | TF2    | 0.85   | 3       | 0.001             | 0.005          |
        +--------+--------+--------+---------+-------------------+----------------+
        | TF3    | TF4    | 0.78   | 2       | 0.002             | 0.01           |
        +--------+--------+--------+---------+-------------------+----------------+
        """
        return self._coregs

    @coregs.setter
    def coregs(self, value):
        self._coregs = value
        

    @property
    def GRN(self):
        """
        Gene Regulatory Network (GRN)

        A `pandas.DataFrame` representing the inferred gene regulatory network.

        Each row corresponds to a target gene and includes information about its
        activators, repressors, and their respective coefficients, as well as coefficients
        performance metrics.
        
        +----------------+----------------+-------------------------------------------+
        | Column         | Type           | Description                               |
        +================+================+===========================================+
        | Target         | str            | Name of the target gene                   |
        +----------------+----------------+-------------------------------------------+
        | Co-act         | list[str]      | Co-activators for each target gene        |
        +----------------+----------------+-------------------------------------------+
        | Co-rep         | list[str]      | Co-repressors for each target gene        |
        +----------------+----------------+-------------------------------------------+
        | Coef.Acts      | list[float]    | Activator coefficients                    |
        +----------------+----------------+-------------------------------------------+
        | Coef.Reps      | list[float]    | Repressor coefficients                    |
        +----------------+----------------+-------------------------------------------+
        | Coef.coActs    | float          | Co-activator coefficients                 |
        +----------------+----------------+-------------------------------------------+
        | Coef.coReps    | float          | Co-repressor coefficients                 |
        +----------------+----------------+-------------------------------------------+
        | R2             | float          | Coefficient of determination (R² score)   |
        +----------------+----------------+-------------------------------------------+
        | RMSE           | float          | Root Mean Square Error                    |
        +----------------+----------------+-------------------------------------------+

        Examples
        --------

        +--------+--------------+--------------+-------------+-------------+---------------+---------------+------+-------+
        | Target | Co-act       | Co-rep       | Coef.Acts   | Coef.Reps   | Coef.coActs   | Coef.coReps   | R2   | RMSE  |
        +--------+--------------+--------------+-------------+-------------+---------------+---------------+------+-------+
        | geneA  | ["A1", "A2"] | ["R1"]       | [0.8, 0.2]  | [0.5]       | 0.3           | 0.1           | 0.91 | 0.04  |
        +--------+--------------+--------------+-------------+-------------+---------------+---------------+------+-------+
        | geneB  | ["A3"]       | ["R2", "R3"] | [0.6]       | [0.3, 0.2]  | 0.25          | 0.15          | 0.87 | 0.06  |
        +--------+--------------+--------------+-------------+-------------+---------------+---------------+------+-------+
        
        """ 
        return self._GRN

 
    def asdict(self) -> dict:
        """
        Convert the object into a dictionary representation.
        
        Returns
        -------
        dict
            A dictionary with keys:
            
            - "metadata": Metadata result of the object.
            - "GRN": GRN inference result.
            - "inferenceParameters": Parameters used for inference.
            - "adjlist": Dictionary with keys "bygene" and "bytf".
            - "coregs": Co-regulators data.
            - "coregsinfo": Additional information about co-regulators.
        """

        orient = "records" # TODO : Change to User paramter after making sure its stable
        return {
            "metadata": self._metadata,
            "GRN": self._GRN.reset_index().to_dict(orient=orient),
            "inferenceParameters": self._inferenceParameters,
            "adjlist": {
                "bygene": self.adjlist.bygene if self.adjlist else None,
                "bytf": self.adjlist.bytf if self.adjlist else None,
            } if self._adjlist else None,
            "coregs": self.coregs.to_dict(orient=orient) if self.coregs is not None else None,
            "coregsinfo": self.coregsinfo,
        }

    @classmethod
    def fromdict(cls, data: dict) -> 'Network':
        """
        Create a Network instance from a dictionary representation.
        
        Parameters
        ----------
        
        data : dict
            - "GRN": A dictionary representing the gene regulatory network, 
              which will be converted to a pandas DataFrame and indexed by "Target".
            - "metadata": Metadata associated with the network.
            - "inferenceParameters": Parameters used for inference.
            - "adjlist" (optional): A dictionary containing adjacency lists 
              with keys 'bygene' and 'bytf'.
            - "coregs" (optional): A dictionary representing coregulators, 
              which will be converted to a pandas DataFrame.
            - "coregsinfo" (optional): Additional information about coregulators.
            
        Returns
        -------
        
        Network
            An instance of the Network class populated with the provided data.
        """
        GRN = pd.DataFrame.from_dict(data["GRN"]).set_index("Target")
        metadata = data["metadata"]
        inferenceParameters = data["inferenceParameters"]
        
        adjlist = data.get("adjlist", None)
        coregs = pd.DataFrame.from_dict(data["coregs"]) if data.get("coregs") else None
        
        network = cls.__new__(cls)  # Create an instance without calling __init__
        network._GRN = GRN
        network._metadata = metadata
        network._inferenceParameters = inferenceParameters
        network._adjlist = None
        network._coregs = coregs
        network.coregsinfo = data.get("coregsinfo", None)
        
        if adjlist:
            network._adjlist = AdjList._set_adjacency_lists(bygene=adjlist['bygene'], bytf=adjlist['bytf'])
        
        return network
    
    
    def save_as_json(self, file_path: str = "network.json") -> None:
        """
        Save the network data as a JSON file.
        
        Parameters
        ----------
        file_path : str, optional
            The file path where the JSON file will be saved. Defaults to "network.json".
            
        Examples
        --------
        
        >>> network = inference.fit(numerical_expression, tf_list)
        >>> network.save_as_json("network.json")
        
        """
        
        orient = "records" # TODO : Change to User paramter after making sure its stable
        
        with open(file_path, "w") as f:
            json.dump(self.asdict(), f, indent=4)

    @classmethod
    def load_from_json(cls, file_path: str) -> 'Network':
        """
        Load a Network instance from a JSON file.
        
        Parameters
        ----------
        
        file_path : str
            The path to the JSON file containing the network data.
            
        Returns
        -------
        
        Network
            An instance of the Network class populated with data from the JSON file.
            
        Raises
        ------
        
        FileNotFoundError
            If the specified file does not exist.
        json.JSONDecodeError
            If the file is not a valid JSON.
            
        Examples
        --------
        
        >>> network.load_from_json("network.json")
        >>> network
        63 Transcription Factors.  867 Target Genes.  7921 Regulatory interactions.
        
        """
        
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.fromdict(data)
                    
        
    def coregulators(
        self,
        max_coreg: int = 2,
        min_coreg: int = 2,
        min_common_genes: int = 10,
        alternative= 'greater',
        adjustMethod: str = "holm",
        n_jobs: int = -1,
        backend: Optional[Literal['loky', 'multiprocessing', 'sequential', 'threading']] = 'loky',
    ):
        """
        Infer coregulators based on specified parameters.
        
        Parameters
        ----------
        max_coreg : int, optional
            Maximum number of coregulators to consider, by default 2.
        min_coreg : int, optional
            Minimum number of coregulators to consider, by default 2.
        min_common_genes : int, optional
            Minimum number of common genes required for coregulation, by default 10.
        alternative : {'two-sided', 'less', 'greater'}, optional
            Specifies the alternative hypothesis for statistical testing, by default 'greater'.
        adjustMethod : {'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg',
            'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'}, optional
            Method for p-value adjustment in multiple testing, by default "holm".
        n_jobs : int, optional
            Number of parallel jobs to run. Use -1 to use all available processors, by default -1.
        backend : {'loky', 'multiprocessing', 'sequential', 'threading'}, optional
            Backend to use for parallel processing, by default 'loky'.
        
        Returns
        -------
        self
            The instance of the class with updated coregulators and coregulator information.
        
        Raises
        ------
        ValueError
            If `alternative` is not one of {'two-sided', 'less', 'greater'}.
        ValueError
            If `adjustMethod` is not one of the supported methods.
        """
        
        if alternative not in ['two-sided', 'less', 'greater']:
            raise ValueError(f"Invalid value for alternative: {alternative}. Must be one of ['two-sided', 'less', 'greater']")
        
        if adjustMethod not in ["bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky"]:
            raise ValueError(f"Invalid value for adjustMethod: {adjustMethod}. Must be one of the supported methods.")
        
        # self._coregs, self.coregsinfo = _coregulators(
        #     self,
        #     max_coreg=max_coreg,
        #     min_coreg=min_coreg,
        #     min_common_genes=min_common_genes,
        #     alternative=alternative,
        #     adjustMethod=adjustMethod,
        #     joblib = (n_jobs,backend)
        # )
        return self._coregs    

    def get_regulators(self, target_gene: str) -> dict:
        """
        Retrieves regulators for a given target gene from the Gene Regulatory Network (GRN).

        Parameters
        ----------
        
        target_gene : str
            The name of the target gene for which regulators are to be fetched.

        Returns
        -------
        
        dict
            A dictionary with two keys:
            
            - 'act': A list of activators for the target gene.

            - 'rep': A list of repressors for the target gene.
        """

        if not hasattr(self, 'adjlist') or not hasattr(self.adjlist, 'bygene'):
            self._adjlist = AdjList(self._GRN)

        bygene = self.adjlist.bygene
        
        if target_gene not in bygene:
            return {'act': [], 'rep': []}
        
        return bygene[target_gene]

    def get_targets(self, regulator : str)-> dict:
        """
        Retrieves the target genes regulated by a specific regulator from the Gene Regulatory Network (GRN).

        Parameters
        ----------
        regulator : str
            The name of the regulator (e.g., transcription factor) whose target genes are to be fetched.

        Returns
        -------
        dict
            - 'act': list
              A list of target genes activated by the regulator.

            - 'rep': list
              A list of target genes repressed by the regulator.       
        """

        if not hasattr(self, 'adjlist') or not hasattr(self.adjlist, 'bygene'):
            self._adjlist = AdjList(self._GRN)

        bytf = self.adjlist.bytf
        
        if regulator not in bytf:
            return {'act': [], 'rep': []}
        
        return bytf[regulator]
    
    
    def visualize(self, file_path = "network.json"):
        if hasattr(self, 'GRN'):
            self.save_as_json(file_path)
        elif not os.path.exists(file_path):
            print("GRN not found at:",file_path)
            return             
        print("Starting Flask Server....")
        env=os.environ.copy()
        env["GRN_PATH"]=os.path.abspath(file_path)
        subprocess.run(["python", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "visualizations", "app.py"))], env=env)
