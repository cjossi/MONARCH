# Standard library imports
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import yaml


@dataclass(frozen = True)
class Config:
    """
    Configuration class for the MONARCH project.
    This class is responsible for loading and storing configuration parameters
    from a YAML file.
    """

    gait_parameters_folder: Path

    @staticmethod
    def load_from_yaml(file_path: str | Path = "config.yaml") -> "Config":
        """
        Load configuration parameters from a YAML file.

        Parameters
        ----------
        file_path : str | Path, optional
            The path to the YAML configuration file.

        Returns
        -------
        Config
            An instance of the Config class with loaded parameters.
        """
        config_path = Path(file_path)
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        path_fields = {
            key: Path(value)
            for key, value in config_data.items()
        }

        return Config(**path_fields)