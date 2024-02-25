import yaml
from pathlib import Path
from .settings import Settings

def get_config(config_file: Path) -> Settings:
	with open(config_file, 'r') as file:
		try:
			parsed_yaml_config = yaml.safe_load(file)
		except yaml.YAMLError as exception:
			print(exception)
	return Settings(**parsed_yaml_config)
