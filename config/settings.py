from pathlib import Path
from pydantic import BaseModel, Field


class DatasetSettings(BaseModel):
	train_path: Path
	val_path: Path
	test_path: Path
	train_preprocessing_config: Path
	val_preprocessing_config: Path
	loader_num_workers: int = Field(0, ge=0)
	train_size: float
	batch_size: int = Field(ge=1)


class Settings(BaseModel):
	dataset: DatasetSettings
