from typing import List, Literal, Optional, Union

from photoholmes.datasets.base import BaseDataset
from photoholmes.datasets.registry import DatasetName
from photoholmes.preprocessing.pipeline import PreProcessingPipeline


class DatasetFactory:
    @staticmethod
    def load(
        dataset_name: Union[str, DatasetName],
        dataset_dir: str,
        item_data: List[Literal["image", "dct_coefficients", "qtables"]] = ["image"],
        transform: Optional[PreProcessingPipeline] = None,
        mask_transform: Optional[PreProcessingPipeline] = None,
        tampered_only: bool = False,
    ) -> BaseDataset:
        """
        Instantiates and returns a dataset object corresponding to the specified
        dataset name.

        Args:
            dataset_name (Union[str, DatasetName]): The name of the dataset to load.
                Can be a string or a DatasetName enum instance.
            dataset_dir (str): The directory path where the dataset is stored.
            item_data (List[Literal["image", "dct_coefficients", "qtables"]]): A list
                specifying the type of data items to be returned by the dataset.
            transform (Optional[PreProcessingPipeline]): An optional callable that
                takes an input and returns a transformed version. Applied to the
                dataset items.
            mask_transform (Optional[PreProcessingPipeline]): An optional callable that
                takes an input mask and returns a transformed version. Applied to the
                mask of the dataset items.
            tampered_only (bool): A flag indicating whether to load only tampered data
                samples.

        Returns:
            Dataset: An instance of a subclass of photoholmes.datasets.base.BaseDataset
                corresponding to the provided dataset name.

        Raises:
            NotImplementedError: If the dataset name provided is not recognized or not
                implemented.

        Examples:
            >>> dataset = DatasetFactory.load("columbia", "/path/to/columbia/dataset")
            >>> dataset = DatasetFactory.load(DatasetName.COVERAGE,
                    "/path/to/coverage/dataset", item_data=["image", "qtables"],
                    tampered_only=True)
        """
        if isinstance(dataset_name, str):
            dataset_name = DatasetName(dataset_name.lower())

        match dataset_name:
            case DatasetName.COLUMBIA:
                from photoholmes.datasets.columbia import ColumbiaDataset

                return ColumbiaDataset(
                    img_dir=dataset_dir,
                    item_data=item_data,
                    transform=transform,
                    mask_transform=mask_transform,
                    tampered_only=tampered_only,
                )
            case DatasetName.COVERAGE:
                from photoholmes.datasets.coverage import CoverageDataset

                return CoverageDataset(
                    img_dir=dataset_dir,
                    item_data=item_data,
                    transform=transform,
                    mask_transform=mask_transform,
                    tampered_only=tampered_only,
                )
            case DatasetName.OSN:
                from photoholmes.datasets.osn import OSNDataset

                return OSNDataset(
                    img_dir=dataset_dir,
                    item_data=item_data,
                    transform=transform,
                    mask_transform=mask_transform,
                    tampered_only=tampered_only,
                )
            case DatasetName.REALISTIC_TAMPERING:
                from photoholmes.datasets.realistic_tampering import (
                    RealisticTamperingDataset,
                )

                return RealisticTamperingDataset(
                    img_dir=dataset_dir,
                    item_data=item_data,
                    transform=transform,
                    mask_transform=mask_transform,
                    tampered_only=tampered_only,
                )

            case DatasetName.DSO1:
                from photoholmes.datasets.dso1 import DSO1Dataset

                return DSO1Dataset(
                    img_dir=dataset_dir,
                    item_data=item_data,
                    transform=transform,
                    mask_transform=mask_transform,
                    tampered_only=tampered_only,
                )

            case _:
                raise NotImplementedError(
                    f"Dataset '{dataset_name}' is not implemented."
                )
