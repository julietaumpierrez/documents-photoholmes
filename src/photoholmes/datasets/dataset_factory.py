from typing import List, Literal, Union

from photoholmes.data.input.registry import DatasetName


class DatasetFactory:
    @staticmethod
    def load(
        dataset_name: Union[str, DatasetName],
        dataset_dir: str,
        item_data: List[Literal["image", "dct_coefficients", "qtables"]] = ["image"],
        transform=None,
        mask_transform=None,
        tampered_only: bool = False,
    ):
        """Instantiates a dataset corresponding to the name passed."""
        if isinstance(dataset_name, str):
            dataset_name = DatasetName(dataset_name.lower())

        match dataset_name:
            case DatasetName.COLUMBIA:
                from photoholmes.data.input.columbia import ColumbiaDataset

                return ColumbiaDataset(
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
