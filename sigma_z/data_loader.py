import sys
from abc import ABC, abstractmethod

from astropy.io import fits

# If GCR Catalogs is not up to date, uncomment this and use your own.
# my_local_gcrcatalogs = '/global/u1/m/mkwiecie/desc/repos/gcr-catalogs'
# sys.path.insert(0, my_local_gcrcatalogs)
import GCRCatalogs


class DataLoader(ABC):
    @abstractmethod
    def get_values(self, value_names):
        pass

    @classmethod
    def from_gcr(cls, catalog_name):
        return GCRDataLoader(catalog_name)

    @classmethod
    def from_fits(cls, file_name):
        return FitsDataLoader(file_name)


class GCRDataLoader(DataLoader):
    def __init__(self, catalog_name):
        self.catalog_name = catalog_name

    def get_values(self, value_names):
        catalog = GCRCatalogs.load_catalog(self.catalog_name)
        values = catalog.get_quantities(value_names)
        return values


class FitsDataLoader(DataLoader):
    def __init__(self, file_name):
        self.file_name = file_name

    def get_values(self, value_names):
        hdu_list = fits.open(self.file_name)

        return_values = {}
        for name in value_names:
            return_values[name] = hdu_list[1].data[name]

        del hdu_list
        return return_values
