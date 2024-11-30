############################################################################
# This is the dataset class for tree cover loss, co2 and driver types data
# Author: Xinshuang Liu, Haoyu Hu
# Email: xil235@ucsd.edu, hah034@ucsd.edu
############################################################################

import torch


class TreeCoverLossDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for Tree Cover Loss and CO2 data grouped by country.

    Attributes:
        data (dict): Dictionary with country codes as keys and data as values.
        country_list (list): List of all country codes sorted alphabetically.
        train_data (dict): Training data split by country (if split_train_test=True).
        test_data (dict): Testing data split by country (if split_train_test=True).
        split_train_test (bool): Whether to split the data into train and test sets.

    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        split_train_test (bool): Whether to split the data into train and test sets.
        year_threshold (int): Year threshold to separate training and testing data.
    """

    def __init__(self, csv_path, split_train_test=False, year_threshold=2010):
        # Initialize dataset attributes
        self.data = {}

        # Load data from the CSV file
        for line in open(csv_path):
            # Skip the header row
            if not line.startswith("CountryCode"):
                CountryCode, Year, TreeCoverLoss, Co2 = line.split(",")

                # Ensure the country code exists in the dictionary
                if not CountryCode in self.data:
                    self.data[CountryCode] = []

                # Append the data as a tuple
                self.data[CountryCode].append(
                    (float(Year), float(TreeCoverLoss), float(Co2))
                )

        # Sort countries alphabetically and ensure data is sorted by year
        self.country_list = sorted(list(self.data.keys()))
        for country in self.country_list:
            self.data[country].sort(key=lambda x: x[0])  # Sort by year

        # Split into training and testing sets if requested
        if not split_train_test:
            for country in self.country_list:
                # Convert data to tensors
                self.data[country] = torch.tensor(self.data[country])
        else:
            self.train_data = {}
            self.test_data = {}

            # Split each country's data
            for country in self.country_list:
                self.train_data[country] = []
                self.test_data[country] = []

                current_data = self.data[country]
                for i in range(len(current_data)):
                    # Split based on year_threshold
                    if current_data[i][0] <= year_threshold:
                        self.train_data[country].append(current_data[i])
                    else:
                        self.test_data[country].append(current_data[i])

                # Convert splits to tensors
                self.train_data[country] = torch.tensor(self.train_data[country])
                self.test_data[country] = torch.tensor(self.test_data[country])

        self.split_train_test = split_train_test

    def __len__(self):
        """
        Returns the number of countries in the dataset.
        """
        return len(self.country_list)

    def __getitem__(self, idx):
        """
        Retrieves data for a specific country by index.

        Args:
            idx (int): Index of the country.

        Returns:
            tuple: (data, country_code) if split_train_test=False.
                   (train_data, test_data, country_code) if split_train_test=True.
        """
        if not self.split_train_test:
            return self.data[self.country_list[idx]], self.country_list[idx]
        else:
            return (
                self.train_data[self.country_list[idx]],
                self.test_data[self.country_list[idx]],
                self.country_list[idx],
            )


class DriverTypeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for Tree Cover Loss and CO2 data grouped by driver type.

    Attributes:
        data (dict): Dictionary with driver types as keys and data as values.
        driver_list (list): List of all driver types sorted alphabetically.
        train_data (dict): Training data split by driver type (if split_train_test=True).
        test_data (dict): Testing data split by driver type (if split_train_test=True).
        split_train_test (bool): Whether to split the data into train and test sets.

    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        split_train_test (bool): Whether to split the data into train and test sets.
        year_threshold (int): Year threshold to separate training and testing data.
    """

    def __init__(self, csv_path, split_train_test=False, year_threshold=2010):
        # Initialize dataset attributes
        self.data = {}

        # Load data from the CSV file
        for line in open(csv_path):
            # Skip the header row
            if not line.startswith("DriverType"):
                driver, year, tree_cover_loss, co2 = line.split(",")

                # Ensure the driver exists in the dictionary
                if not driver in self.data:
                    self.data[driver] = []

                # Append the data as a tuple
                self.data[driver].append(
                    (float(year), float(tree_cover_loss), float(co2))
                )

        # Sort driver types alphabetically and ensure data is sorted by year
        self.driver_list = sorted(list(self.data.keys()))
        for driver in self.data.keys():
            self.data[driver].sort(key=lambda x: x[0])  # Sort by year

        # Split into training and testing sets if requested
        if not split_train_test:
            for k, v in self.data.items():
                # Convert data to tensors
                self.data[k] = torch.tensor(v)
        else:
            self.train_data = {}
            self.test_data = {}

            # Split each driver's data
            for driver in self.data.keys():
                self.train_data[driver] = []
                self.test_data[driver] = []

                current_data = self.data[driver]
                for i in range(len(current_data)):
                    # Split based on year_threshold
                    if current_data[i][0] <= year_threshold:
                        self.train_data[driver].append(current_data[i])
                    else:
                        self.test_data[driver].append(current_data[i])

                # Convert splits to tensors
                self.train_data[driver] = torch.tensor(self.train_data[driver])
                self.test_data[driver] = torch.tensor(self.test_data[driver])

        self.split_train_test = split_train_test

    def __len__(self):
        """
        Returns the number of driver types in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves data for a specific driver type by index.

        Args:
            index (int): Index of the driver type.

        Returns:
            tuple: (data, driver_type) if split_train_test=False.
                   (train_data, test_data, driver_type) if split_train_test=True.
        """
        if not self.split_train_test:
            return self.data[self.driver_list[index]], self.driver_list[index]
        else:
            return (
                self.train_data[self.driver_list[index]],
                self.test_data[self.driver_list[index]],
                self.driver_list[index],
            )
