############################################################################
# This is the dataset class for tree cover loss, co2 and driver types data
# Author: Xinshuang Liu, Haoyu Hu
# Email: xil235@ucsd.edu, hah034@ucsd.edu
############################################################################

import torch


class TreeCoverLossDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, split_train_test=False, year_threshold=2010):
        self.data = {}
        for line in open(csv_path):
            if not line.startswith("CountryCode"):
                CountryCode, Year, TreeCoverLoss, Co2 = line.split(",")
                if not CountryCode in self.data:
                    self.data[CountryCode] = []
                self.data[CountryCode].append(
                    (float(Year), float(TreeCoverLoss), float(Co2))
                )

        self.country_list = sorted(list(self.data.keys()))
        for country in self.country_list:
            self.data[country].sort(key=lambda x: x[0])

        if not split_train_test:
            for country in self.country_list:
                self.data[country] = torch.tensor(self.data[country])
        else:
            self.train_data = {}
            self.test_data = {}
            for country in self.country_list:
                self.train_data[country] = []
                self.test_data[country] = []

                current_data = self.data[country]
                for i in range(len(current_data)):
                    if current_data[i][0] <= year_threshold:
                        self.train_data[country].append(current_data[i])
                    else:
                        self.test_data[country].append(current_data[i])

                self.train_data[country] = torch.tensor(self.train_data[country])
                self.test_data[country] = torch.tensor(self.test_data[country])

        self.split_train_test = split_train_test

    def __len__(self):
        return len(self.country_list)

    def __getitem__(self, idx):
        if not self.split_train_test:
            return self.data[self.country_list[idx]], self.country_list[idx]
        else:
            return (
                self.train_data[self.country_list[idx]],
                self.test_data[self.country_list[idx]],
                self.country_list[idx],
            )


class DriverTypeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, split_train_test=False, year_threshold=2010):
        # data: {"driver": [(year, tree_cover_loss, co2), ...], ...}
        self.data = {}
        for line in open(csv_path):
            if not line.startswith("DriverType"):
                driver, year, tree_cover_loss, co2 = line.split(",")
                if not driver in self.data:
                    self.data[driver] = []
                self.data[driver].append(
                    (float(year), float(tree_cover_loss), float(co2))
                )

        self.driver_list = sorted(list(self.data.keys()))

        # Sort each value in self.data by year
        for driver in self.data.keys():
            self.data[driver].sort(key=lambda x: x[0])

        if not split_train_test:
            for k, v in self.data.items():
                self.data[k] = torch.tensor(v)
        else:
            self.train_data = {}
            self.test_data = {}

            for driver in self.data.keys():
                self.train_data[driver] = []
                self.test_data[driver] = []

                current_data = self.data[driver]
                for i in range(len(current_data)):
                    if current_data[i][0] <= year_threshold:
                        self.train_data[driver].append(current_data[i])
                    else:
                        self.test_data[driver].append(current_data[i])

                self.train_data[driver] = torch.tensor(self.train_data[driver])
                self.test_data[driver] = torch.tensor(self.test_data[driver])

        self.split_train_test = split_train_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.split_train_test:
            return self.data[self.driver_list[index]], self.driver_list[index]
        else:
            return (
                self.train_data[self.driver_list[index]],
                self.test_data[self.driver_list[index]],
                self.driver_list[index],
            )
