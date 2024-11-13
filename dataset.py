##############################################################
# This is the dataset class for tree cover loss and co2 data
# Author: Xinshuang Liu
# Email: xil235@ucsd.edu
##############################################################

import torch

class TreeCoverLossDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, split_train_test=False, year_threshold=2010):
        self.data = {}
        for line in open(csv_path):
            if not line.startswith("CountryCode"):
                CountryCode, Year, TreeCoverLoss, Co2 = line.split(',')
                if not CountryCode in self.data:
                    self.data[CountryCode] = []
                self.data[CountryCode].append((float(Year), float(TreeCoverLoss), float(Co2)))
        
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
            return self.train_data[self.country_list[idx]], \
                self.test_data[self.country_list[idx]], \
                    self.country_list[idx]

