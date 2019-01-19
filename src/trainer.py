import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from statistics import mean


class Trainer:
    def __init__(self, model_a, model_b, dataloader, continuous_dataloader, val_continuous_dataloader, val_dataloader):
        self.model_a = model_a
        self.model_b = model_b
        self.dataloader = dataloader
        self.continuous_dataloader = continuous_dataloader
        self.val_dataloader = val_dataloader
        self.val_continuous_dataloader = val_continuous_dataloader
        self.lambda_ = 1e+7

        self.criterion = nn.NLLLoss()
        self.optimizer_a = optim.Adam(self.model_a.parameters(), lr=1e-2)
        self.optimizer_b = optim.Adam(self.model_b.parameters(), lr=1e-2)

    def train(self, n_epoch, n_c_epoch):

        for epoch in range(n_epoch):
            self.model_a.train()
            images, labels, loss = self._init_learning()
            accuracy, continuous_accuracy = self.validate_model_a()
            print("Epoch:{} loss:{} accuracy:{} con accuracy:{}".format(epoch, loss, accuracy, continuous_accuracy))

        fisher_information = self._calc_fisher_information(images, labels)
        init_parameters = self.init_params()
        self.model_b.load_state_dict(self.model_a.state_dict())
        self.optimizer_b.zero_grad()
        for epoch in range(n_c_epoch):
            self.model_b.train()
            loss, ewc_loss, classification_loss = self._continuous_learning(fisher_information, init_parameters)
            accuracy, continuous_accuracy = self.validate_model_b()
            print("Epoch:{} loss:{} accuracy:{} con accuracy:{} ewc:{} nll:{}".format(epoch, loss,
                accuracy, continuous_accuracy, ewc_loss, classification_loss))

    def _init_learning(self):
        for i, (images, labels) in enumerate(self.dataloader):
            self.optimizer_a.zero_grad()
            preds = self.model_a(images)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer_a.step()

        return images, labels, loss

    def _continuous_learning(self, fisher_information, init_parameters):
        for i, (images, labels) in enumerate(self.continuous_dataloader):
            self.optimizer_b.zero_grad()
            preds = self.model_b(images)

            ewc_penalty_list = []
            for f, p1, p2 in zip(fisher_information, init_parameters, self.get_params()):
                ewc_penalty_list.append(torch.sum(torch.mul(f, (p1 - p2) ** 2)))

            ewc_penalty = torch.stack(ewc_penalty_list).sum()
            classification_loss = self.criterion(preds, labels)

            loss = classification_loss + ewc_penalty
            loss.backward()
            self.optimizer_b.step()

        return loss, ewc_penalty, classification_loss

    def _calc_fisher_information(self, images, labels):
        self.optimizer_a.zero_grad()
        preds = self.model_a(images)
        preds, _ = preds.max(1)
        preds = preds.sum()
        preds.backward()

        params = []
        for param in self.model_a.parameters():
            params.append(self.lambda_ * param.grad ** 2)

        return params

    def init_params(self):
        params = []
        for param in self.model_a.parameters():
            params.append(param.detach())

        return params

    def get_params(self):
        params = []
        for param in self.model_b.parameters():
            params.append(param)

        return params

    def validate_model_b(self):
        self.model_b.eval()
        accuracy_list = []
        for i, (images, labels) in enumerate(self.dataloader):
            batch_size = images.size(0)
            preds = self.model_b(images)
            _, preds = torch.max(preds, 1)
            _accuracy = 100 * (preds == labels).sum().item() / batch_size
            accuracy_list.append(_accuracy)

        accuracy = torch.tensor(accuracy_list).mean()

        continuous_accuracy_list = []
        for i, (images, labels) in enumerate(self.val_continuous_dataloader):
            batch_size = images.size(0)
            preds = self.model_b(images)
            _, preds = torch.max(preds, 1)
            _continuous_accuracy = 100 * (preds == labels).sum().item() / batch_size
            continuous_accuracy_list.append(_continuous_accuracy)

        continuous_accuracy = torch.tensor(continuous_accuracy_list).mean()

        return accuracy, continuous_accuracy

    def validate_model_a(self):
        self.model_b.eval()
        accuracy_list = []
        for i, (images, labels) in enumerate(self.dataloader):
            batch_size = images.size(0)
            preds = self.model_a(images)
            _, preds = torch.max(preds, 1)
            _accuracy = 100 * (preds == labels).sum().item() / batch_size
            accuracy_list.append(_accuracy)

        accuracy = torch.tensor(accuracy_list).mean()

        continuous_accuracy_list = []
        for i, (images, labels) in enumerate(self.val_continuous_dataloader):
            batch_size = images.size(0)
            preds = self.model_a(images)
            _, preds = torch.max(preds, 1)
            _continuous_accuracy = 100 * (preds == labels).sum().item() / batch_size
            continuous_accuracy_list.append(_continuous_accuracy)

        continuous_accuracy = torch.tensor(continuous_accuracy_list).mean()

        return accuracy, continuous_accuracy
