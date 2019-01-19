from src.trainer import Trainer
from src.model import Model
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

continuous_transform = transforms.Compose([
    transforms.Resize((14, 28)),
    transforms.Pad((0, 7, 0, 7)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

dataset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=2)

val_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=2)

continuous_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=continuous_transform)
continuous_dataloader = torch.utils.data.DataLoader(continuous_dataset,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=2)
val_continuous_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=continuous_transform)
val_continuous_dataloader = torch.utils.data.DataLoader(continuous_dataset,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=2)

trainer = Trainer(model_a=Model(), model_b=Model(), dataloader=dataloader, continuous_dataloader=continuous_dataloader, val_dataloader=val_dataloader, val_continuous_dataloader=val_continuous_dataloader)
trainer.train(1, 100)
