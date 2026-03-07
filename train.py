# Импорт библиотек
import os
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from huita import Net, Dataset

def create_dataset(path, device):
    df = pd.read_csv(path)
    inputs = torch.tensor(df.loc[:, ['Roman shifts', 'Counts', 'Brain region']].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    outputs = torch.tensor(df.loc[:, ['Result']].to_numpy(dtype=np.int64), dtype=torch.long).to(device)

    del df

    labels = torch.nn.functional.one_hot(outputs, num_classes=3).to(device)
    labels = labels.reshape((len(labels), 3))

    del outputs
    dataset = data.TensorDataset(inputs, labels)
    del inputs, labels
    return dataset


if __name__ == '__main__':
    print('Инициализация обучения')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Проверяем доступность GPU и приоритетно используем его
    print(f'Используем: {device}')
    path = 'D:\\Помойка 2.0\\pythonic-shit\\roman_specters\\data\\global_dataframe_normed.csv'

    script_path = os.getcwd()
    os.chdir('..')

    model_name='Linear+Adam'

    os.chdir(script_path)
    model_path = os.path.join(script_path, model_name)
    try:
        os.mkdir(model_path)
    except FileExistsError:
        pass
    print('Выходные директории готовы')

    # Загрузка данных
    dataset = create_dataset(path, device)

    train_size = int(len(dataset) * 0.6)
    test_size = int(len(dataset) * 0.2)
    val_size = int(len(dataset)-train_size-test_size)

    train_dataset, val_dataset, test_dataset = data.random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    del dataset, test_dataset

    batch_size = 2048

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    val_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    print('Данные загружены корректно')

    # Инициализация моделей

    model = Net().to(device)

    loss_function = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Модели загружены корректно')

    # Запуск обучения
    print('Старт обучения')
    epochs = 100

    losses_train = []
    losses_val = []
    for i in range(epochs):
        # Обучение
        print(f'Обучение VGG11 эпоха {i}')
        model.train()

        losses = []
        correct = 0
        total = 0

        train_tqdm11 = tqdm(train_dataloader, leave=True)   # Создаем удобный показатель обучения
        for x, y in train_tqdm11:
            x = x.to(device)
            y = y.to(device)
            y = y.reshape((batch_size, 3))

            out = model(x) # Получаем предсказание
            loss = loss_function(out, y) # Считаем потери

            losses.append(loss.item())
            loss_mean = np.mean(losses).item()

            preds = torch.argmax(out, dim=1)
            corr_outs = torch.argmax(y, dim=1)
            correct += (preds == corr_outs).sum().item()
            total += y.size(0)
            accuracy = correct / total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_tqdm11.set_description(f'loss: {loss:.4f}, loss_mean: {loss_mean:.4f}, accuracy: {accuracy:.4f}')   # Выводим служебную информацию
        losses_train.append(loss_mean)

        # Валидация
        model.eval()
        print('Валидация')

        losses = []
        correct = 0
        total = 0
        val_tqdm11 = tqdm(val_dataloader, leave=True) # Создаем удобный показатель валидации
        for x, y in val_tqdm11:
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                y = y.reshape((batch_size, 3))

                out = model(x) # Получаем предсказание
                loss = loss_function(out, y) # Считаем потери

                losses.append(loss.item())
                loss_mean = sum(losses) / len(losses)

                preds = torch.argmax(out, dim=1)
                corr_outs = torch.argmax(y, dim=1)
                correct += (preds == corr_outs).sum().item()
                total += y.size(0)
                accuracy = correct / total

                val_tqdm11.set_description(f'loss: {loss:.4f}, loss_mean: {loss_mean:.4f}, accuracy: {accuracy:.4f}')

        losses_val.append(loss_mean)

        # Сохранение весов и графиков
        model.to('cpu')   # Переносим модель на CPU для сохранения
        torch.save(model.state_dict(), os.path.join(model_path, f'model[{i}].tar'))   # Сохраняем
        model.to(device)     # Возвращаем модель на устройство (GPU или CPU)
        plt.figure(figsize=(20, 20))
        plt.plot(losses_train, label='train loss mean', color='blue')
        plt.plot(losses_val, label='val loss mean', color='red')
        plt.savefig(os.path.join(model_path, f'model_train.jpg'))    # Сохраняем график
    plt.show()