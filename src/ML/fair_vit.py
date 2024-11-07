import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import metrics
import math
import sys
import gc
import pickle

PATH_MODEL_BASE = "../../Models/vit-fair-{}-run-{}.pt"
PATH_PREDICTIONS_BASE = "../../Predictions/vit-fair-{}-run-{}-{}.csv"
PATH_PLOT = "../../plots/vit-fair-{}-run-{}.pkl"
PATH_IMGS = "../../UTKFace"
PATH_TRAIN = "../../Base/train.csv"
PATH_VALID = "../../Base/val.csv"
PATH_TEST = "../../Base/test.csv"

FAIR_INSTANCE = "../../FairTrainingSet/utk3-{}.txt"

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SIZE = int(sys.argv[1])
N_RUN = int(sys.argv[2])
print('SIZE:', SIZE, ' N_RUN:', N_RUN)
PATH_MODEL = PATH_MODEL_BASE.format(SIZE, N_RUN)

EPOCHS = 200

class ImageDataset(Dataset):
  def __init__(self, filenames, labels, directory, transform):
    self.filenames = filenames
    self.labels = labels
    self.directory = directory
    self.transform = transform

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    img_name = os.path.join(self.directory, self.filenames[idx])
    image = Image.open(img_name).convert('RGB')
    label = torch.tensor(self.labels[idx], dtype=torch.float32)

    if self.transform:
      image = self.transform(image)

    return image, label

class ViTSmall(nn.Module):
  def __init__(self):
    super(ViTSmall, self).__init__()
    self.base = timm.create_model('vit_small_patch16_224', pretrained=True)
    self.base.head = nn.Linear(in_features=self.base.head.in_features, out_features=1)

  def forward(self, x):
    return self.base(x)

class EarlyStopping:
  def __init__(self, model_path, patience=20, verbose=True):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_loss = float('inf')
    self.model_path = model_path

  def __call__(self, val_loss, model):
    if val_loss + 0.01 < self.best_loss:
      self.best_loss = val_loss
      self.counter = 0
      torch.save(model.state_dict(), self.model_path)
      if self.verbose:
        print(f"Validation loss improved: {val_loss:.6f}. Model saved.")
    else:
      self.counter += 1
      if self.counter >= self.patience:
        if self.verbose:
          print(f"Early stopping triggered. Best validation loss: {self.best_loss:.6f}")
        return True
    return False


def compute_metrics(y_true, y_pred, names, path_csv):
  print('MAE:', metrics.mean_absolute_error(y_true = y_true, y_pred = y_pred))

  sum = 0
  for age in np.unique(y_true):
    lines = np.where(y_true == age)[0]
    sum += metrics.mean_absolute_error(y_true = y_true[lines], y_pred = y_pred[lines])

  print('MAE balanceado:', sum/len(np.unique(y_true)))

  pd.DataFrame({'image': names, 'predictions': np.reshape(y_pred, (-1))}).to_csv(path_csv, index=False)

def process_dataframe(df):
  gender, race = [], []
  for i in df.filename.values:
    meta_data = i.split('_')
    try:
      race.append(int(meta_data[2]))
      gender.append(int(meta_data[1]))
    except:
      race.append(-1)
      gender.append(-1)
      print(meta_data)
  df['gender'] = gender
  df['race'] = race

df_val = pd.read_csv(PATH_VALID)
df_test = pd.read_csv(PATH_TEST)
process_dataframe(df_val)
process_dataframe(df_test)

df_val["filename"][df_val["filename"] == "39_0_1_20170116174525125.jpg.chip.jpg"] = "39_1_20170116174525125.jpg.chip.jpg"


df_train = pd.read_csv(PATH_TRAIN)
process_dataframe(df_train)
df_train = pd.read_csv(FAIR_INSTANCE.format(SIZE))
df_train["filename"][df_train["filename"] == "61_1_1_20170109142408075.jpg.chip.jpg"] = "61_1_20170109142408075.jpg.chip.jpg"
df_train["filename"][df_train["filename"] == "61_1_3_20170109150557335.jpg.chip.jpg"] = "61_1_20170109150557335.jpg.chip.jpg"

transform = transforms.Compose([
  transforms.Resize(IMG_SIZE),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(filenames=df_train['filename'].tolist(), 
                             labels=df_train['age'].tolist(), 
                             directory=PATH_IMGS,
                             transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ImageDataset(filenames=df_val['filename'].tolist(), 
                             labels=df_val['age'].tolist(), 
                             directory=PATH_IMGS,
                             transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = ImageDataset(filenames=df_test['filename'].tolist(), 
                             labels=df_test['age'].tolist(), 
                             directory=PATH_IMGS,
                             transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ViTSmall().to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
criterion = nn.L1Loss() 


early_stopping = EarlyStopping(model_path=PATH_MODEL, patience=20, verbose=True)

history = {'train_loss': [], 'val_loss': []}
for epoch in range(EPOCHS):
  model.train()
  running_loss = 0.0

  for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  model.eval()
  val_loss = 0.0
  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(DEVICE), labels.to(DEVICE)
      outputs = model(images)
      val_loss += criterion(outputs.squeeze(), labels).item()

  scheduler.step(val_loss)

  print(f"Train Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")
  history['train_loss'].append(running_loss / len(train_loader))
  history['val_loss'].append(val_loss / len(val_loader))

  if early_stopping(val_loss / len(val_loader), model):
    break

def predict(model, loader):
  model.eval()
  preds = []
  true_labels = []
  with torch.no_grad():
    for images, labels in loader:
      images = images.to(DEVICE)
      outputs = model(images)
      preds.append(outputs.cpu().numpy())
      true_labels.append(labels.numpy())

  return np.concatenate(preds), np.concatenate(true_labels)

SAVE_PATH = PATH_PREDICTIONS_BASE.format(SIZE, N_RUN, "val")
val_preds, val_true = predict(model, val_loader)
compute_metrics(y_true = val_true, y_pred = val_preds, names = df_val['filename'].tolist(), path_csv = SAVE_PATH)

SAVE_PATH = PATH_PREDICTIONS_BASE.format(SIZE, N_RUN, "test")
test_preds, test_true = predict(model, test_loader)
compute_metrics(y_true = test_true, y_pred = test_preds, names = df_test['filename'].tolist(), path_csv = SAVE_PATH)

with open(PATH_PLOT.format(SIZE, N_RUN), 'wb') as f:
  pickle.dump(history, f)

del model
gc.collect()
