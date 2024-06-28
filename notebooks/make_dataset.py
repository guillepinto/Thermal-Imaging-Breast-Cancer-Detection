# Data manipulation
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Pytorch essentials for datasets.
import torch.utils
from torch.utils.data import Dataset

# Pytorch essentials
import torch

# Utils
import os
from preprocess import crop_breast

# PyTorch torchvision
from torchvision.transforms import v2

TEST_PATH = "Imagens e Matrizes da Tese de Thiago Alves Elias da Silva/12 Novos Casos de Testes"
TRAIN_PATH = "Imagens e Matrizes da Tese de Thiago Alves Elias da Silva/Desenvolvimento da Metodologia"

def make_dataframe(train_path=TRAIN_PATH, test_path=TEST_PATH):

  patients = []
  labels = []
  segmented_images = []
  matrices = []

  """
  Esta construcción del dataset depende de la estructura del mismo
  """

  # Primero consigo la ruta de imagenes y matrices para cada uno de los pacientes

  for category in os.listdir(test_path):
    # print(category)
    for patient in os.listdir(os.path.join(test_path, category)):
      patient_path = os.path.join(test_path, category, patient)
      # print(patient_path)
      for record in os.listdir(f'{patient_path}/Segmentadas'):
        record_path = os.path.join(f'{patient_path}/Segmentadas', record)
        # print(record_path)
        segmented_images.append(record_path)
        if '-dir.png' in record_path:
          matrix_path = os.path.join(record_path.replace('Segmentadas','Matrizes').replace("-dir.png", ".txt"))
        elif '-esq.png' in record_path:
          matrix_path = os.path.join(record_path.replace('Segmentadas','Matrizes').replace("-esq.png", ".txt"))
        # print(matrix_path)
        if os.path.exists(matrix_path):
          matrices.append(matrix_path)
        else:
          good_part, bad_part = matrix_path[:len(matrix_path)//2], matrix_path[len(matrix_path)//2:]
          bad_part = bad_part.replace('Matrizes', 'Matrizes de Temperatura')
          matrix_path = good_part+bad_part
          matrices.append(matrix_path)
          # print(matrix_path)

        label = patient_path.split('/')[2]
        if label == 'DOENTES':
          label = 1
        else:
          label = 0
        labels.append(label)
        patients.append(record.split('_')[1])

  for category in os.listdir(train_path):
    # print(category)
    for patient in os.listdir(os.path.join(train_path, category)):
      patient_path = os.path.join(train_path, category, patient)
      for record in os.listdir(f'{patient_path}/Segmentadas'):
        record_path = os.path.join(f'{patient_path}/Segmentadas', record)
        # print(record_path)
        segmented_images.append(record_path)
        if '-dir.png' in record_path:
          matrix_path = os.path.join(record_path.replace('Segmentadas','Matrizes').replace("-dir.png", ".txt"))
        elif '-esq.png' in record_path:
          matrix_path = os.path.join(record_path.replace('Segmentadas','Matrizes').replace("-esq.png", ".txt"))
        # print(matrix_path)
        if os.path.exists(matrix_path):
          matrices.append(matrix_path)
        else:
          good_part, bad_part = matrix_path[:len(matrix_path)//2], matrix_path[len(matrix_path)//2:]
          bad_part = bad_part.replace('Matrizes', 'Matrizes de Temperatura')
          matrix_path = good_part+bad_part
          matrices.append(matrix_path)
          # print(matrix_path)

        label = patient_path.split('/')[2]
        if label == 'DOENTES':
          label = 1
        else:
          label = 0
        labels.append(label)
        patients.append(record.split('_')[1])

  # Crear un DataFrame con la información
  data = pd.DataFrame({
      'patient': patients,
      'segmented_image': segmented_images,
      'matrix': matrices,
      'label': labels
  })

  return data

def make_folds(data:pd.DataFrame):
    
    np.random.seed(2024) # seed for reproducibility

    # Extraer los datos para GroupKFold
    X = np.array([i for i in range(len(data))])
    y = data['label'].values
    groups = data['patient'].values

    folds_dict = {}
    groupk_folds = 7
    gkf = GroupKFold(n_splits=groupk_folds)

    # Realizar la validación cruzada por grupos
    for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups), 1):
        train_groups = groups[train_index]

        # Seleccionar aleatoriamente (size) pacientes del conjunto de entrenamiento para validación
        unique_train_groups = np.unique(train_groups)
        random_val_patients = np.random.choice(unique_train_groups, size=8, replace=False)
        
        # Filtrar los índices de los pacientes seleccionados para validación
        val_indices = np.isin(train_groups, random_val_patients)
        
        # Obtener los índices finales para entrenamiento y validación
        final_train_index = train_index[~val_indices]
        val_index = train_index[val_indices]
        
        fold_name = f"fold_{i}"
        folds_dict[fold_name] = {
            'train': final_train_index,
            'val': val_index,
            'test': test_index
        }

    return folds_dict

def make_subdataframes(data:pd.DataFrame, folds:dict):
  # Crear subdataframes
  subdataframes = {}

  for fold_name, indices in folds.items():
      train_df = data.iloc[indices['train']]
      val_df = data.iloc[indices['val']]
      test_df = data.iloc[indices['test']]
      
      subdataframes[fold_name] = {
          'train': train_df,
          'val': val_df,
          'test': test_df
      }
  
  return subdataframes

"""
Constante encontrada al iterar por todas las imágenes segmentadas,
calcular su valor máximo de temperatura y devolver el máximo de todas.
"""

MAX_TEMPERATURE = 36.44

class ThermalDataset(Dataset):
  def __init__(self, dataframe, transform = None, normalize = None, resize = None, crop = None):
    self.dataframe = dataframe
    self.normalize = normalize
    self.transform = transform
    self.resize = resize
    self.crop = crop

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, index):

    """ Carga de la imagen """

    # Entramos a la carpeta y conseguimos la imagen de la lista
    img_path = self.dataframe.iloc[index]['segmented_image']

    # Leemos la imagen segmentada en escala de grises
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = np.array(img)

    """ Carga de la matrix """

    matrix_path = self.dataframe.iloc[index]['matrix']
    # print(matrix_path)

    matrix = np.loadtxt(matrix_path, dtype=np.float32) # https://www.geeksforgeeks.org/import-text-files-into-numpy-arrays/

    """ Consigo la imagen segmentada con los valores de la matrix """

    segmented = np.where(img==0, 0, 1) # int64
    # segmented = img * matrix
    img = (matrix * segmented).astype(np.float32) # float32, shape (480, 640)

    if self.crop:
      img = crop_breast(img) 
      # print(img.shape, img.dtype) # float32, shape (112, 112)

    # Le agrego un canal explícito
    img = np.expand_dims(img, axis=2) # https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html

    if self.normalize:
      img /= MAX_TEMPERATURE

    """ Consiguiendo el label """

    label = self.dataframe.iloc[index]['label']

    """ Convertir las imagenes en tensores y hacer resize """
    if self.transform:
      # Aplicamos las transformaciones a la imagen
      # print(type(img), img.shape)
      img = self.transform(img)

    # self.resize = None if self.resize == 'None' else self.resize

    if self.resize:
      # Todas las imagenes vienen en h: 480, w: 640 (si no se le hizo crop). El objetivo
      # es disminuir el tamaño sin perder la relación de aspecto. https://gist.github.com/tomvon/ae288482869b495201a0
      HEIGHT = self.resize
      r = HEIGHT/img.shape[1] # Calculo la relación de aspecto.
      WIDTH = int(img.shape[2]*r)
      # print(f"Efectivamente, voy a hacer resize a {HEIGHT}x{WIDTH}")
      resize = v2.Resize(size=(HEIGHT, WIDTH), antialias=True)
      img = resize(img)

    return img, label

def get_data(transform, crop=None, resize=None, normalize=False, slice=1, fold:int=None):

    data = make_dataframe()

    # Generate folds
    folds = make_folds(data)

    # Create subdataframes  
    subdataframes = make_subdataframes(data, folds)

    if not fold:
      fold = np.random.choice(range(1, 8))

    fold_name = f'fold_{fold}'
    print(f"FOLD {fold}\n-------------------------------")

    train_dataset = ThermalDataset(subdataframes[fold_name]['train'],
                                    transform=transform, normalize=normalize,
                                    resize=resize, crop=crop)
    val_dataset = ThermalDataset(subdataframes[fold_name]['val'],
                                  transform=v2.ToImage(), normalize=normalize,
                                  resize=resize, crop=crop)
    test_dataset = ThermalDataset(subdataframes[fold_name]['test'],
                                    transform=v2.ToImage(), normalize=normalize,
                                    resize=resize, crop=crop)
    
    # test with less data, it helped me to set up the experiments faster if slice=1
    # then it returns the complete dataset
    train_dataset = torch.utils.data.Subset(train_dataset, 
                                            indices=range(0, len(train_dataset), slice))
    val_dataset = torch.utils.data.Subset(val_dataset, 
                                          indices=range(0, len(val_dataset), slice))
    test_dataset = torch.utils.data.Subset(test_dataset, 
                                            indices=range(0, len(test_dataset), slice))

    return train_dataset, val_dataset, test_dataset

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

# Test

# data = make_dataframe()

# # Generar los folds
# folds = make_folds(data)

# # Crear subdataframes 
# subdataframes = make_subdataframes(data, folds)

# print(subdataframes)

# train, val, test = get_data(transform=v2.ToImage(), resize=300, normalize=False, slice=10, crop=True)

# print(train.__len__(), val.__len__(), test.__len__())

# train_loader = make_loader(train, 4)
# val_loader = make_loader(val, 4)
# test_loader = make_loader(test, 4)

# for images, labels in train_loader:
#     print(images.shape, labels)
#     break

# for images, labels in val_loader:
#     print(images.shape, labels)
#     break

# for images, labels in test_loader:
#     print(images.shape, labels)
#     break