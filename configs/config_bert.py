import torch
from torchvision import transforms

import model_bert as model

train_file = dict()
train_file['ChartFC'] = 'train_barplot_seaborn_imgtext_tesseract.json'

val_files = dict()
val_files['ChartFC'] = {'val': 'valid_barplot_seaborn_imgtext_tesseract.json'}

test_files = dict()
test_files['ChartFC'] = {'test': 'test_barplot_seaborn_imgtext_tesseract.json'}

transform_combo_train = dict()
transform_combo_test = dict()

h_resize, w_resize = 480, 640
transform_combo_train['ChartFC'] = transforms.Compose([
    transforms.Resize((h_resize, w_resize)),
    # transforms.RandomCrop(size=(h_resize, w_resize), padding=8),
    # transforms.RandomRotation(2.8),
    transforms.ToTensor(),
])

transform_combo_test['ChartFC'] = transforms.Compose([
    transforms.Resize((h_resize, w_resize)),
    transforms.ToTensor(),
])

model = model.ChartFCBaseline
root = 'data'  # This will be overwritten by command line argument
dataset = 'ChartFC'  # Should be defined above in the datastore section
data_subset = 1.0  # Random Fraction of data to use for training

train_filename = train_file[dataset]
val_filenames = val_files[dataset]
test_filenames = test_files[dataset]

train_transform = transform_combo_train[dataset]
test_transform = transform_combo_test[dataset]

lut_location = ''  # When training, LUT for question and answer token to idx is computed from scratch if left empty, or

# densenet
densenet_config = (6, 12, 24)
densenet_dim = [128, 256, 1024] # (6, 12, 24)
# densenet_dim = [128, 160, 352]  # (6, 6, 6)
# densenet_dim = [64, 64, 128] # [2, 2, 2]
# densenet_dim = [224, 304, 688] # (12, 12, 12)
# densenet_dim = [96, 112, 240] # densenet (4, 4, 4)


# sizes
text_dim = 768
fusion_out_dim = 1536
num_rf_out = 1536
hidden_size = 1536

# optimizer = torch.optim.Adam
optimizer = torch.optim.Adamax

# hyperparameters
test_interval = 1  # In epochs
test_every_epoch_after = 1
max_epochs = 100
batch_size = 16
dropout_classifier = 0.3
lr = 5e-5
lr_decay_step = 2  # Decay every this many epochs
lr_decay_rate = .7
lr_decay_epochs = range(100, 125, lr_decay_step)
# lr_warmup_steps = [0.5 * lr, 1.0 * lr, 1.0 * lr]
lr_warmup_steps = []

