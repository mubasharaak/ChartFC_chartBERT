import torch
import torch.nn as nn
from torchvision import transforms
import text_encoder, image_encoder, fusion


# main
expt_dir = ""  # directory for results saving
config_location = '/scratch/users/k20116188/prefil/configs/config.py'

train_file = dict()
train_file['ChartFC'] = 'train_barplot_seaborn_diverse_charts.json'

val_files = dict()
val_files['ChartFC'] = {'val': 'valid_barplot_seaborn_diverse_charts.json'}

test_files = dict()
test_files['ChartFC'] = {'test': 'test_barplot_seaborn_diverse_charts.json'}
# test_files['ChartFC'] = {'test': 'labelled_test_sample_barplot_seaborn_imgtext_tesseract.json'}

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

root = 'data'  # This will be overwritten by command line argument
dataset = 'ChartFC'  # Should be defined above in the datastore section

train_filename = train_file[dataset]
val_filenames = val_files[dataset]
test_filenames = test_files[dataset]
train_transform = transform_combo_train[dataset]
test_transform = transform_combo_test[dataset]

lut_location = ''  # When training, LUT for question and answer token to idx is computed from scratch if left empty, or

# optimizer = torch.optim.Adam
optimizer = torch.optim.Adamax

# hyperparameters
test_interval = 1  # In epochs
test_every_epoch_after = 1
max_epochs = 10
batch_size = 8
dropout_classifier = 0.3
lr = 5e-5
lr_decay_step = 2  # Decay every this many epochs
lr_decay_rate = .7
lr_decay_epochs = range(15, 125, lr_decay_step)
lr_warmup_steps = []

# utils
use_ocr = False
ocr_type = "concat"
data_subset = 1  # Random Fraction of data to use for training
data_sampling_seed = 666

# model
criterion = nn.BCEWithLogitsLoss()
txt_token_count = 0
label_count = 1
txt_encoder = None
img_encoder = None
fusion_method = None

# encoders
lstm_embedding_dim = 32
simple_encoder_max_position_embeddings = 512
text_dim = 0
img_dim = 0
pretrained_model = "bert-base-multilingual-cased"

# DenseNet
densenet_config = (6, 12, 24)
densenet_dim = [128, 256, 1024]


# UNITER fusion config
attention_probs_dropout_prob = 0.3
hidden_act = "gelu"
hidden_dropout_prob = 0.3 # todo increase to 0.2
hidden_size = 768
initializer_range = 0.02
intermediate_size = 3072
max_position_embeddings = 512
num_attention_heads = 12
num_hidden_layers = 12
type_vocab_size = 2
vocab_size = 28996

COMPONENTS = {
    "word_embedding": text_encoder.SimpleTextEncoder,
    "lstm": text_encoder.LstmEncoder,
    "bert": text_encoder.BertEncoder,
    "fc": image_encoder.SimpleImageEncoder,
    "alexnet": image_encoder.AlexNetEncoder,
    "resnet": image_encoder.ResNetEncoder,
    "densenet": image_encoder.DenseNetEncoder,
    "vit": image_encoder.ViTEncoder,
    "concat": fusion.ConcatFusion,
    "concat_bigru": fusion.ConcatBiGRUFusion,
    "mult": fusion.MultiplicationFusion,
    "mcb": fusion.MCBFusion,
    "transf": fusion.TransformerFusion,
}

# fusion
fusion_out_dim = 0
fusion_transf_layers = 12
num_classes = 1