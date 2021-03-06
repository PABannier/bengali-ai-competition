import os
import ast
from tqdm import tqdm
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliAIDataset
from metric import macro_recall

import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")

# HYPERPARAMETERS
LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 + l2 + l3) / 3 # Try a weighted loss


def train(dataset, data_loader, model, optimizer):
    model.train()

    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/ data_loader.batch_size)):
        counter += 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        if torch.cuda.is_available():
            image = image.to(DEVICE, type=torch.float)
            grapheme_root = grapheme_root.to(DEVICE, type=torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, type=torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, type=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        final_loss += loss

        o1, o2, o3 = outputs
        t1, t2, t3 = targets
        final_outputs.append(torch.cat((o1,o2,o3), dim=1))
        final_targets.append(torch.stack((t1,t2,t3), dim=1))

    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    print("=================Train=================")
    macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss/counter , macro_recall_score


def evaluate(dataset, data_loader, model):
    model.eval()

    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/ data_loader.batch_size)):
        counter += 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        if torch.cuda.is_available():
            image = image.to(DEVICE, type=torch.float)
            grapheme_root = grapheme_root.to(DEVICE, type=torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, type=torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, type=torch.long)

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        
        loss = loss_fn(outputs, targets)
        final_loss += loss

        o1, o2, o3 = outputs
        t1, t2, t3 = targets
        final_outputs.append(torch.cat((o1,o2,o3), dim=1))
        final_targets.append(torch.stack((t1,t2,t3), dim=1))
        
    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    print("=================Validation=================")
    macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss / counter , macro_recall_score
    

def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)

    model.to(DEVICE)

    train_dataset = BengaliAIDataset(
        folds=TRAINING_FOLDS, 
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        training=True,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )

    valid_dataset = BengaliAIDataset(
        folds=VALIDATION_FOLDS, 
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        training=False,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, 
        batch_size=TEST_BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Pay attention: some schedulers need to step after every batch OR after every epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, 
                                                            factor=0.3, verbose=True)

    # Other ideas: early stopping to prevent overfitting

    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f'../models/{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.h5')

if __name__ == "__main__":
    main()                                              

