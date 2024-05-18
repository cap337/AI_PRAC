import torch
import torch.nn as nn
import torch.optim as optim
from our_model import OurModel, DeepNetworkWithModerateLayers, ShallowNetworkWithWideLayers, DeepNetworkWithBatchNormalization, ResidualNetwork, WideNetworkWithDropout
from dataloader import create_dataloader
from data_extractor import extract_features_and_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize the model, dataloaders, and data
model = OurModel()
input, output = extract_features_and_labels("common_player_info.csv", "Player_Totals.csv")
input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.20, random_state=42)
input_train, input_val, output_train, output_val = train_test_split(input_train, output_train, test_size=0.25, random_state=42)

train_loader = create_dataloader(input_train, output_train)
val_loader = create_dataloader(input_val, output_val)
test_loader = create_dataloader(input_test,output_test)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.01)

# Scheduler and Early Stopping settings
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
early_stopping_patience = 20
best_val_loss = float('inf')
patience_counter = 0

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            validation_loss += criterion(outputs, labels).item()
        validation_loss /= len(val_loader)
    
    print(f'Epoch {epoch+1}, Validation Loss: {validation_loss}')
    scheduler.step(validation_loss)  # Adjust learning rate

    # Early stopping logic
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Stopping early due to no improvement")
            break

# Optionally, load the best model back at the end of training if needed
model.load_state_dict(torch.load('best_model.pth'))



def to_string_predictions(tensor):
    print("age:", tensor[0])
    print("experience:", tensor[1])
    print("g:", tensor[2])
    print("fg_percent:", tensor[3])
    print("trb:", tensor[4])
    print("ast:", tensor[5])
    print("pts:", tensor[6])
    print("height:", tensor[7])
    print("weight:", tensor[8])
    print("position:", tensor[9])

def to_string_other(tensor):
    print("reb:",tensor[0])
    print("assist:",tensor[1])
    print("points:",tensor[2])
    
model.eval()
all_predictions = []
all_input  = []
all_correct = []
with torch.no_grad():
    for inputs, correct in test_loader:  # Note that we don't need labels here, hence _
        outputs = model(inputs)
        all_predictions.append((outputs.cpu().numpy()))
        all_input.append(inputs.cpu().numpy())
        all_correct.append(correct.cpu().numpy())

for i in range(len(all_predictions)):
    print("input")
    for j in all_input[i]:
        to_string_predictions(j)
    print("predicted")
    for j in all_predictions[i]:
        to_string_other(j)
    print("actual")
    for j in all_correct[i]:
        to_string_other(j)

print("Predictions (unnormalized):", all_predictions)