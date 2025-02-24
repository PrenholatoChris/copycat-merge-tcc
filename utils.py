import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Function to calculate mean and standard deviation
def get_mean_and_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    return mean, std
    
def create_resnet_model(path=None, out_features=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    last_layer = next(reversed(model._modules))
    # if type(getattr(model, last_layer)) == torch.nn.modules.linear.Linear:
    #model.__dict__['_modules'][last_layer]
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)
    if(path is not None):
        model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model
    
def train(model, train_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / total
        accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print("Training complete.")

def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
    
        criterion = nn.CrossEntropyLoss()
    
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            test_loss += loss.item() * inputs.size(0)  # Accumulate the loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        avg_loss = test_loss / total
        accuracy = 100 * correct / total
    
        # print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
        return avg_loss, accuracy

def compare_models_in_dataloader(path_model1, path_model2, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model1, model2 = create_resnet_model(path_model1),create_resnet_model(path_model2)
    model1 = model1.to(device)
    model2 = model2.to(device)
    test_loss, accuracy = evaluate(model1, dataloader, device)
    print(f"{path_model1} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    test_loss, accuracy = evaluate(model2, dataloader, device)
    print(f"{path_model2} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
 