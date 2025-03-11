from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_train = transforms.Compose(
    [
        transforms.Resize((224,224)),
#         transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,  # Randomly adjust brightness between -0.1 and +0.1
            contrast=0.1,    # Randomly adjust contrast between -0.1 and +0.1
            saturation=0.1,  # Randomly adjust saturation between -0.1 and +0.1
            hue=0.1         # Randomly adjust hue between -0.1 and +0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)
trnsfrms_valid = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)

class RoiDataset(Dataset):
    def __init__(self, img_lst, transform
                 ):
        super().__init__()
        self.transform = transform
        self.images_lst = img_lst

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image