import cv2
import glob
import torch.utils.data as data
import numpy as np
import torch

def get_skinlesion(data):
    print('-'*30)
    print('Loading images...')
    print('-'*30)

    train_image_list = []
    train_label_list = []
    val_image_list = []
    val_label_list = []
    test_image_list = []
    test_label_list = []

    for filename in data["traindata"]:
        img = cv2.imread(filename)
        train_image_list.append(img)
    for filename in data["trainlabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        train_label_list.append(img)
    for filename in data["valdata"]:
        img = cv2.imread(filename)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        val_image_list.append(img)
    for filename in data["vallabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        val_label_list.append(img)
    for filename in data["testdata"]:
        img = cv2.imread(filename)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        test_image_list.append(img)
    for filename in data["testlabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        test_label_list.append(img)


    return train_image_list, train_label_list, val_image_list, val_label_list, test_image_list, test_label_list

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp, target):
        out1, _ = self.transform(inp, target)
        # out2, _ = self.transform(inp, target)
        return out1, out1


class TransformRot:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp, target):
        out1, _ = self.transform(inp, target)
        # out2, _ = self.transform(inp, target)
        return out1, out1

def get_skinlesion_dataset(root, transform_train=None, transform_val=None):
    # Directly load images from respective folders
    path_train_labeled = glob.glob(root + 'test/rgb/*.png')
    path_train_unlabeled = glob.glob(root + 'train/rgb/*.png')
    path_valid_data = glob.glob(root + 'val/rgb/*.png')
    path_test_data = glob.glob(root + 'val/rgb/*.png')
    
    # path_train_labeled = glob.glob(root + 'rgb/*.png')
    # path_train_unlabeled = glob.glob(root + 'rgb/*.png')
    # path_valid_data = glob.glob(root + 'rgb/*.png')
    # path_test_data = glob.glob(root + 'rgb/*.png')

    # Sort for consistency
    path_train_labeled.sort()
    path_train_unlabeled.sort()
    path_valid_data.sort()
    path_test_data.sort()

    # Corresponding label paths (assuming same filename but different folder)
    path_train_label = [item.replace("train_labeled", "train_labeled_labels").replace("rgb", "gt") for item in path_train_labeled]
    path_valid_label = [item.replace("valid", "valid_labels").replace("rgb", "gt") for item in path_valid_data]
    path_test_label = [item.replace("test", "test_labels").replace("rgb", "gt") for item in path_test_data]

    # Load data using helper function
    data = {
        "train_labeled": path_train_labeled,
        "train_labeled_labels": path_train_label,
        "train_unlabeled": path_train_unlabeled,
        "val": path_valid_data,
        "val_labels": path_valid_label,
        "test": path_test_data,
        "test_labels": path_test_label
    }

    train_labeled_dataset = skinlesion_labeled(data["train_labeled"], data["train_labeled_labels"], transform=transform_train)
    train_unlabeled_dataset = skinlesion_unlabeled(data["train_unlabeled"], label=[None]*len(data["train_unlabeled"]), transform=TransformTwice(transform_train))
    val_dataset = skinlesion_labeled(data["val"], data["val_labels"], transform=transform_val)
    test_dataset = skinlesion_labeled(data["test"], data["test_labels"], transform=transform_val)

    print(f"#Labeled: {len(data['train_labeled'])} #Unlabeled: {len(data['train_unlabeled'])} #Val: {len(data['val'])} #Test: {len(data['test'])}")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

# def get_skinlesion_dataset(root, num_labels, transform_train=None, transform_val=None, transform_forsemi=None):
#     path_train_data = glob.glob(root + 'myTraining_Data248/*.jpg')
#     path_valid_data = glob.glob(root + 'myValid_Data248/*.jpg')
#     path_test_data = glob.glob(root + 'myTest_Data248/*.jpg')

#     #  fix load files seq
#     path_train_data.sort()
#     path_valid_data.sort()
#     path_test_data.sort()

#     ##  index of labeled data
#     # index = list(range(0,len(path_train_data)))
#     # np.random.shuffle(index)
#     # train_labeled_idxs = index[:num_labels]
#     # train_unlabeled_idxs = index[num_labels:]

#     #  index of fixed labeled data
#     if num_labels < 2000:
#         a = np.loadtxt("data_id/skin_id_"+str(num_labels)+".txt", dtype='str')
#         a = [root + "myTraining_Data248/" + item for item in a]
#         train_labeled_idxs = [path_train_data.index(item) for item in a]
#         train_unlabeled_idxs = list(set(list(range(len(path_train_data)))) - set(train_labeled_idxs))
#         #train_unlabeled_idxs.extend(train_labeled_idxs)
#     else:
#         train_labeled_idxs = [path_train_data.index(item) for item in path_train_data]
#         train_unlabeled_idxs = []

#     # label seq
#     path_train_label = ['/'.join(item.replace("myTraining_Data248", "myTraining_Label248").split("/")[:-1]) +"/"+
#                         item.split("/")[-1][:-4]+"_segmentation.png" for item
#                         in path_train_data]
#     path_valid_label = ['/'.join(item.replace("myValid_Data248", "myValid_Label248").split("/")[:-1]) +"/"+
#                         item.split("/")[-1][:-4]+"_segmentation.png" for item
#                         in path_valid_data]
#     path_test_label = ['/'.join(item.replace("myTest_Data248", "myTest_Label248").split("/")[:-1]) +"/"+
#                         item.split("/")[-1][:-4]+"_segmentation.png" for item
#                         in path_test_data]

#     data = {"traindata": path_train_data,
#             "trainlabel": path_train_label,
#             "valdata": path_valid_data,
#             "vallabel": path_valid_label,
#             "testdata": path_test_data,
#             "testlabel": path_test_label}

#     # load data
#     train_data, train_label, val_data, val_label, test_data, test_label = get_skinlesion(data)

#     val_name = path_valid_data
#     test_name= path_test_data
#     train_name = path_train_data


#     train_labeled_dataset = skinlesion_labeled(train_data, train_label,name=train_name,indexs=train_labeled_idxs,
#                                                transform=transform_train)
#     train_unlabeled_dataset = skinlesion_unlabeled(train_data, train_label, indexs=train_unlabeled_idxs,
#                                                    transform=TransformTwice(transform_train))
#     val_dataset = skinlesion_labeled(val_data, val_label, name=val_name,  indexs=None, transform=transform_val)
#     test_dataset = skinlesion_labeled(test_data, test_label, name=test_name, indexs=None, transform=transform_val)

#     print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_data)}")

#     return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """

    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class skinlesion_labeled(data.Dataset):

    def __init__(self, data, label, name=None, indexs=None,
                 transform=None):

        self.data = data
        self.targets = label
        self.transform = transform
        self.name = name


        if indexs is not None:
            self.data = [self.data[item] for item in indexs]
            self.targets = [self.targets[item] for item in indexs]


    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]
    #     print(img, target)

    #     img = img / 255.
    #     target = target / 255.

    #     target[target >= 0.5] = 1
    #     target[target < 0.5] = 0

    #     if self.transform is not None:
    #         img, target = self.transform(img, target)

    #     if self.name is not None:
    #         return img, target, self.name[index]
    #     else:
    #         return img, target

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # print(f"Index: {index}")
        # print(f"Type of img before processing: {type(img)}")  
        # print(f"Type of target before processing: {type(target)}")  

        # If img is a file path, load it
        if isinstance(img, str):
            from PIL import Image
            import numpy as np
            img = Image.open(img).convert("RGB")  
            img = img.resize((256, 256), Image.BILINEAR) 
            img = np.array(img, dtype=np.float32) / 255.  
            img = img.transpose((2, 0, 1))

        # if isinstance(target, str):
        #     target = Image.open(target)  
        #     target = target.resize((256, 256), Image.NEAREST)
        #     target = np.array(target, dtype=np.uint8)            

        # Ensure target is in the correct class range
        target = Image.open(target).convert("L")
        target = target.resize((256, 256), Image.NEAREST)
        target = np.array(target, dtype=np.uint8)    
        # print(f"Unique: {np.unique(target)}")
        # target[target == 2] = 1
        # target[target == 255] = 2

        # print(f"Unique values in target after conversion1: {np.unique(target)}")

        if not np.all(np.isin(target, [0, 1, 2])):  
            raise ValueError(f"Target contains unexpected values: {np.unique(target)}")

        # Convert to torch tensor
        target = torch.tensor(target, dtype=torch.long)

        # target[target >= 0.5] = 1
        # target[target < 0.5] = 0

        # print(f"Unique values in target after thresholding: {np.unique(target)}")
        # Ensure `self.name` is not None and return a default name if missing
        name = self.name[index] if self.name is not None else f"sample_{index}"

        # if self.transform is not None:
        #     print(f"Applying transformation at index {index}...")
        #     img, target = self.transform(img, target)
        #     print(f"Unique values in target after transformation: {np.unique(target)}")


        if self.name is not None:
            return img, target, self.name[index]
        else:
            return img, target, name




    def __len__(self):
        return len(self.data)



class skinlesion_unlabeled(data.Dataset):

    def __init__(self, data, label, indexs=None,
                 transform=None):

        self.data = data
        self.targets = [-1*np.ones_like(label[item]) for item in range(0, len(label))]

        self.transform = transform

        if indexs is not None:
            self.data = [self.data[item] for item in indexs]
            self.targets = [self.targets[item] for item in indexs]
        # self.data = transpose(normalise(self.data))


    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]

    #     img = img / 255.
    #     target = target / 255.

    #     # target[target >= 0.5] = 1
    #     # target[target < 0.5] = 0

    #     if self.transform is not None:
    #         img, target = self.transform(img, target)

    #     return img, target
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if isinstance(img, str):
            from PIL import Image
            import numpy as np
            img = Image.open(img).convert("RGB")  
            img = img.resize((256, 256), Image.BILINEAR) 
            
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.transpose((2, 0, 1))
            # print(f"Image shape before transform: {img.shape}")

        # if isinstance(target, str):
        #     target = Image.open(target)  
        #     target = target.resize((256, 256), Image.NEAREST)
        #     target = np.array(target, dtype=np.uint8)   
        
        # Ensure target is in the correct class range
        if isinstance(target, str):
            target = Image.open(target).convert("L")
            target = target.resize((256, 256), Image.NEAREST)
            target = np.array(target, dtype=np.uint8)    

        # target[target >= 0.5] = 1
        # target[target < 0.5] = 0

        # if self.transform is not None:
        #     img, target = self.transform(img, target)
        # print(f"Unique values in target after conversion: {np.unique(target)}")

        # if not np.all(np.isin(target, [0, 1, 2])):  
        #     raise ValueError(f"Target contains unexpected values: {np.unique(target)}")

        # Convert to torch tensor
        target = torch.tensor(target, dtype=torch.long)
        # print(f"Unique values in target after conversion: {np.unique(target)}")

        return img, target

    def __len__(self):
        return len(self.data)
    

