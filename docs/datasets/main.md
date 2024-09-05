# Dataset Description

## Image Classification

For most datasets, we use 32x32 image size, with 3 channels. Standard image transformations are applied during both the training and testing phases.  Within each domain, samples are independently and identically distributed (IID) among users belonging to that particular domain. Specifically, users possess 32 training samples for DomainNet and Camelyon17, and 256 samples in the case of Digit-Five.

### DomainNet
This dataset contains representations of common objects across six distinct domains _sketch, real image, quickdraw, painting, infograph, clipart_. From the available 345 classes, we arbitrarily select ten _suitcase, teapot, pillow, streetlight, table, bathtub, wine glass, vase, umbrella, bench_ for the classification task given their substantial sample sizes within each domain. Unless explicitly stated, our experiments are based on the _real, sketch_ and _clipart_ domains.

### Camelyon17
This dataset consists of high-resolution histopathology images of lymph node sections. We use the binary classification task of distinguishing between normal and tumor slides. The dataset is divided into 270 training and 130 testing slides. We treat each hospital as a distinct domain. We further partition samples from each hospital to create a dedicated test set for each domain.

### Digit-Five
This dataset comprises images of handwritten digits from five distinct domains _MNIST, SVHN, USPS, SYN, and MNIST-M_. We use the binary classification task of distinguishing between digits 0 and 1. Each domain contains 256 training samples and 256 testing samples. Common CNN architectures are designed to handle RGB images hence we convert the grayscale images to RGB format by replicating the single channel three times.

### CIFAR-10
This dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. This dataset is commonly used for benchmarking image classification models.

### CIFAR-100
This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

### Medical MNIST
This dataset consists of 60,000 28x28 grayscale images of 10 classes (6,000 images per class). The images show different medical conditions, such as pneumonia, emphysema, and cardiomegaly. This dataset is used for lightweight medical image classification tasks. The dataset supports several different biomedical imaging modalities, including X-ray, CT, Histopathology, etc.


## Object Detection

### Pascal-VOC
The PASCAL Visual Object Classes (VOC) dataset is a well-known dataset for object detection. It consists of images annotated with bounding boxes around objects of interest. The dataset contains 20 classes, including animals, vehicles, and household items. The dataset is divided into training and validation sets, with a separate test set for evaluation.

## Text Classification

### AGNews
The AGNews dataset is a widely used benchmark for text classification tasks. It consists of news articles categorized into four distinct classes: World, Sports, Business, and Sci/Tech. Each article is labeled with its corresponding category, making it suitable for training and evaluating text classification models. The dataset is divided into training and test sets, allowing for consistent evaluation of model performance. AGNews is commonly used to benchmark the accuracy and effectiveness of natural language processing algorithms, particularly in the context of news categorization.