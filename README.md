# آموزش جامع پایتورچ (PyTorch) - از مقدمات تا پیشرفته

## فهرست مطالب
1. [مقدمه و نصب](#مقدمه-و-نصب)
2. [تنسورها در PyTorch](#تنسورها-در-pytorch)
3. [عملیات پایه با تنسورها](#عملیات-پایه-با-تنسورها)
4. [رگرسیون خطی](#رگرسیون-خطی)
5. [شبکه‌های عصبی پایه](#شبکه‌های-عصبی-پایه)
6. [طبقه‌بندی دوتایی](#طبقه‌بندی-دوتایی)
7. [طبقه‌بندی چندکلاسه](#طبقه‌بندی-چندکلاسه)
8. [کار با داده‌ها در PyTorch](#کار-با-داده‌ها-در-pytorch)
9. [بهینه‌سازها و توابع خطا](#بهینه‌سازها-و-توابع-خطا)
10. [تکنیک‌های پیشرفته](#تکنیک‌های-پیشرفته)

## مقدمه و نصب

PyTorch یکی از محبوب‌ترین کتابخانه‌های یادگیری عمیق است که توسط Facebook (Meta) توسعه داده شده است. این کتابخانه به دلیل API ساده و انعطاف‌پذیر، محبوبیت زیادی در بین محققان و توسعه‌دهندگان دارد.

### نصب PyTorch

برای نصب PyTorch، می‌توانید از pip یا conda استفاده کنید:

```bash
# نصب با pip
pip install torch torchvision torchaudio

# نصب با conda
conda install pytorch torchvision torchaudio -c pytorch
```

### بررسی نصب و دسترسی به GPU

```python
import torch

# بررسی نسخه PyTorch
print(f"PyTorch Version: {torch.__version__}")

# بررسی دسترسی به GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## تنسورها در PyTorch

تنسورها ساختارهای داده اصلی در PyTorch هستند. آن‌ها شبیه آرایه‌های NumPy هستند، با این تفاوت که می‌توانند روی GPU اجرا شوند و گرادیان‌ها را به صورت خودکار محاسبه کنند.

### انواع تنسورها

```python
import torch

# 1. تنسور اسکالر (۰ بعدی)
scalar = torch.tensor(7)
print(f"اسکالر: {scalar}")
print(f"ابعاد: {scalar.ndim}")
print(f"مقدار: {scalar.item()}")

# 2. تنسور برداری (۱ بعدی)
vector = torch.tensor([1, 2, 3, 4, 5])
print(f"بردار: {vector}")
print(f"شکل: {vector.shape}")
print(f"ابعاد: {vector.ndim}")

# 3. تنسور ماتریسی (۲ بعدی)
matrix = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
print(f"ماتریس:\n{matrix}")
print(f"شکل: {matrix.shape}")
print(f"ابعاد: {matrix.ndim}")

# 4. تنسور ۳ بعدی
tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]],
                         [[9, 10], [11, 12]]])
print(f"تنسور ۳ بعدی:\n{tensor_3d}")
print(f"شکل: {tensor_3d.shape}")
print(f"ابعاد: {tensor_3d.ndim}")
```

### توابع مهم برای ایجاد تنسورها

```python
# 1. ایجاد تنسور با اعداد تصادفی
random_tensor = torch.rand(size=(3, 4))  # توزیع یکنواخت بین ۰ و ۱
random_normal = torch.randn(size=(3, 4))  # توزیع نرمال با میانگین ۰ و واریانس ۱

# 2. ایجاد تنسور با مقادیر ثابت
zeros = torch.zeros(size=(3, 4))  # تنسور پر از صفر
ones = torch.ones(size=(3, 4))  # تنسور پر از یک
custom_value = torch.full(size=(3, 4), fill_value=3.14)  # تنسور با مقدار دلخواه

# 3. ایجاد تنسور با مقادیر متوالی
arange = torch.arange(start=0, end=10, step=1)  # اعداد متوالی با گام مشخص
linspace = torch.linspace(start=0, end=10, steps=5)  # اعداد با فاصله مساوی

# 4. ایجاد ماتریس همانی
eye = torch.eye(3)  # ماتریس همانی ۳×۳

print(f"تنسور تصادفی:\n{random_tensor}")
print(f"\nتنسور نرمال:\n{random_normal}")
print(f"\nتنسور صفر:\n{zeros}")
print(f"\nتنسور یک:\n{ones}")
print(f"\nتنسور با مقدار ثابت:\n{custom_value}")
print(f"\nاعداد متوالی: {arange}")
print(f"\nاعداد با فاصله مساوی: {linspace}")
print(f"\nماتریس همانی:\n{eye}")
```

### انواع داده در تنسورها

PyTorch از انواع مختلف داده پشتیبانی می‌کند:

```python
# 1. اعداد صحیح
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
long_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)  # یا torch.long

# 2. اعداد اعشاری
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # یا torch.float
double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # یا torch.double

# 3. اعداد اعشاری با دقت نیم
half_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)  # یا torch.half

# 4. اعداد باینری
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)

# تبدیل نوع داده
float_to_int = float_tensor.type(torch.int32)
int_to_float = int_tensor.type(torch.float32)

print(f"تنسور صحیح:\n{int_tensor} - نوع: {int_tensor.dtype}")
print(f"\nتنسور اعشاری:\n{float_tensor} - نوع: {float_tensor.dtype}")
print(f"\nتنسور نیم دقت:\n{half_tensor} - نوع: {half_tensor.dtype}")
print(f"\nتنسور منطقی:\n{bool_tensor} - نوع: {bool_tensor.dtype}")
```

## عملیات پایه با تنسورها

### عملیات ریاضی

```python
# 1. عملیات جمع و تفریق
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"جمع: {a + b}")  # یا torch.add(a, b)
print(f"تفریق: {a - b}")  # یا torch.sub(a, b)

# 2. عملیات ضرب و تقسیم
print(f"ضرب عنصر به عنصر: {a * b}")  # یا torch.mul(a, b)
print(f"تقسیم عنصر به عنصر: {a / b}")  # یا torch.div(a, b)

# 3. ضرب ماتریسی
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])
print(f"ضرب ماتریسی:\n{torch.matmul(matrix1, matrix2)}")  # یا matrix1 @ matrix2

# 4. عملیات آماری
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"میانگین: {tensor.mean()}")
print(f"انحراف معیار: {tensor.std()}")
print(f"مجموع: {tensor.sum()}")
print(f"حداقل: {tensor.min()}")
print(f"حداکثر: {tensor.max()}")
```

### تغییر شکل و اندازه تنسورها

```python
tensor = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8]])

# 1. تغییر شکل با reshape
reshaped = tensor.reshape(4, 2)
print(f"تغییر شکل با reshape:\n{reshaped}")

# 2. تغییر شکل با view
viewed = tensor.view(8, 1)
print(f"تغییر شکل با view:\n{viewed}")

# 3. اضافه کردن بعد جدید
unsqueezed = tensor.unsqueeze(dim=0)  # اضافه کردن بعد در ابتدا
print(f"اضافه کردن بعد:\n{unsqueezed}")

# 4. حذف بعد با اندازه ۱
squeezed = unsqueezed.squeeze()
print(f"حذف بعد:\n{squeezed}")

# 5. جابجایی ابعاد
transposed = tensor.transpose(0, 1)
print(f"جابجایی ابعاد:\n{transposed}")

# 6. تغییر مکان عناصر
permuted = tensor.permute(1, 0)
print(f"تغییر مکان عناصر:\n{permuted}")
```

### عملیات پیشرفته

```python
# 1. ایندکس‌گذاری پیشرفته
tensor = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

print(f"سطر اول: {tensor[0]}")
print(f"ستون دوم: {tensor[:, 1]}")
print(f"عنصر خاص: {tensor[1, 2]}")  # سطر دوم، ستون سوم

# 2. ایندکس‌گذاری با ماسک
mask = tensor > 5
print(f"ماسک:\n{mask}")
print(f"عناصر بزرگتر از ۵:\n{tensor[mask]}")

# 3. برش‌زدن (Slicing)
print(f"برش از سطرها:\n{tensor[1:]}")  # از سطر دوم تا آخر
print(f"برش از ستون‌ها:\n{tensor[:, :2]}")  # دو ستون اول

# 4. الحاق تنسورها
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# الحاق افقی
horizontal_stack = torch.hstack([tensor1, tensor2])
print(f"الحاق افقی:\n{horizontal_stack}")

# الحاق عمودی
vertical_stack = torch.vstack([tensor1, tensor2])
print(f"الحاق عمودی:\n{vertical_stack}")
```

## رگرسیون خطی

### مدل رگرسیون خطی ساده

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. تعریف مدل
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x):
        return self.linear(x)

# 2. ساخت داده‌های مصنوعی
X = torch.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + torch.randn_like(X) * 0.5  # y = 2x + 1 + noise

# 3. تقسیم داده‌ها
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. تعریف مدل و پارامترهای آموزش
model = LinearRegressionModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. آموزش مدل
epochs = 1000
losses = []

for epoch in range(epochs):
    # پیش‌بینی و محاسبه خطا
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    losses.append(loss.item())
    
    # بروزرسانی پارامترها
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 6. رسم نتایج
plt.figure(figsize=(12, 4))

# نمودار داده‌ها و پیش‌بینی
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, label='داده‌های آموزش')
plt.scatter(X_test, y_test, label='داده‌های آزمون')
with torch.no_grad():
    plt.plot(X, model(X), 'r-', label='مدل')
plt.legend()
plt.title('رگرسیون خطی')

# نمودار خطا
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title('نمودار خطا')
plt.xlabel('تکرار')
plt.ylabel('خطا')

plt.tight_layout()
plt.show()
```

## طبقه‌بندی دوتایی

### ایجاد داده‌های دایره‌ای

```python
from sklearn.datasets import make_circles

# ایجاد ۱۰۰۰ نمونه داده
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# تبدیل به تنسور و تقسیم به مجموعه آموزش و آزمون
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### مدل شبکه عصبی ساده برای طبقه‌بندی دوتایی

```python
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x))
```

### محاسبه دقت

```python
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc
```

### آموزش مدل طبقه‌بندی دوتایی

```python
# تعریف مدل، تابع خطا و بهینه‌ساز
model_0 = CircleModelV0().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

# حلقه آموزش
epochs = 100
for epoch in range(epochs):
    # آموزش
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # آزمون
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
```

### افزودن غیرخطی‌سازی با ReLU

```python
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # تابع فعال‌سازی ReLU

    def forward(self, x):
        # اعمال ReLU بین لایه‌ها
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
```

## طبقه‌بندی چندکلاسه

### ایجاد داده‌های چندکلاسه

```python
from sklearn.datasets import make_blobs

# ایجاد داده با ۴ کلاس
NUM_CLASSES = 4
NUM_FEATURES = 2
X_blob, y_blob = make_blobs(n_samples=1000,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    cluster_std=1.5,
    random_state=42
)

# تبدیل به تنسور
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
```

### مدل طبقه‌بندی چندکلاسه

```python
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)
```

### آموزش مدل طبقه‌بندی چندکلاسه

```python
# تعریف مدل، تابع خطا و بهینه‌ساز
model_4 = BlobModel(input_features=NUM_FEATURES, 
                    output_features=NUM_CLASSES, 
                    hidden_units=8).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.1)

# حلقه آموزش
epochs = 100
for epoch in range(epochs):
    # آموزش
    model_4.train()
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # آزمون
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)
```

## ذخیره و بارگذاری مدل‌ها

```python
from pathlib import Path

# ذخیره مدل
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# بارگذاری مدل
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
```

## شبکه‌های عصبی پایه

### ساختار پایه یک شبکه عصبی

```python
import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

# مثال استفاده
model = SimpleNeuralNetwork(input_size=10, hidden_size=20, output_size=2)
```

### توابع فعال‌سازی

```python
# 1. ReLU - تابع یکسوساز
x = torch.randn(10)
relu = nn.ReLU()
print(f"ReLU: {relu(x)}")

# 2. Sigmoid - تابع سیگموید
sigmoid = nn.Sigmoid()
print(f"Sigmoid: {sigmoid(x)}")

# 3. Tanh - تابع تانژانت هایپربولیک
tanh = nn.Tanh()
print(f"Tanh: {tanh(x)}")

# 4. Softmax - تابع نرمال‌سازی نمایی
softmax = nn.Softmax(dim=0)
print(f"Softmax: {softmax(x)}")

# 5. LeakyReLU - تابع یکسوساز نشتی
leaky_relu = nn.LeakyReLU(0.01)
print(f"LeakyReLU: {leaky_relu(x)}")
```

### لایه‌های مختلف شبکه عصبی

```python
# 1. لایه خطی (تمام متصل)
linear = nn.Linear(in_features=10, out_features=5)

# 2. لایه کانولوشن
conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# 3. لایه نرمال‌سازی دسته‌ای
batch_norm = nn.BatchNorm2d(16)

# 4. لایه Dropout
dropout = nn.Dropout(p=0.5)

# 5. لایه MaxPool
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
```

## کار با داده‌ها در PyTorch

### Dataset و DataLoader

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. تعریف یک Dataset سفارشی
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. ساخت داده‌های نمونه
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, (1000,))

# 3. ایجاد Dataset
dataset = CustomDataset(X, y)

# 4. ایجاد DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. استفاده از DataLoader در حلقه آموزش
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"Batch {batch_idx}")
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {target.shape}")
    if batch_idx == 2:  # نمایش فقط ۳ دسته اول
        break
```

### تبدیلات داده

```python
from torchvision import transforms

# 1. تبدیلات پایه برای تصاویر
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 2. تبدیلات داده‌افزایی
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])
```

## بهینه‌سازها و توابع خطا

### انواع بهینه‌سازها

```python
# 1. SGD - گرادیان نزولی تصادفی
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 2. Adam - بهینه‌ساز Adam
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# 3. RMSprop
optimizer_rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# 4. Adagrad
optimizer_adagrad = torch.optim.Adagrad(model.parameters(), lr=0.01)
```

### توابع خطا

```python
# 1. MSE - میانگین مربعات خطا
mse_loss = nn.MSELoss()

# 2. Cross Entropy - آنتروپی متقاطع
cross_entropy = nn.CrossEntropyLoss()

# 3. Binary Cross Entropy - آنتروپی متقاطع دودویی
bce_loss = nn.BCELoss()

# 4. L1 Loss - خطای قدر مطلق
l1_loss = nn.L1Loss()

# مثال استفاده
predictions = torch.randn(3, 5, requires_grad=True)
targets = torch.empty(3, dtype=torch.long).random_(5)

loss = cross_entropy(predictions, targets)
print(f"Loss: {loss.item()}")
```

## تکنیک‌های پیشرفته

### مدیریت حافظه و بهینه‌سازی

```python
# 1. انتقال مدل به GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# 2. خالی کردن حافظه کش
torch.cuda.empty_cache()

# 3. استفاده از amp برای آموزش با دقت مختلط
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### تکنیک‌های آموزش پیشرفته

```python
# 1. برنامه‌ریزی نرخ یادگیری
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 2. Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 3. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### تکنیک‌های ارزیابی مدل

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # محاسبه معیارهای ارزیابی
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

## نکات و ترفندها

### بهترین شیوه‌های کدنویسی

1. همیشه از `model.train()` و `model.eval()` استفاده کنید
2. از `with torch.no_grad()` برای ارزیابی استفاده کنید
3. داده‌ها را به دستگاه مناسب (GPU/CPU) منتقل کنید
4. گرادیان‌ها را قبل از هر مرحله به‌روزرسانی صفر کنید
5. از `DataLoader` برای مدیریت بهتر داده‌ها استفاده کنید

### دیباگ کردن مدل

```python
# 1. چاپ خلاصه مدل
from torchsummary import summary
summary(model, input_size=(3, 224, 224))

# 2. بررسی گرادیان‌ها
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.grad}")

# 3. بررسی مصرف حافظه
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## منابع و مراجع

1. [مستندات رسمی PyTorch](https://pytorch.org/docs/stable/index.html)
2. [آموزش‌های PyTorch](https://pytorch.org/tutorials/)
3. [مخزن مثال‌های PyTorch](https://github.com/pytorch/examples)
4. [انجمن PyTorch](https://discuss.pytorch.org/)

## نتیجه‌گیری

در این آموزش جامع، ما مفاهیم اصلی و پیشرفته PyTorch را بررسی کردیم. از مفاهیم پایه مانند تنسورها تا تکنیک‌های پیشرفته مانند آموزش با دقت مختلط و مدیریت حافظه را پوشش دادیم. این دانش به شما کمک می‌کند تا پروژه‌های یادگیری ماشین و یادگیری عمیق خود را با PyTorch توسعه دهید.

برای تمرین بیشتر، پیشنهاد می‌کنیم:
1. پروژه‌های کوچک را پیاده‌سازی کنید
2. در چالش‌های Kaggle شرکت کنید
3. کدهای منبع باز را مطالعه کنید
4. با جامعه PyTorch در ارتباط باشید

## کار با تصاویر در PyTorch

### شبکه‌های کانولوشنی (CNN)

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # لایه کانولوشن اول
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # لایه کانولوشن دوم
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # لایه کانولوشن سوم
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# مثال استفاده
model = SimpleCNN(num_classes=10)
```

### کار با تصاویر و torchvision

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. تبدیلات تصویر
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 2. بارگذاری مجموعه داده CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=transform)

# 3. ایجاد DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. نمایش تصاویر نمونه
import matplotlib.pyplot as plt

def show_images(dataloader):
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx in range(10):
        axes[idx].imshow(images[idx].permute(1, 2, 0))
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### انتقال یادگیری با مدل‌های پیش‌آموزش‌دیده

```python
import torchvision.models as models

# 1. بارگذاری مدل ResNet پیش‌آموزش‌دیده
resnet = models.resnet18(pretrained=True)

# 2. تنظیم لایه آخر برای مسئله جدید
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)

# 3. فریز کردن وزن‌های لایه‌های قبلی
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc.requires_grad = True

# 4. آموزش مدل
def train_transfer_learning(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
```

## پردازش زبان طبیعی با PyTorch

### کار با داده‌های متنی

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

# 1. تعریف توکنایزر
tokenizer = get_tokenizer('basic_english')

# 2. ساخت واژه‌نامه
def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# 3. تبدیل متن به تنسور
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: int(x)

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)
```

### مدل LSTM برای طبقه‌بندی متن

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        out = self.fc(hidden[-1, :, :])
        return out

# مثال استفاده
vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 256
num_classes = 2

model = TextClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
```

### استفاده از Transformers

```python
from transformers import BertTokenizer, BertModel

# 1. بارگذاری توکنایزر و مدل BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

# 2. تعریف مدل طبقه‌بندی با BERT
class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                          attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

# 3. پردازش متن با BERT
def process_text(texts, tokenizer, max_length=512):
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']
```

### Word Embeddings

```python
# 1. استفاده از Embedding پیش‌آموزش‌دیده
from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=100)
word_embeddings = nn.Embedding.from_pretrained(glove.vectors)

# 2. آموزش Word2Vec
from gensim.models import Word2Vec

sentences = [['این', 'یک', 'جمله', 'است'], 
            ['این', 'جمله', 'دیگری', 'است']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# تبدیل به تنسور PyTorch
weights = torch.FloatTensor(model.wv.vectors)
embedding = nn.Embedding.from_pretrained(weights)
```

## پروژه‌های نمونه

### طبقه‌بندی تصاویر MNIST

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 1. آماده‌سازی داده
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True,
                                         transform=transform,
                                         download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=False,
                                        transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=100,
                                         shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=100,
                                        shuffle=False)

# 2. تعریف مدل
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# 3. آموزش و ارزیابی
model = ConvNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')
```

### تحلیل احساسات با LSTM

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                 weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
```

## بهینه‌سازی عملکرد

### تکنیک‌های بهینه‌سازی پیشرفته

```python
# 1. استفاده از JIT برای کامپایل مدل
from torch import jit

scripted_model = jit.script(model)
traced_model = jit.trace(model, example_input)

# 2. استفاده از QuantizationAware Training
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

qconfig = get_default_qconfig("fbgemm")
prepared_model = prepare_fx(model, qconfig)
quantized_model = convert_fx(prepared_model)

# 3. استفاده از Distributed Training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
model = DDP(model, device_ids=[rank])
```

### پروفایل کردن و دیباگ

```python
# 1. استفاده از پروفایلر PyTorch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(input)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# 2. بررسی مصرف حافظه
def print_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
```

## نکات تکمیلی

### مدیریت خطاها و استثناها

```python
try:
    output = model(input)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("CUDA out of memory. Trying with smaller batch size...")
        torch.cuda.empty_cache()
    else:
        raise e
```

### تست و اعتبارسنجی مدل

```python
import pytest
import torch.testing

def test_model_output():
    model = YourModel()
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    
    # بررسی شکل خروجی
    assert output.shape == (1, num_classes)
    
    # بررسی مقادیر خروجی
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)
```

## منابع بیشتر

1. [مستندات رسمی PyTorch](https://pytorch.org/docs/stable/index.html)
2. [آموزش‌های رسمی PyTorch](https://pytorch.org/tutorials/)
3. [مخزن مثال‌های PyTorch](https://github.com/pytorch/examples)
4. [PyTorch در GitHub](https://github.com/pytorch/pytorch)
5. [انجمن PyTorch](https://discuss.pytorch.org/)
6. [کانال یوتیوب PyTorch](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## نتیجه‌گیری نهایی

در این آموزش جامع، ما مفاهیم اصلی و پیشرفته PyTorch را بررسی کردیم. از مفاهیم پایه مانند تنسورها تا تکنیک‌های پیشرفته مانند شبکه‌های عصبی کانولوشنی، پردازش زبان طبیعی و بهینه‌سازی عملکرد را پوشش دادیم. این دانش به شما کمک می‌کند تا پروژه‌های یادگیری ماشین و یادگیری عمیق خود را با PyTorch توسعه دهید.

برای ادامه یادگیری، پیشنهاد می‌کنیم:
1. پروژه‌های عملی را پیاده‌سازی کنید
2. در چالش‌های Kaggle شرکت کنید
3. کدهای منبع باز را مطالعه کنید
4. با جامعه PyTorch در ارتباط باشید
5. در دوره‌های آنلاین شرکت کنید
6. مقالات و مستندات جدید را دنبال کنید