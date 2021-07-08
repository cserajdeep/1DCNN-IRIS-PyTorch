
# 1D-CNN 
 ```ruby
 class OneDCNN(nn.Module):
  def __init__(self, num_classes, in_channels, out_channels, kernel_size=3, stride=1):
    super(OneDCNN,self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    
    self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
    self.relu = nn.ReLU(inplace=True)
    self.fc1 = nn.Linear(out_channels,50)
    self.fc2 = nn.Linear(50,num_classes)
 ```

# Area Under Curve 
<img width="480" alt="1dcnniris_auc" src="https://user-images.githubusercontent.com/18000553/124859968-78549c00-dfce-11eb-893f-bb5c1c500abb.png">

# 1DCNN IRIS PyTorch
