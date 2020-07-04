## Python工具：可視化loss和命中率 ( A python tool for visualize loss and accuracy while training )

### 1. 效果圖（How this look?）
<img src="./img/demo.gif" height=530px width=508>

### 2. 如何使用？（How to use?）
#### Step1. 
創造一個canvas object,可以自己調整參數來設定想要的port, 畫面大小等等...   
（Create a canvas object. you can manually setting web server port, canvas size, border size...）

```
from utils.web_render import WebRenderer

canvas = WebRenderer(port=12345,
                     batch_size=batch_size,
                     sample_nums=len(trainloader.dataset), 
                     update_per_batches=update_per_batches, 
                     total_epoches=total_epoches, 
                     mode='auto', 
                     blank_size=70, 
                     epoch_pixel=30, 
                     max_vis_loss=10,
                     canvas_h=500,
                     x_ruler=5,
                     y_ruler=2)

canvas_t = threading.Thread(target=canvas.start)
canvas_t.deamon = True
canvas_t.start()
atexit.register(program_exit)

```

中文參數列表 :
| input參數   |      說明      |
|:----------:|:-------------|
| port   |  要開啟的port (網址127.0.0.1:port) |
| sample_nums   |  總共多少訓練資料 |
| update_per_batches   |  在訓練時多少batch要顯示一次 |
|total_epoches| 總共要訓練多少epoches|
|blank_size|畫布邊緣留多少pixel(boarder size)|
|epoch_pixel|每個epoch(x軸的一個刻度) 需要用多少pixel顯示|
|max_vis_loss|最大顯示的loss值|
|canvas_h|畫布的高要多少pixel (顯示數值部分) |
|x_ruler|在一個epoch裡面要畫多少線(灰線)|
|y_ruler|在y軸一個刻度裡面的灰線數量|

English table:
| inputs |      discription      |
|:----------:|:-------------|
| port   | web server's port (127.0.0.1:port)|
| sample_nums   | the numbder of training data |
| update_per_batches   |  how often you update loss and accuracy data while training? (how many batches?)|
|total_epoches| max epoches|
|blank_size| the border size|
|epoch_pixel| how many pixels in one epoch? |
|max_vis_loss| visualize max loss vlaue |
|canvas_h|canvas height (pixels) |
|x_ruler|how many lines you want to display in one epoch (vertical line number)|
|y_ruler|how many lines you want to display in y-axis (horizontal line number)|


#### Step2.
將資料傳入 canvas 裡    
passing acc and loss data (data type=list) to canvas object.


```
if batch%update_per_batches == 0:
    # for record
    _, predicted = torch.max(outs.data, 1)
    correct = (predicted == labels).sum().item()
    accs["train"].append(100*correct/batch_size)
    losses["train"].append(loss.data.item())
    # for visualize
    canvas.updating(accs=accs["train"], 
                    losses=losses["train"], 
                    show_this=True, mode='train')
```

中文參數列表 :
| input參數   |      說明      |
|:----------:|:-------------|
| accs   |  將紀錄命中率的list傳入 |
| losses   |  將紀錄loss值的list傳入 |
| show_this   |  這次的數值要不要顯示 |
| mode   |  訓練還是測試 |

English table:
| inputs |      discription      |
|:----------:|:-------------|
| accs   |  accs list |
| losses   |  losses list |
| show_this   |  want to show value or not (this batch) |
| mode   |  train or test |

#### Step3.
 demo_train.py 是參考[pytorch training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 寫成的訓練範例(cifar10).
 可以執行：
 
 ```
  python3 demo_train.py
 ```
然後打開網頁並輸入127.0.0.1:12345,可以看到訓練的loss和accuracy.


- - - -
### 目前測試環境：
 &#9745; Windows10  
 &#9745; Mac  
 &#9744; Linux
- - - -
Author : NTNU AIoT Lab.  
Email: ntnuchou141253@gmail.com