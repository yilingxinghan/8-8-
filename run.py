from model.CNN import CNNModel
import tkinter as tk
import numpy as np

# 加载模型
CNN = CNNModel("./weight/CNNModel.h5")
CNN.summary()

# 创建主窗口
root = tk.Tk()
root.title("8x8 数字识别")
root.geometry("400x360")  # 固定窗口大小
root.resizable(False, False)  # 禁止调整窗口大小
# 设置画布尺寸
canvas_size = 8
cell_size = 40
# 在画布上方创建Label
label = tk.Label(root, text="作者：一零星寒", font=("宋体", 14))
label.pack()

canvas = tk.Canvas(root, width=cell_size * canvas_size, height=cell_size * canvas_size)
canvas.pack()

# 创建8x8的网格
grid = [[None for _ in range(canvas_size)] for _ in range(canvas_size)]
for i in range(canvas_size):
    for j in range(canvas_size):
        grid[i][j] = canvas.create_rectangle(j * cell_size, i * cell_size, 
                                              (j + 1) * cell_size, (i + 1) * cell_size, fill="white")

# 获取当前点阵数组的函数
def get_grid_state():
    grid_state = []
    for i in range(canvas_size):
        row = []
        for j in range(canvas_size):
            current_color = canvas.itemcget(grid[i][j], "fill")
            row.append(1 if current_color == "black" else 0)  # 1代表黑色，0代表白色
        grid_state.append(row)
    return grid_state

# 更新Label的内容
def update_label():
    image_array = np.array(get_grid_state()).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0) 
    result_pred = CNN.predict(image_array)
    label.config(text=f"推理结果: {result_pred}")

# 定义一个函数，用于在点击单元格时更改它的颜色
def change_color(event):
    x = event.x // cell_size
    y = event.y // cell_size
    current_color = canvas.itemcget(grid[y][x], "fill")
    new_color = "black" if current_color == "white" else "white"
    canvas.itemconfig(grid[y][x], fill=new_color)
    update_label()  # 每次点击后更新Label

# 绑定点击事件
canvas.bind("<Button-1>", change_color)

# 运行主循环
root.mainloop()