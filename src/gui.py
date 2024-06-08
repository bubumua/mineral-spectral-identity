import tkinter as tk
from tkinter import ttk

# 先加载界面框架，内容稍后加载
window = tk.Tk()
# 设置窗口名称
window.title('矿物光谱快速识别')
# 设置窗口大小、位置
window.geometry('1000x700+0+0')
label_loading=ttk.Label(window, text="程序加载中，请稍候...", font=("Microsoft YaHei", 20))
label_loading.pack(pady=20)
window.update()

from tkinter import messagebox
import tkinter.filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import demix as dm
# import numpy as np
# import matplotlib.pyplot as plt


# 删除加载文本
label_loading.destroy()

################################################################
# 1、光谱显示区
# 设置边框
frame1 = tk.Frame(window, bd=5, relief='sunken')
frame1.grid(row=0, column=0, columnspan=5, sticky='ew')
frame1.config(width=1400, height=300) 
# 光谱数据定义
# 光谱库
spectrum_lib=None
# 测试光谱（库）
multi_lib=None
# 未知光谱
unknown_spectrum=None
# 未知光谱名称
unknown_spectrum_name=None
# 已选择的库光谱名
selected_lib_spectra_names=None

# 光谱显示区子区（库光谱）
frame11=tk.Frame(frame1)
frame11.grid(row=0,column=0,sticky='ew')
frame11.config( height=300) 

# 光谱显示区子区（未知光谱）
frame12=tk.Frame(frame1)
frame12.grid(row=0,column=1,sticky='ew')
frame12.config(height=300) 

# 创建一个新的图形
fig = Figure(figsize=(5, 3))
fig1 = Figure(figsize=(5, 3))
fig2 = Figure(figsize=(5, 3))

# 将图形添加到Frame中
canvas_spectrum = FigureCanvasTkAgg(fig, master=frame1)  # A tk.DrawingArea.
canvas_spectrum1 = FigureCanvasTkAgg(fig1, master=frame11) 
canvas_spectrum2 = FigureCanvasTkAgg(fig2, master=frame12) 

def update_test_spectra(selected_spectrum, fitted=None):
    global canvas_spectrum2, fig2
    global unknown_spectrum_name
    # 创建一个新的图形
    fig2 = Figure(figsize=(5, 3))
    ax_spectrum = fig2.add_subplot(111)
    ax_spectrum.plot(selected_spectrum['wavelength'], selected_spectrum['reflectance'], label=unknown_spectrum_name)
    if fitted is not None:
        ax_spectrum.plot(selected_spectrum['wavelength'], fitted)
    # 添加图例
    ax_spectrum.legend() 
    # 将图形添加到Frame中
    canvas_spectrum2.get_tk_widget().pack_forget()
    canvas_spectrum2 = FigureCanvasTkAgg(fig2, master=frame12)  # A tk.DrawingArea.
    canvas_spectrum2.draw()
    canvas_spectrum2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
    
def update_lib_spectra(selected_spectrums:list):
    global canvas_spectrum1, fig1, selected_lib_spectra_names
    selected_lib_spectra_names = selected_spectrums
    # 创建一个新的图形
    fig1 = Figure(figsize=(5, 3))
    ax_spectrum = fig1.add_subplot(111)
    print('draw lib spec')
    for mine_name in selected_lib_spectra_names:
        ax_spectrum.plot(spectrum_lib[mine_name]['wavelength'], spectrum_lib[mine_name]['reflectance'], label=mine_name)
    # 添加图例
    ax_spectrum.legend()
    # 将图形添加到Frame中
    canvas_spectrum1.get_tk_widget().pack_forget()
    canvas_spectrum1 = FigureCanvasTkAgg(fig1, master=frame11)
    canvas_spectrum1.draw()
    canvas_spectrum1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)


################################################################
# 2、文件选择区
frame2 = tk.Frame(window, bd=5, relief='sunken')
frame2.grid(row=1, column=0, rowspan=2, sticky='ns')
frame2.config(width=400, height=300) 
# 创建两个Listbox，一个用于显示库中的文件，一个用于显示未知的文件
listbox_library = tk.Listbox(frame2, selectmode=tk.MULTIPLE)
listbox_library.pack(side='left',fill=tk.BOTH, expand=1)
listbox_unknown = tk.Listbox(frame2, selectmode=tk.SINGLE)
listbox_unknown.pack(side='right',fill=tk.BOTH, expand=1)

# 定义文件被点击后的事件处理器
def on_lib_file_select(event):
    global spectrum_lib
    global canvas_spectrum1
    global selected_lib_spectra_names
    # 获取被选中的文件名
    selected_indices = listbox_library.curselection()
    if len(selected_indices)==0:
        print('selected_indices is empty')
        return
    selected_files = []
    # 遍历索引并获取每个索引对应的文件名
    for index in selected_indices:
        filename = listbox_library.get(index)
        mine_name = filename.split(".")[0]
        selected_files.append(mine_name)
    print(selected_files)
    selected_lib_spectra_names = selected_files
    update_lib_spectra(selected_files)
    
# 定义文件被点击后的事件处理器
def on_unknown_file_select(event):
    global multi_lib
    # global canvas_spectrum1
    global unknown_spectrum
    global unknown_spectrum_name
    # 获取被选中的文件名
    filename = listbox_unknown.get(listbox_unknown.curselection())
    # 打印文件名
    print(filename)
    mine_name = filename.split(".")[0]
    spectrum = multi_lib[mine_name]
    unknown_spectrum = spectrum
    unknown_spectrum_name = mine_name
    update_test_spectra(spectrum)

# 添加事件处理器
listbox_library.bind('<<ListboxSelect>>', on_lib_file_select)
listbox_unknown.bind('<<ListboxSelect>>', on_unknown_file_select)

# 3、饼图显示区
frame3 = tk.Frame(window, bd=5, relief='sunken')
frame3.grid(row=1, column=1, rowspan=2, sticky='')
frame3.config(width=400, height=400)

# 创建一个新的图形并画饼图
fig1 = Figure(figsize=(4, 4))

# 将图形添加到Frame中
canvas_pie = FigureCanvasTkAgg(fig1, master=frame3) 

def update_pie(ec:list, mine_name:str):
    global canvas_pie

    # 分解ec为两个列表
    minenames = [item[0] for item in ec]
    accounts = [item[1] for item in ec]
    # 创建一个新的图形并画新的饼图
    new_fig = Figure(figsize=(4, 4), dpi=100)
    new_ax = new_fig.add_subplot(111)
    new_ax.pie(accounts, labels=minenames, autopct='%1.1f%%', shadow=True, startangle=90)
    new_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    new_ax.set_title(mine_name)
    # 删除旧的画布并添加新的画布
    canvas_pie.get_tk_widget().pack_forget()
    new_canvas = FigureCanvasTkAgg(new_fig, master=frame3)
    new_canvas.draw()
    new_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    # 更新画布引用以便于下一次更新
    canvas_pie = new_canvas
    return


#############################################################
# 4、数据显示区
frame4 = tk.Frame(window, bd=5, relief='sunken')
frame4.grid(row=1, column=2, sticky='nsew')
frame4.config(width=400, height=200)
label_result = tk.Label(frame4, text='Result:',font=('Arial', 18))
label_result.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def update_result(ec:list, evar:float, rss:float):
    global frame4
    # 删除数据显示区的所有子控件
    for widget in frame4.winfo_children():
        widget.destroy()

    # 遍历endmember_coefficients_less并为每个元素创建一个Label
    for minename, account in ec:
        label = tk.Label(frame4, anchor='w',text=f"{minename}: {account*100:.1f}%", font=('Arial', 18))
        label.pack(fill='x')
        print(f"{minename}: {account*100:.0f}")

    # 创建一个Label来显示rss
    label_rss = tk.Label(frame4, anchor='w',text=f"RSS: {rss:.2f}", font=('Arial', 18))
    label_rss.pack(fill='x')

    # 创建一个Label来显示evar
    label_evar = tk.Label(frame4, anchor='w', text=f"拟合可信度: {evar*100:.1f}%", font=('Arial', 18))
    label_evar.pack(fill='x')


#########################################################
# 5、按钮区
frame5 = tk.Frame(window, bd=5, relief='sunken')
frame5.grid(row=2, column=2, sticky='nswe')
frame5.config(width=400, height=100)
# 解混参数
max_mines=2
max_mines_var = tk.StringVar()
max_mines_var.set(str(max_mines))
max_mines=max_mines_var.get()
min_account=0.1
min_account_var = tk.StringVar()
demix_method='LASSO'
demix_method_index=0
lib_path_text_pre='Library Directory: '
test_path_text_pre='Unknown Spectrum Directory: '
lib_path_text='Library Directory: '
test_path_text='Unknown Spectrum Directory: '
isSettingOpened = False
openedSettings = None
# 定义设置按钮的点击事件
def open_settings():
    global max_mines
    global demix_method_index
    global openedSettings
    global isSettingOpened
    # 判断菜单是否已经打开
    if openedSettings:
        openedSettings.lift()
        return

    # 创建一个新的顶级窗口
    settings_window = tk.Toplevel(window)
    settings_window.title('Settings')
    settings_window.attributes('-topmost', 1)
    settings_window.geometry('300x400+1000+100')
    openedSettings = settings_window
    isSettingOpened = True
    
    # 定义选择目录的函数
    def choose_lib_directory():
        global spectrum_lib
        global lib_path_text
        global lib_path_text_pre
        # 打开文件夹选择对话框
        directory = tkinter.filedialog.askdirectory()
        print(directory)
        # update the label to show the selected directory
        lib_path_text = lib_path_text_pre + directory
        label_lib.config(text=lib_path_text)
        # 获取目录下的所有文件
        files = os.listdir(directory)
        # 更新listbox_library
        listbox_library.delete(0, tk.END)  # 删除所有现有的项
        for file in files:
            listbox_library.insert(tk.END, file)  # 添加新的项
        spectrum_lib = dm.create_library(directory)
        spectrum_lib = dm.preprocess_library(spectrum_lib)
        print(spectrum_lib)
        return directory
    
    def choose_unknown_directory():
        global multi_lib
        global test_path_text
        # 打开文件夹选择对话框
        directory = tkinter.filedialog.askdirectory()
        # print(directory)
        # update the label to show the selected directory
        test_path_text = test_path_text_pre + directory
        label_unknown.config(text = test_path_text)
        # 获取目录下的所有文件
        files = os.listdir(directory)
        # 更新Listbox
        listbox_unknown.delete(0, tk.END)
        for file in files:
            listbox_unknown.insert(tk.END, file)
        multi_lib=dm.create_library(directory)
        multi_lib=dm.preprocess_library(multi_lib)
        return directory
    
    # label to show the selected library directory
    label_lib = tk.Label(settings_window, text=lib_path_text, wraplength=300)
    label_lib.pack()
    # 添加选择库按钮
    button_add_lib = tk.Button(settings_window, text='选择库', command=choose_lib_directory)
    button_add_lib.pack()
    # label to show the selected test spectrum directory
    label_unknown = tk.Label(settings_window, text=test_path_text, wraplength=300)
    label_unknown.pack()
    # add button to choose test spectrum directory
    button_add_test = tk.Button(settings_window, text='选择未知光谱', command=choose_unknown_directory)
    button_add_test.pack()
    # label of max_mines
    label_max_mines = tk.Label(settings_window, text='最大矿物数：')
    label_max_mines.pack()
    # Spinbox设置最大矿物数
    spinbox_max_mines = tk.Spinbox(settings_window,textvariable=max_mines_var, from_=1, to=9, increment=1)
    # 设置默认值
    spinbox_max_mines.delete(0, 'end')
    spinbox_max_mines.insert(0, max_mines)
    # 将Spinbox添加到窗口中
    spinbox_max_mines.pack()
    # Spinbox设置最小占比
    label_min_account = tk.Label(settings_window, text='最小占比：(%)')
    label_min_account.pack()
    spinbox_min_account = tk.Spinbox(settings_window, from_=0, to=100, increment=1)
    spinbox_min_account.delete(0, 'end')
    spinbox_min_account.insert(0, int(min_account*100))
    spinbox_min_account.pack()
    # label to show the selected demix method
    label_demix_method = tk.Label(settings_window, text='解混算法：')
    label_demix_method.pack()
    # 添加下拉框，用于选择解混算法
    combobox_options = ttk.Combobox(settings_window)
    # 设置下拉框的选项和默认值
    options = ['LASSO', 'RIDGE', 'LINEAR']
    combobox_options['values'] = options
    combobox_options.current(demix_method_index)
    combobox_options.pack()
    # 设置状态文本
    label_status = tk.Label(settings_window, text='')
    label_status.config(fg='red')
    label_status.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
    
    # 定义关闭窗口的事件处理器
    def on_settings_window_close():
        global max_mines
        global min_account
        global demix_method
        global lib_path_text
        global test_path_text
        global demix_method_index
        global isSettingOpened
        global openedSettings
        try:
            max_mines = int(max_mines_var.get())
            if max_mines<0:
                label_status.config(text='参数错误: 最大矿物数必须为正整数!')
                return
        except ValueError:
            label_status.config(text='参数错误: 最大矿物数必须为正整数!')
            return
        try:
            min_account = float(spinbox_min_account.get())/100
            if min_account<0 or min_account>100:
                label_status.config(text='参数错误: 最小占比大于0且小于100!')
                return
        except ValueError:  
            label_status.config(text='参数错误: 最小占比大于0且小于100!')
            return
        demix_method = combobox_options.get()
        demix_method_index = combobox_options.current()
        lib_path_text=label_lib.cget('text')
        test_path_text=label_unknown.cget('text')
        isSettingOpened = False
        openedSettings = None 
        settings_window.destroy()
        return
    
    settings_window.protocol('WM_DELETE_WINDOW', on_settings_window_close)
    # 添加确认按钮
    button_confirm = tk.Button(settings_window, text='确认', command=on_settings_window_close)
    button_confirm.pack()
    print(max_mines, min_account, demix_method)
    
def demix():
    global canvas_pie, max_mines, min_account, spectrum_lib, multi_lib, unknown_spectrum, demix_method
    
    if spectrum_lib is None:
        # 弹出一个错误框
        messagebox.showerror('错误', '请先选择光谱库')
        return
    if unknown_spectrum is None:
        # 弹出一个错误框
        messagebox.showerror('错误', '请先选择要识别的光谱')
        return
    
    # 根据demix_method的值，选择不同的解混方法
    if demix_method == 'LASSO':
        endmember_coefficients_less, explained_var, rss,fitted = dm.unmix_lasso(spectrum_lib,unknown_spectrum,max_mines=int(max_mines), min_account=min_account)
    elif demix_method == 'RIDGE':
        endmember_coefficients_less, explained_var, rss, fitted = dm.unmix_ridge(spectrum_lib, unknown_spectrum, max_mines=max_mines, min_account=min_account)
    elif demix_method == 'LINEAR':
        endmember_coefficients_less, explained_var, rss, fitted = dm.unmix_linear(spectrum_lib, unknown_spectrum, max_mines=max_mines, min_account=min_account)
    else:
        # 弹出一个错误框
        messagebox.showerror('错误', '解混方法错误')
        return
    # 更新饼图
    update_pie(endmember_coefficients_less, unknown_spectrum_name)
    # 设置Label的文本为返回值
    update_result(endmember_coefficients_less, abs(explained_var), rss)
    # 添加拟合光谱作为对比
    # update_spectra(unknown_spectrum, fitted)
    # print(endmember_coefficients_less, explained_var, rss)

# 开始按钮
start_button = tk.Button(frame5, text='识别', command=demix, font=('Arial',18)) 
start_button.pack(side=tk.LEFT,fill=tk.BOTH, expand=1)
# 设置按钮
settings_button = tk.Button(frame5, text='设置', command=open_settings, font=('Arial',18))
settings_button.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

# 设置行和列的权重以确保它们在窗口大小改变时按比例缩放
window.grid_rowconfigure(0, weight=2)
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)
window.grid_columnconfigure(2, weight=1)

window.mainloop()
