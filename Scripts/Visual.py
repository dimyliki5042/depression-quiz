import pandas as pd
import tkinter as tk
from tkinter import ttk
import NN as nn
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


ROOT = tk.Tk()
DATA = None
ACTIVE_FRAME = None
QUESTION_ID = 0
PRGRBAR = tk.IntVar()
ANSWER = tk.DoubleVar()
ANSWERS = []
EVALS_RESULT = None
ACCURACY = 0.0
C_MATRIX = None

def Show_Metrics():
    draw_window = tk.Toplevel(ROOT)
    draw_window.title('График метрик')
    plt_index = 0
    colors = ['blue', 'orange', 'green', 'purple']
    titles = ['Потери перекрестной энтропии', 'Ошибки модели']
    types = ['Тренировочная', 'Валидационная']
    for val_index, val in enumerate(EVALS_RESULT):
        tk.Label(draw_window, text=titles[val_index],
             font=('', 20, 'bold')).pack(expand=1)
        fig = plt.Figure(figsize=(7, 2.5))
        for i, x in enumerate(EVALS_RESULT[val]):
            plt_index += i + 1
            plot1 = fig.add_subplot(1, 2, i + 1)
            plot1.set_title(types[i])
            plot1.plot(
                range(len(EVALS_RESULT[val][x])),
                EVALS_RESULT[val][x], colors[i])
        canvas = FigureCanvasTkAgg(fig, draw_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt_index -= 2

def Show_Confusion_Matrix():
    draw_window = tk.Toplevel(ROOT)
    draw_window.title('Матрица ошибок')
    fig = plt.Figure()
    plot1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(C_MATRIX, annot=True, fmt='d', cmap='Blues', ax=plot1)
    plot1.set_xlabel('Предсказанные значения')
    plot1.set_ylabel('Истинные значения')
    plot1.set_title('Матрица ошибок')
    canvas = FigureCanvasTkAgg(fig, draw_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def Create_Frame(master, title, text, typeAnswr=None):
    global ACTIVE_FRAME, ANSWERS
    ANSWER.set(0)
    frame = tk.Frame(
        master,
        padx= 10,
        pady= 10,
        width=600,
        height=400
    )
    frame.pack(side='top')

    title_txt = tk.Text(frame, height=4, width=70, wrap=tk.WORD)
    title_txt.insert(tk.END, title)
    title_txt.tag_add('format', 1.0, tk.END)
    title_txt.tag_config('format', justify='center', font=('', 20, 'bold'))
    title_txt.grid(row=1, column=1)
    title_txt['state'] = tk.DISABLED

    text_txt = tk.Text(frame, height=5, width=70, wrap=tk.WORD)
    text_txt.insert(tk.END, text)
    text_txt.tag_add('format', 1.0, tk.END)
    text_txt.tag_config('format', font=('', 14))
    text_txt.grid(row=2, column=1)
    text_txt['state'] = tk.DISABLED

    if typeAnswr == 'Scale':
        arr = DATA['variants'][QUESTION_ID]
        answer = tk.Scale(frame, orient=tk.HORIZONTAL, length=400, from_=arr[0], to=arr[1], variable=ANSWER)
    elif typeAnswr == 'Spinbox':
        answer = tk.Spinbox(frame, from_=0.0, increment=0.01, textvariable=ANSWER)
    elif typeAnswr == 'Radiobtn':
        answer = tk.Frame(frame)
        dct = DATA['variants'][QUESTION_ID]
        for key in dct.keys():
            tk.Radiobutton(answer, text=key, value=dct[key], variable=ANSWER).pack(side='left')
    else:
        return frame
    answer.grid(row=3, column=1, pady=20)
    return frame

def Click(event:tk.Event):
    global ACTIVE_FRAME, QUESTION_ID, EVALS_RESULT, ACCURACY, C_MATRIX
    ACTIVE_FRAME.destroy()
    ANSWERS.append(ANSWER.get())
    if QUESTION_ID != len(DATA):
        ACTIVE_FRAME = Create_Frame(ROOT, f'{QUESTION_ID + 1}: {DATA['title'][QUESTION_ID]}',
                                     DATA['text'][QUESTION_ID], DATA['type'][QUESTION_ID])
        QUESTION_ID += 1
    else:
        result, EVALS_RESULT, ACCURACY, C_MATRIX = nn.Get_Answer(ANSWERS[1:])
        ACTIVE_FRAME = Create_Frame(ROOT, 'Финал', result)
        btn = event.widget
        btn['text'] = 'Выход'
        btn.bind('<Button-1>', Close_App)
        swap_menu = tk.Menu(ROOT)
        swap_menu.add_command(label='График метрик', command=Show_Metrics)
        swap_menu.add_command(label='Матрица ошибок', command=Show_Confusion_Matrix)
        ROOT.config(menu=swap_menu)
    PRGRBAR.set(QUESTION_ID)
    ROOT.update_idletasks()
    
def Close_App(event):
    ROOT.destroy()

def Start():
    global ACTIVE_FRAME
    ROOT.title('Quiz')
    ROOT.geometry('600x400')
    ROOT.resizable(False, False)
    ACTIVE_FRAME = Create_Frame(ROOT, 'Добро пожаловать', 'Пройдите тест получите отчет о симптомах депрессии у себя')
    PRGRBAR.set(QUESTION_ID)
    ttk.Progressbar(orient=tk.HORIZONTAL, variable=PRGRBAR, length=200, maximum=13).pack(side=tk.BOTTOM)
    conf_btn = tk.Button(ROOT, text='Далее', font=('', 14), width=50)
    conf_btn.bind('<Button-1>', Click)
    conf_btn.pack(side=tk.BOTTOM, pady=20)
    ROOT.mainloop()

def main():
    global DATA
    DATA = pd.read_json('Sources\\questions.json')
    Start()

if __name__ == '__main__':
    main()