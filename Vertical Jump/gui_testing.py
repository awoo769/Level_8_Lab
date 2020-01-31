from tkinter import *

master = Tk()

w = Canvas(master, width=200, height=100)
w.pack()

w.create_line(100, 0, 100, 100)
w.create_line(0, 100, 200, 0, fill="red", dash=5)

w.create_rectangle(50, 25, 150, 75, fill="blue")

mainloop()