from tkinter import * #Please note small case lettering for tkinter
master = Tk()
e = Entry(master)
master.title="calculator"
# master.configure(backgroud="pink")
Label(master, text="Enter Number",width=10).grid(row=0)
e1 = Entry(master)
e1.grid(row=0, column=1 , columnspan=4)

e.focus_set()
def calback():
  print (e.get())
b = Button(master, text="1", width=10)
b.grid(row=1, column=0)
b = Button(master, text="2", width=10)
b.grid(row=1, column=1)
b = Button(master, text="3", width=10)
b.grid(row=1, column=2)
b = Button(master, text="+", width=10)
b.grid(row=1, column=3)
b = Button(master, text="4", width=10)
b.grid(row=2, column=0)
b = Button(master, text="5", width=10)
b.grid(row=2, column=1)
b = Button(master, text="6", width=10)
b.grid(row=2, column=2)
b = Button(master, text="-", width=10)
b.grid(row=2, column=3)
b = Button(master, text="7", width=10)
b.grid(row=3, column=0)
b = Button(master, text="8", width=10)
b.grid(row=3, column=1)
b = Button(master, text="9", width=10)
b.grid(row=3, column=2)
b = Button(master, text="*", width=10)
b.grid(row=3, column=3)
b = Button(master, text=".", width=10)
b.grid(row=4, column=0)
b = Button(master, text="0", width=10)
b.grid(row=4, column=1)
b = Button(master, text="=", width=10)
b.grid(row=4, column=2)
b = Button(master, text="/", width=10)
b.grid(row=4, column=3)
master.mainloop()