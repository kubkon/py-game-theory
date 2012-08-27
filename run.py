from tkinter import *
from tkinter import ttk

def create_form(*args):
  pass

if __name__ == '__main__':
  root = Tk()
  root.title("PyGameTheory")
  
  mainframe = ttk.Frame(root, padding="3 3 12 12")
  mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
  mainframe.columnconfigure(0, weight=1)
  mainframe.rowconfigure(0, weight=1)
  
  row = StringVar()
  column = StringVar()
  
  row_entry = ttk.Entry(mainframe, width=4, textvariable=row)
  row_entry.grid(column=1, row=1, sticky=(W, E))
  column_entry = ttk.Entry(mainframe, width=4, textvariable=column)
  column_entry.grid(column=1, row=2, sticky=(W, E))
  
  ttk.Button(mainframe, text="Create form", command=create_form).grid(column=2, row=3, sticky=W)
  
  ttk.Label(mainframe, text="rows").grid(column=2, row=1, sticky=W)
  ttk.Label(mainframe, text="columns").grid(column=2, row=2, sticky=W)
  
  for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)
  
  row_entry.focus()
  root.bind('<Return>', create_form)
  
  root.mainloop()
  