from tkinter import *
from tkinter import ttk

if __name__ == '__main__':
  def compute_ne():
    pass
  
  def create_form():
    ### Matrix_view_frame
    top_level = Toplevel()
    top_level.title("PyGameTheory -- Strategic Form")
    matrix_view_frame = ttk.Frame(top_level, padding="3 3 12 12")
    matrix_view_frame.grid(column=0, row=0, sticky=(N, W, E, S))
    matrix_view_frame.columnconfigure(0, weight=1)
    matrix_view_frame.rowconfigure(0, weight=1)
    num_rows = int(row.get())
    num_columns = int(column.get())
    payoff_matrix = {(m, n): 0 for m in range(num_rows) for n in range(num_columns)}
    entries = {key: ttk.Entry(matrix_view_frame, width=4, textvariable=payoff_matrix[key]) for key in payoff_matrix}
    for key in entries: entries[key].grid(column=key[1]+1, row=key[0]+1, sticky=(W, E))
    ttk.Button(matrix_view_frame, text="Compute NE", command=compute_ne).grid(column=0, row=num_rows+2, sticky=E)
  
  root = Tk()
  root.title("PyGameTheory")
  ### Mainframe
  main_frame = ttk.Frame(root, padding="3 3 12 12")
  main_frame.grid(column=0, row=0, sticky=(N, W, E, S))
  main_frame.columnconfigure(0, weight=1)
  main_frame.rowconfigure(0, weight=1)
  row = StringVar()
  column = StringVar()
  row_entry = ttk.Entry(main_frame, width=4, textvariable=row)
  row_entry.grid(column=1, row=1, sticky=(W, E))
  column_entry = ttk.Entry(main_frame, width=4, textvariable=column)
  column_entry.grid(column=1, row=2, sticky=(W, E))
  ttk.Button(main_frame, text="Create form", command=create_form).grid(column=2, row=3, sticky=W)
  ttk.Label(main_frame, text="rows").grid(column=2, row=1, sticky=W)
  ttk.Label(main_frame, text="columns").grid(column=2, row=2, sticky=W)
  for child in main_frame.winfo_children(): child.grid_configure(padx=5, pady=5)
  row_entry.focus()
  ### Mainloop
  root.bind('<Return>', create_form)
  root.mainloop()
  