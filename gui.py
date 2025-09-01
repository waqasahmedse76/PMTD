import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import functions
import server_deploy_tool
import os


class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive Model Training and Deployment")

        # Window size & resizing behavior
        self.root.geometry("900x850")
        self.root.minsize(700, 600)

        # Root grid: make key rows stretchy
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)  # Parameters
        self.root.rowconfigure(3, weight=1)  # Prompt
        self.root.rowconfigure(4, weight=1)  # Output

        # State
        self.model_var = tk.StringVar()
        self.columns = []
        self.selected_csv = ""
        self.dep_vars = []
        self.indep_vars = []
        self.param_entries = {}

        # Server Deployment Variables
        self.server_url_var = tk.StringVar()
        self.model_file_path = ""

        # Models/params registry (kept in sync with functions.py)
        # Tuple format: (type, required)
        self.models = {
            "XGBoost": {
                "learning_rate": (float, True),
                "epochs": (int, True),        # used as fallback for n_estimators
                "n_estimators": (int, False),
                "max_depth": (int, False),
                "min_child_weight": (float, False),
                "subsample": (float, False),
                "colsample_bytree": (float, False),
                "gamma": (float, False),
            },
            "LinearRegression": {
                "fit_intercept": (bool, False),
                "copy_X": (bool, False),
                "positive": (bool, False),
                "n_jobs": (int, False),
            },
            "Ridge": {
                "alpha": (float, False),
                "fit_intercept": (bool, False),
                "solver": (str, False),       # 'auto','svd','cholesky','lsqr','sparse_cg','sag','saga'
                "max_iter": (int, False),
                "tol": (float, False),
                "positive": (bool, False),
                "random_state": (int, False),
            },
            "Lasso": {
                "alpha": (float, False),
                "fit_intercept": (bool, False),
                "max_iter": (int, False),
                "tol": (float, False),
                "selection": (str, False),    # 'cyclic' or 'random'
                "positive": (bool, False),
                "random_state": (int, False),
            },
            "ElasticNet": {
                "alpha": (float, False),
                "l1_ratio": (float, False),   # 0..1
                "fit_intercept": (bool, False),
                "max_iter": (int, False),
                "tol": (float, False),
                "selection": (str, False),
                "positive": (bool, False),
                "random_state": (int, False),
            },
        }

        self.build_gui()

        # Sizegrip
        grip = ttk.Sizegrip(self.root)
        grip.grid(row=7, column=0, sticky="se", padx=4, pady=4)

    def build_gui(self):
        # ===== Model Selection =====
        model_frame = ttk.LabelFrame(self.root, text="Model Selection")
        model_frame.grid(row=0, column=0, padx=10, pady=(10, 8), sticky="ew")
        model_frame.columnconfigure(0, weight=0)
        model_frame.columnconfigure(1, weight=1)
        model_frame.columnconfigure(2, weight=0)
        model_frame.rowconfigure(1, weight=1)

        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        model_dropdown = ttk.Combobox(
            model_frame, textvariable=self.model_var, values=list(self.models.keys()), state="readonly"
        )
        model_dropdown.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        model_dropdown.bind("<<ComboboxSelected>>", self.display_model_params)

        load_info_btn = ttk.Button(model_frame, text="Load Model Info", command=self.load_model_info)
        load_info_btn.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        model_info_scroll = ttk.Scrollbar(model_frame, orient="vertical")
        self.model_info_text = tk.Text(model_frame, height=4, wrap="word", yscrollcommand=model_info_scroll.set)
        self.model_info_text.grid(row=1, column=0, columnspan=2, padx=(5, 0), pady=(0, 5), sticky="nsew")
        model_info_scroll.config(command=self.model_info_text.yview)
        model_info_scroll.grid(row=1, column=2, sticky="ns", padx=(0, 5), pady=(0, 5))

        # ===== Model Parameters (expands) =====
        self.param_frame = ttk.LabelFrame(self.root, text="Model Parameters")
        self.param_frame.grid(row=1, column=0, padx=10, pady=8, sticky="nsew")
        self.param_frame.columnconfigure(0, weight=0)
        self.param_frame.columnconfigure(1, weight=1)
        self.param_frame.rowconfigure(999, weight=1)

        # ===== Data =====
        data_frame = ttk.LabelFrame(self.root, text="Data")
        data_frame.grid(row=2, column=0, padx=10, pady=8, sticky="ew")
        data_frame.columnconfigure(0, weight=1)
        df_actions = ttk.Frame(data_frame)
        df_actions.grid(row=0, column=0, pady=5)
        df_actions.grid_anchor("center")
        ttk.Button(df_actions, text="Upload Training Data", command=self.upload_csv).grid(row=0, column=0, padx=5)

        # ===== Prompt (expands) =====
        prompt_frame = ttk.LabelFrame(self.root, text="ChatGPT Prompt")
        prompt_frame.grid(row=3, column=0, padx=10, pady=8, sticky="nsew")
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(0, weight=1)
        prompt_scroll = ttk.Scrollbar(prompt_frame, orient="vertical")
        self.prompt_text = tk.Text(prompt_frame, height=6, yscrollcommand=prompt_scroll.set, wrap="word")
        self.prompt_text.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=5)
        prompt_scroll.config(command=self.prompt_text.yview)
        prompt_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=5)

        # ===== Training Output (expands) =====
        output_frame = ttk.LabelFrame(self.root, text="Training Output")
        output_frame.grid(row=4, column=0, padx=10, pady=8, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical")
        self.output_box = tk.Text(output_frame, height=10, yscrollcommand=output_scroll.set, wrap="word")
        self.output_box.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=5)
        output_scroll.config(command=self.output_box.yview)
        output_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=5)

        # ===== Train Button =====
        actions_frame = ttk.Frame(self.root)
        actions_frame.grid(row=5, column=0, padx=10, pady=8, sticky="ew")
        actions_frame.columnconfigure(0, weight=1)
        actions_frame.grid_anchor("center")
        ttk.Button(actions_frame, text="Train Model", command=self.train_model).grid(row=0, column=0, padx=5, pady=2)

        # ===== Deploy =====
        deploy_frame = ttk.LabelFrame(self.root, text="Deploy Trained Model to Server")
        deploy_frame.grid(row=6, column=0, padx=10, pady=(8, 10), sticky="ew")
        deploy_frame.columnconfigure(0, weight=0)
        deploy_frame.columnconfigure(1, weight=1)

        ttk.Label(deploy_frame, text="Server URL:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        ttk.Entry(deploy_frame, textvariable=self.server_url_var).grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )

        deploy_buttons = ttk.Frame(deploy_frame)
        deploy_buttons.grid(row=1, column=0, columnspan=2, pady=5)
        deploy_buttons.grid_anchor("center")
        ttk.Button(deploy_buttons, text="Load Model File", command=self.load_model_file).grid(row=0, column=0, padx=8)
        ttk.Button(deploy_buttons, text="Deploy Model", command=self.deploy_model_to_server).grid(row=0, column=1, padx=8)

    def load_model_info(self):
        model = self.model_var.get()
        if not model:
            messagebox.showerror("Error", "Please select a model first.")
            return
        try:
            description = functions.get_model_info(model)
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(tk.END, description)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model info: {e}")

    def display_model_params(self, event=None):
        # Clear old params
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()

        model = self.model_var.get()
        if not model:
            return

        # Headings
        header1 = ttk.Label(self.param_frame, text="Parameter", font=("", 9, "bold"))
        header2 = ttk.Label(self.param_frame, text="Value", font=("", 9, "bold"))
        header1.grid(row=0, column=0, padx=5, pady=(5, 2), sticky="w")
        header2.grid(row=0, column=1, padx=5, pady=(5, 2), sticky="w")

        row = 1
        for param, (ptype, required) in self.models[model].items():
            param_type_name = ptype.__name__
            label_text = f"{param} ({param_type_name})"
            if required:
                label_text += " *"
            label = ttk.Label(self.param_frame, text=label_text, foreground="red" if required else "black")
            label.grid(row=row, column=0, padx=5, pady=2, sticky="e")
            entry = ttk.Entry(self.param_frame)
            entry.grid(row=row, column=1, padx=5, pady=2, sticky="ew")
            self.param_entries[param] = (entry, ptype, required)
            row += 1

        self.param_frame.rowconfigure(row, weight=1)

    def upload_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        self.selected_csv = path
        self.columns = functions.analyze_csv(path)
        self.ask_dep_vars()

    def ask_dep_vars(self):
        popup = tk.Toplevel(self.root)
        popup.title("Select Dependent Variable(s)")
        popup.minsize(300, 300)
        popup.grab_set()

        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)
        popup.rowconfigure(1, weight=0)

        container = ttk.Frame(popup)
        container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(container, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vsb.set)

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(inner_id, width=canvas.winfo_width())

        inner.bind("<Configure>", _on_configure)

        var_dict = {}
        for col in self.columns:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(inner, text=col, variable=var)
            chk.pack(anchor="w", padx=4, pady=2)
            var_dict[col] = var

        def on_submit():
            self.dep_vars = [col for col, v in var_dict.items() if v.get()]
            self.indep_vars = [col for col in self.columns if col not in self.dep_vars]
            if not self.dep_vars:
                messagebox.showerror("Error", "Select at least one dependent variable.")
                return
            popup.destroy()

        ttk.Button(popup, text="Submit", command=on_submit).grid(row=1, column=0, sticky="e", padx=10, pady=(0, 10))

    def train_model(self):
        model = self.model_var.get()
        if not model or not self.selected_csv or not self.dep_vars:
            messagebox.showerror("Error", "Please complete all required inputs.")
            return
        params = {}
        for param, (entry, ptype, required) in self.param_entries.items():
            val = entry.get()
            if required and not val:
                messagebox.showerror("Missing Input", f"{param} is required.")
                return
            if val:
                try:
                    if ptype == bool:
                        params[param] = val.lower() in ["true", "1", "yes"]
                    else:
                        params[param] = ptype(val)
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid value for {param}. Expected {ptype.__name__}.")
                    return
        try:
            prompt = functions.generate_prompt(model, params, self.dep_vars, self.indep_vars, self.selected_csv)
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(tk.END, prompt)
            code = functions.get_code_from_chatgpt(prompt)
            if not code:
                raise Exception("No code returned by ChatGPT.")
            output = functions.execute_code(code)
            self.output_box.delete(1.0, tk.END)
            self.output_box.insert(tk.END, output)
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))

    def load_model_file(self):
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.pkl *.joblib")])
        if not path:
            return
        self.model_file_path = path
        messagebox.showinfo("Model Loaded", f"Model file loaded: {os.path.basename(path)}")

    def deploy_model_to_server(self):
        server_url = self.server_url_var.get()
        if not server_url:
            messagebox.showerror("Missing URL", "Please enter the server URL.")
            return
        if not self.model_file_path:
            messagebox.showerror("Missing Model", "Please load a model file first.")
            return
        try:
            result = server_deploy_tool.deploy_model_to_server(self.model_file_path, server_url)
            messagebox.showinfo("Deployment Result", result)
        except Exception as e:
            messagebox.showerror("Deployment Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    # Optional HiDPI tweak:
    # root.tk.call('tk', 'scaling', 1.2)
    app = MLApp(root)
    root.mainloop()
