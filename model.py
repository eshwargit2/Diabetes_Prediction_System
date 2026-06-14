#V1 branch

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Global variables
df = None
X = None
y = None
indep_selected = None
classifier = None
scaler = None



# Function to load the dataset ,--------------first page---------------------------------------------
def load_dataset():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Info", f"Dataset Loaded Successfully!\nColumns: {', '.join(df.columns)}")
            
            #show dataset in the console
            preview_button.pack(pady=20)  # Show the preview button after loading the dataset
            next_button.pack(pady=20)  # Enable the Next button after loading the dataset
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    else:
        messagebox.showerror("Error", "Failed to load dataset")

# Function to go to the enter variables page
def go_to_enter_variables_page():
    root.withdraw()
    enter_vars_window = tk.Toplevel(root)
    enter_vars_window.title("Enter Variables")
    enter_vars_window.geometry("800x600")
    enter_vars_window.configure(bg='#F0F8FF')
    
    # Title
    title_label = tk.Label(enter_vars_window, text="Variable Configuration", bg='#F0F8FF', fg='#1e3d58', font=("Helvetica", 18, 'bold'))
    title_label.pack(pady=20)
    
    # Main content frame
    content_frame = tk.Frame(enter_vars_window, bg='#F0F8FF')
    content_frame.pack(pady=10, padx=20, fill='both', expand=True)
    
    # Input frame with dark background
    input_frame = tk.Frame(content_frame, bg="#2C3E50", relief='solid', bd=2)
    input_frame.pack(pady=20, padx=20, fill='x')
    
    # Dependent variable section with proper alignment
    dep_label = tk.Label(input_frame, text="Dependent Variable:", bg='#2C3E50', fg='white', font=("Helvetica", 12, 'bold'))
    dep_label.pack(anchor='w', padx=20, pady=(15, 5))
    
    dep_entry = tk.Entry(input_frame, width=50, bg='#f0f8ff', font=("Arial", 11), bd=2)
    dep_entry.pack(anchor='w', padx=20, pady=(0, 15))

    # Independent variables info section
    info_frame = tk.Frame(content_frame, bg='#F0F8FF')
    info_frame.pack(pady=15, padx=20, fill='x')
    
    info_label = tk.Label(info_frame, text="✓ Independent variables are selected automatically from remaining columns.", bg='#F0F8FF', fg='#1e3d58', font=("Helvetica", 11, 'italic'))
    info_label.pack(anchor='w')

#set variable ---------------------second page--------------------------------------
    def set_variables():
        selected_dep = dep_entry.get().strip()

        global X, y, indep_selected
        try:
            # Validate and set independent and dependent variables
            if selected_dep not in df.columns:
                raise ValueError("Invalid dependent variable name.")
            
            # Automatically select all other columns as independent variables
            indep_selected = [col for col in df.columns if col != selected_dep]
            
            # Assign X and y based on selection
            X = df[indep_selected].values
            y = df[selected_dep].values

            # Handle categorical variables by encoding them
            label_encoder = LabelEncoder()
            for col in indep_selected:
                if df[col].dtype == 'object':  # Assuming categorical columns are of type 'object'
                    df[col] = label_encoder.fit_transform(df[col])

            # Ensure all data is numeric
            X = df[indep_selected].values  # Make sure X is re-assigned after encoding

            messagebox.showinfo("Info", "Variables Set Successfully!")
            enter_vars_window.destroy()
            go_to_algorithm_page()

        except ValueError as e:
            messagebox.showerror("Error", f"Error: {e}")

    # Button frame for proper alignment
    button_frame = tk.Frame(content_frame, bg='#F0F8FF')
    button_frame.pack(pady=30, fill='x')
    
    # Set Variables button
    set_btn = tk.Button(button_frame, text="Set Variables", command=set_variables, bg='#1e3d58', fg='white', font=("Helvetica", 12, 'bold'), relief="raised", bd=3, width=15)
    set_btn.pack(side='left', padx=10)
    
    # Preview Dataset button
    preview_btn = tk.Button(button_frame, text="Preview Dataset", command=preview_dataset, bg='#1e3d58', fg='white', font=("Helvetica", 12, 'bold'), relief="raised", bd=3, width=15)
    preview_btn.pack(side='left', padx=10)

    # Back button
    def go_back_to_root():
        enter_vars_window.destroy()
        root.deiconify()
    
    back_btn = tk.Button(button_frame, text="Back", command=go_back_to_root, bg='#1e3d58', fg='white', font=("Helvetica", 12, 'bold'), relief="raised", bd=3, width=15)
    back_btn.pack(side='left', padx=10)
    


# Function to go to the algorithm selection page,------------------third page -------------------------------------
def go_to_algorithm_page():
    algo_window = tk.Toplevel(root)
    algo_window.title("Select Algorithm")
    algo_window.geometry("800x600")
    algo_window.configure(bg='#F0F8FF')

    def select_algorithm():
        algo_window.destroy()
        go_to_train_model_page()

    tk.Button(algo_window, text="Select Logistic Regression", command=select_algorithm, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)
    
    def go_back_to_root():
        algo_window.destroy()
        root.deiconify()
    
    tk.Button(algo_window, text="Back", command=go_back_to_root, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)

# Function to go to the training model page ,------------------------4th page ---------------------------------------
def go_to_train_model_page():
    train_model_window = tk.Toplevel(root)
    train_model_window.title("Train Model")
    train_model_window.geometry("800x600")
    train_model_window.configure(bg='#F0F8FF')

    def train_model():
        try:
            global scaler, classifier, X_train, X_test, y_train, y_test
            # Split the dataset into training and testing sets (80-20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scaling the data (normalize features)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train the Logistic Regression model
            classifier = LogisticRegression(max_iter=3000, random_state=42)
            classifier.fit(X_train, y_train)

            # Make predictions and calculate evaluation metrics
            y_pred = classifier.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)

            result_text = (
                f"Model Trained Successfully!\n\n"
                f"Accuracy: {score * 100:.2f}%\n"
                f"Confusion Matrix:\n{conf_matrix}\n\n"
                f"Classification Report:\n{class_report}"
            )
            #popup message , model traind result 
            messagebox.showinfo("Model Result", result_text)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while training the model: {e}")

    tk.Button(train_model_window, text="Train Model", command=train_model, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)
    tk.Button(train_model_window, text="Go to Prediction Page", command=go_to_prediction_page, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)
    
    def go_back_to_root():
        train_model_window.destroy()
        root.deiconify()
    
    tk.Button(train_model_window, text="Back", command=go_back_to_root, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4).pack(pady=20)

# Function to go to the prediction page , ----------------------------5th page -------------------------
def go_to_prediction_page():
    prediction_window = tk.Toplevel(root)
    prediction_window.title("Enter Input Values")
    prediction_window.geometry("900x700")
    prediction_window.configure(bg='#F0F8FF')

    # Title
    title_label = tk.Label(prediction_window, text="Make a Prediction", bg='#F0F8FF', fg='#1e3d58', font=("Helvetica", 18, 'bold'))
    title_label.pack(pady=20)

    # Main content frame
    main_frame = tk.Frame(prediction_window, bg='#F0F8FF')
    main_frame.pack(pady=10, padx=20, fill='both', expand=True)

    # Create canvas with scrollbar
    canvas = tk.Canvas(main_frame, bg='#2C3E50', relief='solid', bd=2, highlightthickness=0)
    scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
    
    # Scrollable frame inside canvas
    scrollable_frame = tk.Frame(canvas, bg='#2C3E50')
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack canvas and scrollbar
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    # Enable mousewheel scrolling only on canvas
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind("<MouseWheel>", _on_mousewheel)

    inputs = []

    # Create input fields
    for feature in indep_selected:
        # Create frame for each feature
        feature_frame = tk.Frame(scrollable_frame, bg='#2C3E50')
        feature_frame.pack(anchor='w', pady=10, padx=20, fill='x')
        
        # Label
        label = tk.Label(feature_frame, text=f"{feature}:", bg='#2C3E50', fg='white', font=("Helvetica", 11, 'bold'), width=20, anchor='w')
        label.pack(anchor='w', pady=(0, 5))
        
        # Entry
        entry = tk.Entry(feature_frame, bg='#f0f8ff', font=("Arial", 11), bd=2, width=40)
        entry.pack(anchor='w', padx=(0, 0))
        inputs.append(entry)

    def predict():
        try:
            input_values = [float(entry.get()) for entry in inputs]  # Get input values from the user
            input_array = np.array(input_values).reshape(1, -1)  # Reshape the input values
            input_array = scaler.transform(input_array)  # Scale input data using the fitted scaler
            
            prediction = classifier.predict(input_array)  # Get the prediction from the classifier
            
            # Display a message depending on the prediction result
            if prediction[0] == 1:
                messagebox.showinfo("Prediction", "Predicted: Diabetes")
            else:
                messagebox.showinfo("Prediction", "Predicted: No Diabetes")
    
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {e}")

    # Button frame for proper alignment
    button_frame = tk.Frame(prediction_window, bg='#F0F8FF')
    button_frame.pack(pady=20, fill='x')
    
    # Predict button
    predict_btn = tk.Button(button_frame, text="Predict", command=predict, bg='#1e3d58', fg='white', font=("Helvetica", 12, 'bold'), relief="raised", bd=3, width=15)
    predict_btn.pack(side='left', padx=5)
    
    # Back button
    def go_back_to_root():
        prediction_window.destroy()
        root.deiconify()
    
    back_btn = tk.Button(button_frame, text="Back", command=go_back_to_root, bg='#1e3d58', fg='white', font=("Helvetica", 12, 'bold'), relief="raised", bd=3, width=15)
    back_btn.pack(side='left', padx=10)






# Main , ---------------------------------------------root  Window-------------------------
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("800x600")
root.configure(bg='#F0F8FF')

navbar = tk.Frame(root, bg="#2C3E50", height=60)
navbar.pack(fill="x")   # full width

# Title inside navbar
title = tk.Label(
    navbar,
    text="Diabetes Prediction System",
    bg="#2C3E50",          # background color
    fg="white",         # text color
    font=("Sekuya", 22, "bold")
)

title.pack(pady=15)

tk.Button(root, text="Load Dataset", command=load_dataset, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=5).pack(pady=30)
next_button = tk.Button(root, text="Next", command=go_to_enter_variables_page, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=5)


# preview dataset function----
def preview_dataset():
    global df

    if df is None:
        return

    # new window
    preview = tk.Toplevel(root)
    preview.title("Dataset Preview")
    preview.geometry("1000x500")

    # frame
    frame = tk.Frame(preview)
    frame.pack(fill="both", expand=True)

    # table
    tree = ttk.Treeview(frame)

    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    # headings
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120)

    # insert all rows
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # scrollbars
    scroll_y = tk.Scrollbar(
        frame,
        orient="vertical",
        command=tree.yview
    )

    scroll_x = tk.Scrollbar(
        frame,
        orient="horizontal",
        command=tree.xview
    )

    tree.configure(
        yscrollcommand=scroll_y.set,
        xscrollcommand=scroll_x.set
    )

    # pack
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

preview_button =tk.Button(root, text="Preview Dataset", command=preview_dataset, bg='#1e3d58', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=4)

root.mainloop()