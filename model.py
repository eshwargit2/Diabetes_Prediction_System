#main branch 
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
    enter_vars_window.configure(bg='#F5F7FA')
    
    # Header frame
    header_frame = tk.Frame(enter_vars_window, bg='#1e3d58', height=70)
    header_frame.pack(fill='x')
    
    # Title
    title_label = tk.Label(header_frame, text="Variable Configuration", bg='#1e3d58', fg='#FFFFFF', font=("Helvetica", 22, 'bold'))
    title_label.pack(pady=15)
    
    # Main content frame
    content_frame = tk.Frame(enter_vars_window, bg='#F5F7FA')
    content_frame.pack(pady=10, padx=30, fill='both', expand=True)
    
    # Input frame with white background
    input_frame = tk.Frame(content_frame, bg="#FFFFFF", relief='flat', bd=0)
    input_frame.pack(pady=20, padx=20, fill='x')
    
    # Dependent variable section with proper alignment
    dep_label = tk.Label(input_frame, text="Dependent Variable:", bg='#FFFFFF', fg='#1e3d58', font=("Helvetica", 12, 'bold'))
    dep_label.pack(anchor='w', padx=20, pady=(15, 5))
    
    dep_entry = tk.Entry(input_frame, width=50, bg='#ECF0F1', font=("Arial", 11), bd=0, relief='flat', fg='#2C3E50')
    dep_entry.pack(anchor='w', padx=20, pady=(0, 15), ipady=8)

    # Independent variables info section
    info_frame = tk.Frame(content_frame, bg='#F5F7FA')
    info_frame.pack(pady=15, padx=20, fill='x')
    
    info_label = tk.Label(info_frame, text="✓ Independent variables are selected automatically from remaining columns.", bg='#F5F7FA', fg='#27AE60', font=("Helvetica", 11, 'italic'))
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
    button_frame = tk.Frame(content_frame, bg='#F5F7FA')
    button_frame.pack(pady=30, fill='x')
    
    # Set Variables button
    set_btn = tk.Button(button_frame, text="Set Variables", command=set_variables, bg='#3498DB', fg='white', font=("Helvetica", 12, 'bold'), relief="raised", bd=0, width=15, padx=15, pady=8, cursor='hand2')
    set_btn.pack(side='left', padx=10)
    
    # Preview Dataset button
    preview_btn = tk.Button(button_frame, text="Preview Dataset", command=preview_dataset, bg='#9B59B6', fg='white', font=("Helvetica", 12, 'bold'), relief="raised", bd=0, width=15, padx=15, pady=8, cursor='hand2')
    preview_btn.pack(side='left', padx=10)

    # Back button
    def go_back_to_root():
        enter_vars_window.destroy()
        root.deiconify()
    
    back_btn = tk.Button(button_frame, text="Back", command=go_back_to_root, bg='#E74C3C', fg='white', font=("Helvetica", 12, 'bold'), relief="raised", bd=0, width=15, padx=15, pady=8, cursor='hand2')
    back_btn.pack(side='left', padx=10)
    


# Function to go to the algorithm selection page,------------------third page -------------------------------------
def go_to_algorithm_page():
    algo_window = tk.Toplevel(root)
    algo_window.title("Select Algorithm")
    algo_window.geometry("800x600")
    algo_window.configure(bg='#F5F7FA')

    # Header frame
    header_frame = tk.Frame(algo_window, bg='#1e3d58', height=70)
    header_frame.pack(fill='x')
    
    # Title
    title_label = tk.Label(header_frame, text="Select Algorithm", bg='#1e3d58', fg='#FFFFFF', font=("Helvetica", 22, 'bold'))
    title_label.pack(pady=15)

    # Content frame
    content_frame = tk.Frame(algo_window, bg='#F5F7FA')
    content_frame.pack(pady=50, padx=30, fill='both', expand=True)

    def select_algorithm():
        algo_window.destroy()
        go_to_train_model_page()

    tk.Button(content_frame, text="Logistic Regression", command=select_algorithm, bg='#3498DB', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=0, padx=30, pady=12, width=25, cursor='hand2').pack(pady=20)
    
    def go_back_to_root():
        algo_window.destroy()
        root.deiconify()
    
    tk.Button(content_frame, text="Back", command=go_back_to_root, bg='#E74C3C', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=0, padx=30, pady=12, width=25, cursor='hand2').pack(pady=20)

# Function to go to the training model page ,------------------------4th page ---------------------------------------
def go_to_train_model_page():
    train_model_window = tk.Toplevel(root)
    train_model_window.title("Train Model")
    train_model_window.geometry("800x600")
    train_model_window.configure(bg='#F5F7FA')

    # Header frame
    header_frame = tk.Frame(train_model_window, bg='#1e3d58', height=70)
    header_frame.pack(fill='x')
    
    # Title
    title_label = tk.Label(header_frame, text="Train Model", bg='#1e3d58', fg='#FFFFFF', font=("Helvetica", 22, 'bold'))
    title_label.pack(pady=15)

    # Content frame
    content_frame = tk.Frame(train_model_window, bg='#F5F7FA')
    content_frame.pack(pady=50, padx=30, fill='both', expand=True)

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

    tk.Button(content_frame, text="Train Model", command=train_model, bg='#27AE60', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=0, padx=30, pady=12, width=25, cursor='hand2').pack(pady=20)
    tk.Button(content_frame, text="Go to Prediction Page", command=go_to_prediction_page, bg='#3498DB', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=0, padx=30, pady=12, width=25, cursor='hand2').pack(pady=20)
    
    def go_back_to_root():
        train_model_window.destroy()
        root.deiconify()
    
    tk.Button(content_frame, text="Back", command=go_back_to_root, bg='#E74C3C', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=0, padx=30, pady=12, width=25, cursor='hand2').pack(pady=20)

# Function to go to the prediction page , ----------------------------5th page -------------------------
def go_to_prediction_page():
    prediction_window = tk.Toplevel(root)
    prediction_window.title("Enter Input Values")
    prediction_window.state('zoomed')  # Full screen on Windows
    prediction_window.configure(bg='#F5F7FA')

    # Header frame
    header_frame = tk.Frame(prediction_window, bg='#1e3d58', height=80)
    header_frame.pack(fill='x')
    
    # Title
    title_label = tk.Label(header_frame, text="Make a Prediction", bg='#1e3d58', fg='#FFFFFF', font=("Helvetica", 28, 'bold'))
    title_label.pack(pady=15)

    # Main content frame
    main_frame = tk.Frame(prediction_window, bg='#F5F7FA')
    main_frame.pack(pady=20, padx=30, fill='both', expand=True)

    # Create left and right columns without boxes
    left_frame = tk.Frame(main_frame, bg='#F5F7FA')
    left_frame.pack(side='left', fill='both', expand=True, padx=15)

    right_frame = tk.Frame(main_frame, bg='#F5F7FA')
    right_frame.pack(side='right', fill='both', expand=True, padx=15)

    inputs = []

    # Distribute input fields between left and right columns
    mid_point = len(indep_selected) // 2
    
    for idx, feature in enumerate(indep_selected):
        # Choose left or right frame
        target_frame = left_frame if idx < mid_point else right_frame
        
        # Create frame for each feature with subtle background
        feature_frame = tk.Frame(target_frame, bg='#FFFFFF', relief='flat', bd=0)
        feature_frame.pack(anchor='w', pady=12, padx=15, fill='x')
        
        # Label with improved styling
        label = tk.Label(feature_frame, text=f"{feature}:", bg='#FFFFFF', fg='#1e3d58', font=("Helvetica", 11, 'bold'), width=22, anchor='w')
        label.pack(anchor='w', pady=(8, 4), padx=10)
        
        # Entry with better styling
        entry = tk.Entry(feature_frame, bg='#ECF0F1', font=("Arial", 11), bd=0, relief='flat', fg='#2C3E50')
        entry.pack(anchor='w', padx=10, pady=(0, 8), fill='x', ipady=8)
        
        # Bind focus events for better UX
        def on_focus_in(event, e=entry):
            e.config(bg='#D5DBDB')
        def on_focus_out(event, e=entry):
            e.config(bg='#ECF0F1')
        
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)
        
        inputs.append(entry)

    def predict():
        try:
            input_values = [float(entry.get()) for entry in inputs]  # Get input values from the user
            input_array = np.array(input_values).reshape(1, -1)  # Reshape the input values
            input_array = scaler.transform(input_array)  # Scale input data using the fitted scaler
            
            prediction = classifier.predict(input_array)  # Get the prediction from the classifier
            
            # Display a message depending on the prediction result
            if prediction[0] == 1:
                messagebox.showinfo("Prediction Result", "🔴 Predicted: Diabetes")
            else:
                messagebox.showinfo("Prediction Result", "🟢 Predicted: No Diabetes")
    
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {e}")

    # Button frame for proper alignment
    button_frame = tk.Frame(prediction_window, bg='#F5F7FA')
    button_frame.pack(pady=25, fill='x')
    
    # Predict button
    predict_btn = tk.Button(button_frame, text="Predict", command=predict, bg='#27AE60', fg='white', font=("Helvetica", 13, 'bold'), relief="raised", bd=0, width=18, padx=20, pady=12, cursor='hand2')
    predict_btn.pack(side='left', padx=10)
    
    # Back button
    def go_back_to_root():
        prediction_window.destroy()
        root.deiconify()
    
    back_btn = tk.Button(button_frame, text="Back", command=go_back_to_root, bg='#E74C3C', fg='white', font=("Helvetica", 13, 'bold'), relief="raised", bd=0, width=18, padx=20, pady=12, cursor='hand2')
    back_btn.pack(side='left', padx=10)






# Main , ---------------------------------------------root  Window-------------------------
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("900x700")
root.configure(bg='#F5F7FA')


#title , navbar box
navbar = tk.Frame(root, bg="#1e3d58", height=30,)
navbar.pack(fill="x")
# Title inside navbar
title = tk.Label(navbar,text="🏥 Diabetes Prediction System",bg="#1e3d58",fg="#FFFFFF",font=("Helvetica", 26, "bold"))
title.pack(pady=20)

#description box
footer = tk.Frame(root,bg='#F5F7FA',height=30)
footer.pack_propagate(True)
center_box = tk.Frame(footer,bg="#1e3d58",bd=2,padx=50,pady=5).pack(expand=True)   
description = tk.Label(center_box,bg="#1e3d58",text="This system using Machine Learning that analyzes patient health data and predicts diabetes or no diabetes",fg="white",pady=10,font=("Helvetica", 10, "bold")).pack(padx=30,pady=30)
footer.pack(side="top", fill="x",padx=30,pady=30 )     


# Load Button 
load_btn = tk.Button(root, text="Load Dataset", command=load_dataset, bg='#3498DB', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=0, padx=30, pady=12, width=20, cursor='hand2')
load_btn.pack(pady=40)

#next Button
next_button = tk.Button(root, text="Next", command=go_to_enter_variables_page, bg='#27AE60', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=0, padx=30, pady=12, width=20, cursor='hand2')





# preview dataset function----
def preview_dataset():
    global df

    if df is None:
        return

    # new window
    preview = tk.Toplevel(root)
    preview.title("Dataset Preview")
    preview.geometry("1000x500")
    preview.configure(bg='#F5F7FA')

    # frame
    frame = tk.Frame(preview, bg='#F5F7FA')
    frame.pack(fill="both", expand=True)

    # table
    tree = ttk.Treeview(frame, style="Treeview")

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

# preview Button 
preview_button =tk.Button(root, text="Preview Dataset", command=preview_dataset, bg='#9B59B6', fg='white', font=("Helvetica", 14, 'bold'), relief="raised", bd=0, padx=30, pady=12, width=20, cursor='hand2')
root.mainloop()