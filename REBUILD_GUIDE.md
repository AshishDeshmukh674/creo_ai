# 🔄 COMPLETE REBUILD GUIDE - Fresh Training to Production


------------------------------------------------------------------------------------------------

cd ..\..\py
# Activate your environment if not already done
# For conda environment (your setup):form main folder
conda activate .\venv
# OR if above doesn't work, try:
# .\venv\python.exe -m pip install -r requirements.txt (direct usage)

pip install -r requirements.txt
-----------------------------------------------------------------------------------
# Prepare dataset (if not already done)
python prepare_dataset.py

# Train the model
python train_lora_t5.py

# Merge LoRA weights
python merge_lora.py

# Export to ONNX
python export_onnx.py

----------------------------------------------------
Cmake installation:
winget install Kitware.CMake
or

$url = "https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-windows-x86_64.zip"; $output = "$PWD\cmake.zip"; Invoke-WebRequest -Uri $url -OutFile $output; Expand-Archive $output -DestinationPath "$PWD\cmake" -Force

Now let me test if the portable CMake works:
.\cmake\cmake-3.30.2-windows-x86_64\bin\cmake.exe --version

Let me check what files are actually available in the current directory:
Get-ChildItem -Path . -Name

cd "C:\Users\ashis\Desktop\creo_ai\cpp\build"



----------------------------------------


cd cpp
mkdir build -Force 
cd build
cmake ..

# From project root
mkdir cpp\onnx_model
copy py\onnx_model\t5_creo.onnx cpp\onnx_model\
copy py\onnx_model\spiece.model cpp\onnx_model\


cmake --build . --config Release











cd ..\Release
.\nl2trail.exe "create a cube"


---------------------------------------------------------------------------------


## 📁 Clean Project Structure

C:\Users\asdeshmukh\Desktop\ai_model\
├── .venv\              # Python virtual environment
├── cpp\                # C++ production code
│   ├── include\        # Header files
│   ├── src\           # Source files
│   └── CMakeLists.txt # Build configuration
├── data\              # Training data
│   └── creo_dataset.jsonl
└── py\                # Python training scripts
    ├── prepare_dataset.py
    ├── train_lora_t5.py
    ├── merge_lora.py
    ├── export_onnx.py
    └── requirements.txt


## 🐍 *Step 1: Setup Python Environment (1 minute)*

powershell
# Navigate to project
cd "C:\Users\asdeshmukh\Desktop\ai_model"

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -r py\requirements.txt


## 📊 *Step 2: Prepare Training Data (30 seconds)*

powershell
# Process JSONL dataset into HuggingFace format
python py\prepare_dataset.py


*Expected Output:*
- Creates py\ds_creo\ directory
- Converts 5 JSONL examples to train/test split

## 🧠 *Step 3: Train the Model (2-3 minutes)*

powershell
# Train T5 model with LoRA fine-tuning
python py\train_lora_t5.py


*Expected Output:*
- Creates py\t5_creo_lora\ directory with adapter weights
- Shows training progress for 5 steps
- Model trains on your Creo commands

## 🔗 *Step 4: Merge Model Weights (30 seconds)*

powershell
# Merge LoRA adapter with base T5 model
python py\merge_lora.py


*Expected Output:*
- Creates py\t5_creo_merged\ directory with full model

## 📦 *Step 5: Export to ONNX (1 minute)*

powershell
# Convert PyTorch model to ONNX for C++ inference
python py\export_onnx.py


*Expected Output:*
- Creates py\onnx_model\ directory
- Exports t5_creo.onnx (300MB+ model file)
- Exports spiece.model (tokenizer)

## 🔧 *Step 6: Setup C++ Dependencies (if not done)*

powershell
# Only needed if ONNX Runtime not installed
# Download: https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip
# Extract to: C:\onnxruntime


## 🏗 *Step 7: Build C++ Application (1 minute)*

powershell
# Create build directory
mkdir cpp\build -Force
cd cpp\build

# Configure CMake
cmake ..

# Build Release version
cmake --build . --config Release


*Expected Output:*
- Creates cpp\build\Release\nl2trail.exe
- Copies ONNX Runtime DLL automatically

## 📂 *Step 8: Copy Model Files (30 seconds)*

powershell
# Copy trained models to C++ directory
mkdir ..\onnx_model -Force
Copy-Item "..\..\py\onnx_model\t5_creo.onnx" "..\onnx_model\"
Copy-Item "..\..\py\onnx_model\spiece.model" "..\onnx_model\"


## 🎯 *Step 9: Test Your Application*

powershell
# Test with training examples
.\Release\nl2trail.exe "Create a 50mm cube"
.\Release\nl2trail.exe "Create a cylinder with radius 25mm and height 100mm"
.\Release\nl2trail.exe "Make a rectangular block 30x20x15mm"


## ⏱ *Total Time: ~5-6 minutes*

- Data prep: 30 seconds
- Training: 2-3 minutes  
- Export: 1 minute
- Build: 1 minute
- Setup: 30 seconds

## 🎉 *Expected Final Result*

Your application will generate actual Creo trail commands like:

# CREO TRAIL FILE
~ Command `ProCmdDashboardActivate`
~ Activate `Start Here > Modeling`
~ Command `ProCmdModelNew`
~ FocusIn `UITools`
~ Activate `UITools > Rectangle`
~ Input `50` `50`
~ Command `ProCmdExtrudeDashboard`
~ Input `50`
~ Command `ProCmdDashboardAccept`


Ready to start? Run the commands in order! 🚀