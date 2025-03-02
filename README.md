## StarfishData CLI 🚀  


**Synthetic Q&A Data Generation CLI**  

![StarfishData](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Typer](https://img.shields.io/badge/CLI-Typer-blue) ![License](https://img.shields.io/badge/License-MIT-green)

StarfishData CLI is a **Python-based command-line tool** designed to **generate synthetic Q&A pairs** while ensuring **uniqueness** using **TF-IDF & cosine similarity**. Built on **Llama models**, it enables AI-powered dataset creation for training and research purposes.

---

## **📌 Features**
✅ **Generate Q&A pairs** from any topic  
✅ **Ensures uniqueness** (removes duplicates using TF-IDF & cosine similarity)  
✅ **Works with both remote & local models**  
✅ **CLI-based** (easy command-line execution)  
✅ **Downloads & manages models** automatically  

---

## **🚀 Installation**

### **Using Poetry (Recommended)**
```bash
poetry install
```

### **Manual Installation**
```bash
pip install typer torch transformers rich llama-cpp-python huggingface_hub scikit-learn
```

---

## **📖 Usage**

### **🔹 Get CLI Help**
```bash
starfishdata --help
```

### **🔹 Generate Q&A Pairs**
```bash
starfishdata generate --prompt "History of AI" --num-records 5
```

### **🔹 Download Model**
```bash
starfishdata download --hf-token <YOUR_HF_TOKEN>
```

### **🔹 Cleanup Models**
```bash
starfishdata cleanup-models
```

---

## **⚙️ CLI Reference**
### **1️⃣ `generate` – Generate Q&A Data**
```bash
starfishdata generate --prompt "Topic" --num-records 5
```
| Flag | Description | Default |
|------|------------|---------|
| `--prompt` | Topic to generate Q&A about | Required |
| `--num-records` | Number of Q&A pairs (max 100) | 1 |
| `--file` | GGUF model file | Default model |
| `--output-file` | Output JSONL file | `output.jsonl` |
| `--cleanup` | Delete models after inference | False |

---

### **2️⃣ `download` – Download Model**
```bash
starfishdata download --hf-token <YOUR_HF_TOKEN>
```
| Flag | Description | Required |
|------|------------|----------|
| `--name` | Hugging Face model name | ✅ |
| `--file` | GGUF model file | ✅ |
| `--hf-token` | Hugging Face API Token | ✅ |

---

### **3️⃣ `cleanup-models` – Delete Models**
```bash
starfishdata cleanup-models
```
Deletes all **downloaded models** from cache.

---

## **🔄 Switching & Using Custom Models**
StarfishData allows users to **switch models** or **use local models** for generation. You can specify a **different model file** or **use a locally downloaded GGUF model**.

### **📌 Using a Different Model**
```bash
starfishdata generate --prompt "History of AI" --num-records 5 --file "custom_model.gguf"
```
👉 **This allows you to use a different GGUF model file.**

### **📌 Downloading a New Model**
If you want to switch models, you can **download a different one** from Hugging Face:
```bash
starfishdata download --name "NewModel/Qwen2" --file "new_model.gguf" --hf-token <YOUR_HF_TOKEN>
```

### **📌 Using a Local Model**
You can use a **locally stored GGUF model** instead of downloading:
```bash
starfishdata generate --prompt "Space Exploration" --num-records 10 --file "/path/to/local_model.gguf"
```
**Ensure the model file exists before running this command.**

---

## **📄 License**
This project is licensed under the **MIT License**.

---

## **🌟 Contributing**
Contributions are welcome! To contribute:  
1. **Fork** the repo  
2. **Create a feature branch** (`feature-new-thing`)  
3. **Submit a PR** 🚀  

---

🎉 **Happy Coding!** 🚀  
Let me know if you need further refinements!
