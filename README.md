# Step 1: 

```bash
# sudo nano ~/.streamlit/config.toml
```
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
port = 8501

# Step 2:

```bash
streamlit run faal_pred_version_multilabel.py --server.port 8503
  ```
# Step3:
```bash
http://localhost:8501/
  ```
