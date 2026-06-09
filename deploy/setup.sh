#!/bin/bash
# Oracle Cloud Always Free — one-shot setup script
# Run as: sudo bash setup.sh
set -e

APP_DIR="/home/ubuntu/local-llm-api"
REPO="https://github.com/sfc38/local-llm-api.git"

echo "==> Updating system packages"
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git curl

echo "==> Installing Ollama"
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama

echo "==> Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "   Ollama is up."

echo "==> Pulling model (this may take 5–10 minutes)"
ollama pull qwen2.5vl:3b

echo "==> Cloning repo"
sudo -u ubuntu git clone "$REPO" "$APP_DIR"

echo "==> Creating Python environment and installing dependencies"
sudo -u ubuntu bash -c "
    cd $APP_DIR
    python3 -m venv venv
    venv/bin/pip install --upgrade pip -q
    venv/bin/pip install -r requirements.txt -q
    mkdir -p logs
"

echo "==> Installing systemd services"
cp "$APP_DIR/deploy/local-llm-api.service"       /etc/systemd/system/
cp "$APP_DIR/deploy/local-llm-streamlit.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable local-llm-api local-llm-streamlit
systemctl start  local-llm-api local-llm-streamlit

echo "==> Configuring firewall (ufw)"
ufw allow OpenSSH
ufw allow 8000/tcp   # FastAPI
ufw allow 8501/tcp   # Streamlit
ufw --force enable

PUBLIC_IP=$(curl -s https://ifconfig.me || echo "<your-oracle-ip>")

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  FastAPI docs : http://$PUBLIC_IP:8000/docs"
echo "  Streamlit UI : http://$PUBLIC_IP:8501"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status  local-llm-api"
echo "  sudo systemctl restart local-llm-api"
echo "  sudo journalctl -u local-llm-api -f"
echo "  sudo journalctl -u local-llm-streamlit -f"
