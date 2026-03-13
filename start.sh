#!/bin/bash
# MiroFish startup script - Anthropic backend via LiteLLM proxy

set -e

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ERROR: ANTHROPIC_API_KEY not set"
  echo "Usage: ANTHROPIC_API_KEY=sk-ant-... ./start.sh"
  exit 1
fi

if [ -z "$ZEP_API_KEY" ]; then
  echo "ERROR: ZEP_API_KEY not set"
  echo "Get a free key at: https://app.getzep.com/"
  echo "Usage: ANTHROPIC_API_KEY=sk-ant-... ZEP_API_KEY=z_... ./start.sh"
  exit 1
fi

# Update .env with Zep key
sed -i '' "s/your_zep_api_key_here/$ZEP_API_KEY/" .env

echo "Starting LiteLLM proxy (Anthropic backend) on port 4000..."
python3.11 -m litellm --model anthropic/claude-sonnet-4-5 --port 4000 &
LITELLM_PID=$!
echo "LiteLLM PID: $LITELLM_PID"

sleep 3

echo "Starting MiroFish backend on port 5001..."
cd backend
source .venv/bin/activate
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY python run.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
cd ..

echo "Starting MiroFish frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ MiroFish running:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:5001"
echo "   LiteLLM:  http://localhost:4000"
echo ""
echo "Press Ctrl+C to stop all services"

trap "kill $LITELLM_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
