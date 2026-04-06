#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SimPath Experiment Runner — play.sh
# ═══════════════════════════════════════════════════════════════
#
# 사전 준비:
#   cp .env.example .env && vim .env    # API 키 설정
#
# 실행:
#   ./play.sh assistments openai          풀 실험 (ASSISTments + OpenAI)
#   ./play.sh assistments anthropic       풀 실험 (ASSISTments + Anthropic)
#   ./play.sh assistments both            풀 실험 (ASSISTments + 둘 다)
#   ./play.sh assistments mock            LLM 없이 (mock ordering)
#   ./play.sh ednet openai                풀 실험 (EdNet + OpenAI)
#
# 단계별 실행:
#   ./play.sh assistments download        데이터 다운로드만
#   ./play.sh assistments preprocess      전처리만
#   ./play.sh assistments train           KT 모델 학습만
#   ./play.sh assistments run openai      실험만
#
# 옵션:
#   ./play.sh assistments openai quick    소량 검증 (10명)
#   ./play.sh assistments openai noabl    ablation 스킵
#
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

# ── .env 로드 ──
if [ -f .env ]; then
  set -a
  source .env
  set +a
  echo "[.env] loaded"
else
  echo "[.env] not found — cp .env.example .env 로 생성하세요"
fi

# ── Conda ──
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sim

# ── 인자 파싱 ──
DATASET="${1:-assistments}"
SECOND="${2:-mock}"
THIRD="${3:-}"

SCRIPT="scripts/run_real_experiment.py"
EXTRA_FLAGS=""

# 단계별 실행 감지
case "$SECOND" in
  download|preprocess|train)
    STAGE="$SECOND"
    PROVIDER="mock"
    ;;
  run)
    STAGE="run"
    PROVIDER="${THIRD:-mock}"
    ;;
  mock|openai|anthropic|both)
    STAGE="all"
    PROVIDER="$SECOND"
    ;;
  *)
    echo "Usage: ./play.sh [assistments|ednet] [mock|openai|anthropic|both|download|preprocess|train|run] [quick|noabl]"
    exit 1
    ;;
esac

# 추가 옵션
case "$THIRD" in
  quick)  EXTRA_FLAGS="--quick" ;;
  noabl)  EXTRA_FLAGS="--no-ablation" ;;
esac
case "${4:-}" in
  quick)  EXTRA_FLAGS="--quick" ;;
  noabl)  EXTRA_FLAGS="--no-ablation" ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " SimPath Experiment"
echo " Dataset  : $DATASET"
echo " Stage    : $STAGE"
echo " Provider : $PROVIDER"
[ -n "$EXTRA_FLAGS" ] && echo " Flags    : $EXTRA_FLAGS"
echo "═══════════════════════════════════════════════════════════"

# ── API 키 체크 ──
if [ "$STAGE" = "all" ] || [ "$STAGE" = "run" ]; then
  case "$PROVIDER" in
    openai)
      [ -z "${OPENAI_API_KEY:-}" ] && echo "Error: OPENAI_API_KEY not in .env" && exit 1
      echo " Model: GPT-5.4"
      ;;
    anthropic)
      [ -z "${ANTHROPIC_API_KEY:-}" ] && echo "Error: ANTHROPIC_API_KEY not in .env" && exit 1
      echo " Model: Claude Sonnet 4.6"
      ;;
    both)
      [ -z "${OPENAI_API_KEY:-}" ] && echo "Error: OPENAI_API_KEY not in .env" && exit 1
      [ -z "${ANTHROPIC_API_KEY:-}" ] && echo "Error: ANTHROPIC_API_KEY not in .env" && exit 1
      echo " Models: GPT-5.4 + Claude Sonnet 4.6"
      ;;
    mock)
      echo " Mode: mock (no LLM)"
      ;;
  esac
fi

# ── 단위 테스트 ──
echo ""
echo "[Test] Unit tests..."
python -m pytest tests/test_core.py -v --tb=short -q
echo ""

# ── 실행 ──
if [ "$STAGE" = "all" ]; then
  echo "[1/4] Downloading $DATASET data..."
  python "$SCRIPT" download --dataset "$DATASET"

  echo ""
  echo "[2/4] Preprocessing..."
  python "$SCRIPT" preprocess --dataset "$DATASET"

  echo ""
  echo "[3/4] Training KT models (AKT + SAINT + DKT)..."
  python "$SCRIPT" train --dataset "$DATASET"

  echo ""
  echo "[4/4] Running experiment..."
  python "$SCRIPT" run --dataset "$DATASET" --provider "$PROVIDER" $EXTRA_FLAGS
else
  python "$SCRIPT" "$STAGE" --dataset "$DATASET" --provider "$PROVIDER" $EXTRA_FLAGS
fi

# ── 결과 ──
echo ""
echo "Results:"
ls -lh outputs/results/*.json 2>/dev/null || echo "  (no results yet)"
ls -lh outputs/logs/*.jsonl 2>/dev/null || echo "  (no LLM logs)"
ls -lh outputs/checkpoints/*.pt 2>/dev/null || echo "  (no checkpoints)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Done!"
echo "═══════════════════════════════════════════════════════════"
