#!/usr/bin/env bash
set -e
python -m src.agent.eval.lama_eval
python -m src.agent.eval.gsm8k_eval
