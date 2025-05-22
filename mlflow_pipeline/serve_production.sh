#!/bin/bash
mlflow models serve -m "models:/HeartAttackRiskModel/Production" -p 5002