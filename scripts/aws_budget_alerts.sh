#!/usr/bin/env bash
# Create a monthly AWS cost budget that emails an alert at every $STEP of spend, up to $MAX.
#
# Credits are excluded from the tracked cost, so alerts fire on real usage even while
# promotional credits cover the bill.
#
# Usage: scripts/aws_budget_alerts.sh you@example.com [STEP=5] [MAX=50]
# AWS caps notifications per budget, so the thresholds are spread over as many budgets as needed.
set -euo pipefail

email="${1:?usage: $0 EMAIL [STEP] [MAX]}"
step="${2:-5}"
max="${3:-50}"
per_budget=5 # AWS limit on notifications per budget
prefix="llm-gateway-monthly"

account_id="$(aws sts get-caller-identity --query Account --output text)"
export AWS_REGION=us-east-1 # the Budgets API lives in us-east-1

# Run an AWS call; treat "already exists" as success, fail loudly on anything else.
idempotent() { # description, command...
  local desc="$1" err
  shift
  if err="$("$@" 2>&1 >/dev/null)"; then
    echo "  created: $desc"
  elif [[ "$err" == *DuplicateRecord* ]]; then
    echo "  exists:  $desc"
  else
    echo "  FAILED:  $desc" >&2
    echo "$err" >&2
    exit 1
  fi
}

create_budget() { # name
  idempotent "budget $1 (limit \$$max/month)" aws budgets create-budget --account-id "$account_id" --budget "{
    \"BudgetName\": \"$1\",
    \"BudgetType\": \"COST\",
    \"TimeUnit\": \"MONTHLY\",
    \"BudgetLimit\": {\"Amount\": \"$max\", \"Unit\": \"USD\"},
    \"CostTypes\": {\"IncludeCredit\": false, \"IncludeRefund\": false}
  }"
}

add_alert() { # budget-name threshold
  idempotent "alert at \$$2 on $1" aws budgets create-notification --account-id "$account_id" --budget-name "$1" \
    --notification "NotificationType=ACTUAL,ComparisonOperator=GREATER_THAN,Threshold=$2,ThresholdType=ABSOLUTE_VALUE" \
    --subscribers "SubscriptionType=EMAIL,Address=$email"
}

i=0
for threshold in $(seq "$step" "$step" "$max"); do
  name="$prefix-$((i / per_budget + 1))"
  if ((i % per_budget == 0)); then create_budget "$name"; fi
  add_alert "$name" "$threshold"
  i=$((i + 1))
done

echo "Done. Check: aws budgets describe-budgets --account-id $account_id --region us-east-1"
