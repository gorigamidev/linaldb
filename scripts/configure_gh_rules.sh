#!/bin/bash

# Configuration script for LINALDB GitHub Repository Rulesets
# Requires: gh CLI authenticated with 'repo' scope

REPO="gorigamidev/linaldb"

echo "🚀 Configuring Repository Rulesets for $REPO..."

# Define the ruleset JSON
RULESET_JSON=$(cat <<EOF
{
  "name": "Main Protection",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["~DEFAULT_BRANCH"],
      "exclude": []
    }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_signatures" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          { "context": "Test" },
          { "context": "Format" },
          { "context": "Clippy" },
          { "context": "Smoke Test" }
        ]
      }
    }
  ]
}
EOF
)

# Use gh api to create the ruleset
echo "Creating/Updating ruleset via GitHub API..."
echo "$RULESET_JSON" | gh api --method POST /repos/$REPO/rulesets --input - \
  || echo "Ruleset already exists or error occurred. Check GitHub UI."

echo "✅ Configuration complete."
echo "Visit https://github.com/$REPO/settings/rules to verify."
