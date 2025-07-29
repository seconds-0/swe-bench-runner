# GitHub Secrets Setup

## Adding API Keys for CI Integration Tests

To enable integration tests in CI, you need to add API keys as GitHub secrets.

## OpenRouter API Key (Recommended)

OpenRouter provides universal access to multiple models with a single API key, making it ideal for CI testing.

### Via GitHub UI

1. Go to your repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add:
   - Name: `OPENROUTER_API_KEY`
   - Value: `your-openrouter-api-key`

### Via GitHub CLI

```bash
# Install GitHub CLI if needed
# brew install gh  # macOS
# sudo apt install gh  # Ubuntu

# Authenticate
gh auth login

# Add the secret
gh secret set OPENROUTER_API_KEY --body "your-openrouter-api-key"
```

## Other Provider Keys (Optional)

If you want to test provider-specific features:

### OpenAI
```bash
# For CI testing with provided key
gh secret set OPENAI_API_KEY --body "your-openai-api-key"
```

### Anthropic
```bash
gh secret set ANTHROPIC_API_KEY --body "your-anthropic-key"
```

## Verifying Secrets

After adding secrets, you can verify they're available:

1. Check the "Secrets" section in repository settings
2. Run a CI workflow with `[integration]` in the commit message
3. Check the integration test logs

## Cost Management

The provided OpenRouter key has usage limits. To monitor usage:
1. Visit https://openrouter.ai/settings
2. Check the usage dashboard
3. Set spending limits if needed

## Security Notes

- Secrets are encrypted and only exposed to workflows
- Never log or output API keys in CI
- Rotate keys periodically
- Use read-only keys when possible
- Monitor usage for unexpected spikes

## CI Workflow

The integration tests will run:
- Weekly on Monday at 3 AM UTC
- On release branches
- When commit message contains `[integration]`
- On manual workflow dispatch

With OpenRouter configured, all provider integration tests can run using a single API key!
