# service-ambitions

## Configuration

The CLI requires an OpenAI API key available in the `OPENAI_API_KEY` environment variable. The key is loaded after `.env` files are processed and the application will exit if the variable is missing.

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_api_key_here
```

For production deployments, inject the variable using your platform's secret manager instead of committing keys to source control.

