# Natural Language to PromQL (nlpromql)

## 1. Overview

`nlpromql` is a Go application designed to translate natural language queries into Prometheus Query Language (PromQL) expressions. It leverages Large Language Models (LLMs) through the LangChainGo library to understand user intent and map it to relevant Prometheus metrics and labels. The system builds an internal information structure from your Prometheus instance to provide context to the LLM, enabling more accurate query generation.

The application can run in two modes:
-   **Chat Mode:** An interactive command-line interface where users can type natural language queries and receive PromQL suggestions.
-   **Server Mode:** Exposes an HTTP API endpoint (`/v1/promql`) that accepts natural language queries and returns PromQL.

## 2. Prerequisites

Before running `nlpromql`, ensure you have the following:

*   **Go:** Version 1.20 or higher.
*   **Prometheus Access:** Network access to a running Prometheus instance. Credentials may be required depending on your Prometheus setup.
*   **LLM API Keys:** Depending on the LLM model you intend to use, you'll need API keys for services like OpenAI, Anthropic, or Cohere.

## 3. Configuration

Configuration is primarily handled through environment variables and command-line flags.

### 3.1. Prometheus Configuration

The application requires access to your Prometheus instance to build its information structure. Configure this using the following environment variables:

*   `PROMETHEUS_URL`: The URL of your Prometheus server (e.g., `http://localhost:9090`). (Required)
*   `PROMETHEUS_USER`: Username for Prometheus basic authentication (if applicable).
*   `PROMETHEUS_PASSWORD`: Password for Prometheus basic authentication (if applicable).

If both `PROMETHEUS_USER` and `PROMETHEUS_PASSWORD` are set, basic authentication will be used. If only one is set, the application will report an error. If neither is set, no authentication will be used.

### 3.2. LLM Configuration

#### LLM Model Selection

*   **`-llm_model_name`** (Command-line flag)
    *   Specifies the LangChainGo LLM model to use.
    *   The format is typically `<provider>/<model_identifier>`.
    *   Default: `"openai/gpt-3.5-turbo"`
    *   Examples:
        *   `"openai/gpt-3.5-turbo"`
        *   `"openai/gpt-4"`
        *   `"anthropic/claude-2"`
        *   `"anthropic/claude-instant-1.2"`
        *   *(Support for Cohere models can be added in the future)*

#### LLM API Keys

API keys are required to authenticate with the LLM providers. They can be provided via command-line flags or environment variables. **The command-line flag will always take precedence if set.**

*   **OpenAI:**
    *   Flag: `-openai_api_key="YOUR_OPENAI_KEY"`
    *   Environment Variable: `OPENAI_API_KEY`

*   **Anthropic:**
    *   Flag: `-anthropic_api_key="YOUR_ANTHROPIC_KEY"`
    *   Environment Variable: `ANTHROPIC_API_KEY`

*   **Cohere:**
    *   Flag: `-cohere_api_key="YOUR_COHERE_KEY"`
    *   Environment Variable: `COHERE_API_KEY`
    *   *(Note: While the flag and environment variable are recognized, Cohere model integration is not yet fully implemented in the LLM selection switch in `main.go`.)*

If the required API key for the selected `llm_model_name` is not found either via its flag or environment variable, the application will print an error and exit.

#### Parallel Processing Configuration

The application supports parallel batch processing of LLM requests to improve performance when building the information structure from Prometheus. This is especially useful when dealing with large numbers of metrics and labels.

*   **`-batch_size`** (Command-line flag)
    *   Specifies the number of items (metrics or labels) to process in a single LLM request.
    *   Default: `10`
    *   Example: `-batch_size=20` will process 20 items per LLM request.

*   **`-max_workers`** (Command-line flag)
    *   Specifies the maximum number of parallel workers making concurrent LLM requests.
    *   Default: `5`
    *   Example: `-max_workers=10` will allow up to 10 concurrent LLM requests.

**Note:** Increasing these values can significantly speed up the initial information structure building phase, but be mindful of:
- Your LLM provider's rate limits
- API costs (more parallel requests may consume your quota faster)
- Memory usage on your system

## 4. Running the Application

### 4.1. Build

First, build the application:
```bash
go build .
```
This will create an executable named `nlpromql` (or `nlpromql.exe` on Windows).

### 4.2. Running in Chat Mode

Chat mode allows interactive querying.

**Example with OpenAI (using environment variable for API key):**
```bash
export PROMETHEUS_URL="http://localhost:9090"
export OPENAI_API_KEY="your_openai_api_key_here"
./nlpromql -mode="chat" -llm_model_name="openai/gpt-3.5-turbo"
```

**Example with OpenAI (using flag for API key):**
```bash
export PROMETHEUS_URL="http://localhost:9090"
./nlpromql -mode="chat" -llm_model_name="openai/gpt-3.5-turbo" -openai_api_key="your_openai_api_key_here"
```

**Example with custom batch processing configuration:**
```bash
export PROMETHEUS_URL="http://localhost:9090"
export OPENAI_API_KEY="your_openai_api_key_here"
./nlpromql -mode="chat" -llm_model_name="openai/gpt-3.5-turbo" -batch_size=20 -max_workers=10
```
This configuration will process 20 metrics/labels per LLM request with up to 10 concurrent workers, significantly speeding up the initial information structure build.

**Example with Anthropic (using environment variable for API key):**
```bash
export PROMETHEUS_URL="http://localhost:9090"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
./nlpromql -mode="chat" -llm_model_name="anthropic/claude-2"
```

**Example with Anthropic (using flag for API key):**
```bash
export PROMETHEUS_URL="http://localhost:9090"
./nlpromql -mode="chat" -llm_model_name="anthropic/claude-2" -anthropic_api_key="your_anthropic_api_key_here"
```

Once in chat mode, type your natural language query and press Enter. Type `exit` to quit.

### 4.3. Running in Server Mode

Server mode starts an HTTP server (default port: 8080) providing an API for PromQL generation.

**Example:**
```bash
export PROMETHEUS_URL="http://localhost:9090"
export OPENAI_API_KEY="your_openai_api_key_here"
./nlpromql -mode="server" -llm_model_name="openai/gpt-3.5-turbo" -port="8081"
```
The server will listen on port `8081`. You can then send GET requests to:
`http://localhost:8081/v1/promql?query=<your_natural_language_query>`

## 5. Development

(Placeholder for future development notes, e.g., running tests, code structure overview)

### TODO
*   Implement support for Cohere models in the LLM selection logic.
*   Add more sophisticated scoring for relevant metrics and labels.
*   Expand unit test coverage.
*   Consider adding caching for LLM responses to reduce costs and improve latency for repeated queries.

---
This README provides a basic guide to configuring and running `nlpromql`.
Remember to replace placeholder API keys and URLs with your actual values.
The application will first attempt to build an information structure from Prometheus, which may take some time on the first run depending on the size of your Prometheus data. Subsequent runs will load this structure from disk (by default, in an `info` directory).
```
