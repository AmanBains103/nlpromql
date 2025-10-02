package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/prashantgupta17/nlpromql/info_structure"
	"github.com/prashantgupta17/nlpromql/langchain"
	"github.com/prashantgupta17/nlpromql/llm"
	"github.com/prashantgupta17/nlpromql/prometheus"
	"github.com/prashantgupta17/nlpromql/query_processing"
	"github.com/prashantgupta17/nlpromql/server"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
	lcOpenai "github.com/tmc/langchaingo/llms/openai"
)

// TODO: Update README.md to document -llm_model_name, API key flags (-openai_api_key, -anthropic_api_key, -cohere_api_key), and their corresponding environment variables.
func main() {
	mode := flag.String("mode", "server", "Mode of operation: 'server' or 'chat'")
	port := flag.String("port", "8080", "Port for the HTTP server (server mode only)")
	llmModelNameFlag := flag.String("llm_model_name", "openai/gpt-3.5-turbo", "The identifier for the LangChainGo LLM model to use (e.g., 'openai/gpt-3.5-turbo', 'anthropic/claude-2').")
	openaiAPIKeyFlag := flag.String("openai_api_key", "", "OpenAI API key. Overrides OPENAI_API_KEY environment variable.")
	anthropicAPIKeyFlag := flag.String("anthropic_api_key", "", "Anthropic API key. Overrides ANTHROPIC_API_KEY environment variable.")
	_ = flag.String("cohere_api_key", "", "Cohere API key. Overrides COHERE_API_KEY environment variable.") // Defined, not used yet - assigned to blank identifier
	batchSize := flag.Int("batch_size", 10, "Batch size for parallel LLM requests when building information structure")
	maxWorkers := flag.Int("max_workers", 5, "Maximum number of parallel workers for LLM requests")

	flag.Parse()

	// API Key Resolution (Flag > Env)
	finalOpenAIAPIKey := *openaiAPIKeyFlag
	if finalOpenAIAPIKey == "" {
		finalOpenAIAPIKey = os.Getenv("OPENAI_API_KEY")
	}

	finalAnthropicAPIKey := *anthropicAPIKeyFlag
	if finalAnthropicAPIKey == "" {
		finalAnthropicAPIKey = os.Getenv("ANTHROPIC_API_KEY")
	}

	// // finalCohereAPIKey := *cohereAPIKeyFlag // Defined, not used yet
	// // finalCohereAPIKey := *cohereAPIKeyFlag // Commented out as it's not used yet
	// // if finalCohereAPIKey == "" {
	// // finalCohereAPIKey = os.Getenv("COHERE_API_KEY")
	// // }


	var lcModel llms.Model
	var err error
	modelName := *llmModelNameFlag

	fmt.Printf("Attempting to initialize LLM model: %s\n", modelName)

	switch {
	case strings.HasPrefix(modelName, "openai/"):
		if finalOpenAIAPIKey == "" {
			fmt.Fprintln(os.Stderr, "OpenAI API key not provided via flag (-openai_api_key) or environment variable (OPENAI_API_KEY).")
			os.Exit(1)
		}
		modelID := strings.TrimPrefix(modelName, "openai/")
		lcModel, err = lcOpenai.New(lcOpenai.WithToken(finalOpenAIAPIKey), lcOpenai.WithModel(modelID))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error initializing Langchain OpenAI model (%s): %v\n", modelID, err)
			os.Exit(1)
		}
		fmt.Printf("Successfully initialized Langchain OpenAI model: %s\n", modelID)
	case strings.HasPrefix(modelName, "anthropic/"):
		if finalAnthropicAPIKey == "" {
			fmt.Fprintln(os.Stderr, "Anthropic API key not provided via flag (-anthropic_api_key) or environment variable (ANTHROPIC_API_KEY).")
			os.Exit(1)
		}
		modelID := strings.TrimPrefix(modelName, "anthropic/")
		lcModel, err = anthropic.New(anthropic.WithModel(modelID)) // Assumes ANTHROPIC_API_KEY is read by New() or by http client
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error initializing Langchain Anthropic model (%s): %v\n", modelID, err)
			os.Exit(1)
		}
		fmt.Printf("Successfully initialized Langchain Anthropic model: %s\n", modelID)
	// TODO: Add case for "cohere/..." if/when Cohere is implemented
	default:
		fmt.Fprintf(os.Stderr, "Unsupported LLM model name: %s. Please use format like 'openai/model-id' or 'anthropic/model-id'.\n", modelName)
		os.Exit(1)
	}

	chosenLLMClient := langchain.NewLangChainClientWithBatchConfig(lcModel, *batchSize, *maxWorkers)
	// NewLangChainClient currently doesn't return an error. If it could, error should be handled:
	// if err != nil {
	// fmt.Fprintf(os.Stderr, "Error creating LangChainClient: %v\n", err)
	// os.Exit(1)
	// }
	
	fmt.Printf("LLM client configured with batch size: %d, max workers: %d\n", *batchSize, *maxWorkers)

	// 3. Get Prometheus Credentials from Environment Variables
	promURL, promUser, promPassword, err := getPrometheusCredentials()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error getting Prometheus credentials:", err)
		os.Exit(1)
	}

	promClient := prometheus.NewPrometheusConnect(promURL, promUser, promPassword)

	infoBuilder, err := info_structure.NewInfoBuilder(promClient, chosenLLMClient, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error getting info builder:", err)
		os.Exit(1)
	}

	err = infoBuilder.BuildInformationStructure()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error building information structure:", err)
		os.Exit(1)
	}
	fmt.Println("Information Structure Built Successfully.")
	// Verbose printing of map lengths can be removed or put behind a debug flag if too noisy
	// fmt.Println("Metric Map:", len(infoBuilder.MetricMap.AllNames))
	// fmt.Println("Label Map:", len(infoBuilder.LabelMap.AllNames))
	// fmt.Println("MetricLabelMap:", len(*infoBuilder.MetricLabelMap))
	// fmt.Println("LabelValueMap:", len(*infoBuilder.LabelValueMap))
	// fmt.Println("NlpToMetricMap:", len(*infoBuilder.NlpToMetricMap))


	// Main application logic based on mode
	switch *mode {
	case "server":
		promqlServer := server.NewPromQLServer(
			chosenLLMClient,
			*infoBuilder.MetricMap,
			*infoBuilder.LabelMap,
			*infoBuilder.MetricLabelMap,
			*infoBuilder.LabelValueMap,
			*infoBuilder.NlpToMetricMap,
		)
		fmt.Printf("Starting server on port %s...\n", *port)
		if err := promqlServer.Start(*port); err != nil {
			fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
			os.Exit(1)
		}
	case "chat":
		fmt.Println("Entering chat mode...")
		runChatMode(chosenLLMClient,
			*infoBuilder.MetricMap,
			*infoBuilder.LabelMap,
			*infoBuilder.MetricLabelMap,
			*infoBuilder.LabelValueMap,
			*infoBuilder.NlpToMetricMap,
		)
	default:
		fmt.Fprintf(os.Stderr, "Invalid mode: %s. Use 'server' or 'chat'.\n", *mode)
		os.Exit(1)
	}
}

func runChatMode(llmClient llm.LLMClient, metricMap info_structure.MetricMap, labelMap info_structure.LabelMap,
	metricLabelMap info_structure.MetricLabelMap, labelValueMap info_structure.LabelValueMap,
	nlpToMetricMap info_structure.NlpToMetricMap) {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("Enter your query about Prometheus data (or type 'exit'): ")
		userQuery, _ := reader.ReadString('\n')
		userQuery = strings.TrimSpace(userQuery)

		if userQuery == "exit" {
			break
		}

		_, relevantMetrics, relevantLabels, relevantHistory, err := query_processing.ProcessUserQuery(
			llmClient, userQuery, metricMap, labelMap, metricLabelMap, labelValueMap, nlpToMetricMap,
		)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error processing user query:", err)
			continue
		}

		// Debugging prints for relevance data can be verbose; consider a debug flag for these
		// fmt.Println("Possible Matches:", possibleMatches)
		// fmt.Println("Relevant Metrics:", relevantMetrics)
		// fmt.Println("Relevant Labels:", relevantLabels)
		// fmt.Println("Relevant History:", relevantHistory)

		promqlOptions, err := llmClient.GetPromQLFromLLM(userQuery, relevantMetrics, relevantLabels, relevantHistory)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error generating PromQL options:", err)
			continue
		}

		if len(promqlOptions) == 0 {
			fmt.Println("No PromQL queries generated for the given input.")
		} else {
			fmt.Println("Generated PromQL options:")
			for i, option := range promqlOptions {
				fmt.Printf("%d. %s\n", i+1, option)
			}
		}
	}
}

// getPrometheusCredentials retrieves Prometheus credentials from environment variables.
func getPrometheusCredentials() (string, string, string, error) {
	promURL := os.Getenv("PROMETHEUS_URL")
	if promURL == "" {
		return "", "", "", fmt.Errorf("PROMETHEUS_URL environment variable not set")
	}

	promUser := os.Getenv("PROMETHEUS_USER")
	promPassword := os.Getenv("PROMETHEUS_PASSWORD")

	// Optional: Check if both username and password are provided if one is present
	if (promUser != "" && promPassword == "") || (promUser == "" && promPassword != "") {
		return "", "", "", fmt.Errorf("both PROMETHEUS_USER and PROMETHEUS_PASSWORD must be set if one is provided, or neither")
	}

	return promURL, promUser, promPassword, nil
}
