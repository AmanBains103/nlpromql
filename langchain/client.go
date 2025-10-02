package langchain

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/prashantgupta17/nlpromql/llm"
	"github.com/prashantgupta17/nlpromql/prompts"
	"github.com/tmc/langchaingo/llms"
	// Dependencies for specific llms.Model implementations are managed in main.go
)

// LangChainClient implements the llm.LLMClient interface using LangChainGo.
type LangChainClient struct {
	llmModel llms.Model // Generic LangChainGo LLM model
}

// NewLangChainClient creates a new LangChainClient.
// The specific model (e.g., OpenAI, Anthropic) should be initialized and passed here.
func NewLangChainClient(model llms.Model) *LangChainClient {
	return &LangChainClient{
		llmModel: model,
	}
}

// GetMetricSynonyms gets synonyms for the given metrics from the LLM.
func (c *LangChainClient) GetMetricSynonyms(metricMap map[string]string) (map[string][]string, error) {
	if c.llmModel == nil {
		return nil, errors.New("LangChain LLM model is not initialized")
	}

	// Define batch size for parallel processing
	batchSize := 10
	totalMetrics := len(metricMap)
	
	// If the map is small enough, process in a single call
	if totalMetrics <= batchSize {
		return c.processMetricBatch(metricMap)
	}

	// Split metrics into batches
	batches := make([]map[string]string, 0)
	currentBatch := make(map[string]string)
	count := 0
	
	for metric, description := range metricMap {
		currentBatch[metric] = description
		count++
		
		if count >= batchSize {
			batches = append(batches, currentBatch)
			currentBatch = make(map[string]string)
			count = 0
		}
	}
	
	// Add remaining metrics to the last batch
	if len(currentBatch) > 0 {
		batches = append(batches, currentBatch)
	}

	// Process batches in parallel
	type result struct {
		synonyms map[string][]string
		err      error
	}
	
	resultChan := make(chan result, len(batches))
	
	// Launch goroutines for parallel processing
	for _, batch := range batches {
		go func(b map[string]string) {
			synonyms, err := c.processMetricBatch(b)
			resultChan <- result{synonyms: synonyms, err: err}
		}(batch)
	}
	
	// Collect results
	allSynonyms := make(map[string][]string)
	for i := 0; i < len(batches); i++ {
		res := <-resultChan
		if res.err != nil {
			return nil, fmt.Errorf("error processing batch: %w", res.err)
		}
		
		// Merge results
		for metric, synonyms := range res.synonyms {
			allSynonyms[metric] = synonyms
		}
	}
	
	return allSynonyms, nil
}

// processMetricBatch processes a single batch of metrics
func (c *LangChainClient) processMetricBatch(metricMap map[string]string) (map[string][]string, error) {
	metricMapJSON, err := json.MarshalIndent(metricMap, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("error marshalling metricMap: %w", err)
	}

	prompt := fmt.Sprintf(prompts.MetricSynonymPrompt, string(metricMapJSON))

	// TODO: Adjust for potential differences in how various models handle chat/completion and system prompts.
	// This is a simplified example assuming a completion-style model.
	// For chat models, the call would be llms.GenerateFromSinglePrompt or similar with specific message structuring.

	// Using llms.Call directly for simplicity, assuming the model supports simple text in/out.
	// More complex models/scenarios might require llms.CreateChatCompletion or llms.GenerateContent.
	// Corrected: llms.Call is a method on the model instance: c.llmModel.Call
	response, err := c.llmModel.Call(context.Background(), prompt) // Removed c.llmModel from args, added relevant llms.CallOptions if needed
	if err != nil {
		return nil, fmt.Errorf("LangChain LLM call failed: %w", err)
	}

	var synonyms map[string][]string
	if err := json.Unmarshal([]byte(response), &synonyms); err != nil {
		// Fallback: if the response is not a valid JSON map, wrap it in a "response" key.
		// This handles cases where the LLM might return a raw string or a list not directly unmarshallable.
		// More robust error handling and response parsing might be needed here based on observed LLM outputs.
		// e.g. some models might wrap their json output in ```json ... ```
		// For now, we'll try to unmarshal as is, and if it fails, assume it's a string that needs to be wrapped,
		// or it's a malformed JSON.
		// A more sophisticated approach would involve inspecting the string, trying to clean it, etc.
		return nil, fmt.Errorf("error unmarshalling LLM response: %w. Raw response: %s", err, response)
	}

	return synonyms, nil
}

// GetLabelSynonyms gets synonyms for the given labels from the LLM.
func (c *LangChainClient) GetLabelSynonyms(labelNames []string) (map[string][]string, error) {
	if c.llmModel == nil {
		return nil, errors.New("LangChain LLM model is not initialized")
	}

	// Define batch size for parallel processing
	batchSize := 20
	totalLabels := len(labelNames)
	
	// If the list is small enough, process in a single call
	if totalLabels <= batchSize {
		return c.processLabelBatch(labelNames)
	}

	// Split labels into batches
	batches := make([][]string, 0)
	for i := 0; i < totalLabels; i += batchSize {
		end := i + batchSize
		if end > totalLabels {
			end = totalLabels
		}
		batches = append(batches, labelNames[i:end])
	}

	// Process batches in parallel
	type result struct {
		synonyms map[string][]string
		err      error
	}
	
	resultChan := make(chan result, len(batches))
	
	// Launch goroutines for parallel processing
	for _, batch := range batches {
		go func(b []string) {
			synonyms, err := c.processLabelBatch(b)
			resultChan <- result{synonyms: synonyms, err: err}
		}(batch)
	}
	
	// Collect results
	allSynonyms := make(map[string][]string)
	for i := 0; i < len(batches); i++ {
		res := <-resultChan
		if res.err != nil {
			return nil, fmt.Errorf("error processing batch: %w", res.err)
		}
		
		// Merge results
		for label, synonyms := range res.synonyms {
			allSynonyms[label] = synonyms
		}
	}
	
	return allSynonyms, nil
}

// processLabelBatch processes a single batch of labels
func (c *LangChainClient) processLabelBatch(labelNames []string) (map[string][]string, error) {
	labelNamesJSON, err := json.MarshalIndent(labelNames, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("error marshalling labelNames: %w", err)
	}

	prompt := fmt.Sprintf(prompts.LabelSynonymPrompt, string(labelNamesJSON))

	// Corrected: llms.Call is a method on the model instance: c.llmModel.Call
	response, err := c.llmModel.Call(context.Background(), prompt) // Removed c.llmModel from args
	if err != nil {
		return nil, fmt.Errorf("LangChain LLM call failed: %w", err)
	}

	var synonyms map[string][]string
	if err := json.Unmarshal([]byte(response), &synonyms); err != nil {
		return nil, fmt.Errorf("error unmarshalling LLM response: %w. Raw response: %s", err, response)
	}

	return synonyms, nil
}

// ProcessUserQuery processes the user query and returns relevant information.
func (c *LangChainClient) ProcessUserQuery(userQuery string) (map[string]interface{}, error) {
	if c.llmModel == nil {
		return nil, errors.New("LangChain LLM model is not initialized")
	}

	prompt := fmt.Sprintf(prompts.ProcessQueryPrompt, userQuery)

	// Corrected: llms.Call is a method on the model instance: c.llmModel.Call
	response, err := c.llmModel.Call(context.Background(), prompt) // Removed c.llmModel from args
	if err != nil {
		return nil, fmt.Errorf("LangChain LLM call failed: %w", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal([]byte(response), &result); err != nil {
		return nil, fmt.Errorf("error unmarshalling LLM response: %w. Raw response: %s", err, response)
	}

	return result, nil
}

// GetPromQLFromLLM gets PromQL queries from the LLM based on the user query and relevant context.
func (c *LangChainClient) GetPromQLFromLLM(userQuery string, relevantMetrics llm.RelevantMetricsMap, relevantLabels llm.RelevantLabelsMap, relevantHistory map[string]interface{}) ([]string, error) {
	if c.llmModel == nil {
		return nil, errors.New("LangChain LLM model is not initialized")
	}

	relevantMetricsJSON, err := json.MarshalIndent(relevantMetrics, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("error marshalling relevantMetrics: %w", err)
	}

	relevantLabelsJSON, err := json.MarshalIndent(relevantLabels, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("error marshalling relevantLabels: %w", err)
	}

	relevantHistoryJSON, err := json.MarshalIndent(relevantHistory, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("error marshalling relevantHistory: %w", err)
	}

	// Construct the user prompt part for GetPromQLFromLLM
	// This follows the structure observed in openai/client.go's newFunction
	userPromptForPromQL := fmt.Sprintf("#Relevant Metrics:\n%s\n\n#Relevant Labels:\n%s\n\n#Relevant History:\n%s\n\n#User Query:\n%s",
		string(relevantMetricsJSON),
		string(relevantLabelsJSON),
		string(relevantHistoryJSON),
		userQuery,
	)

	// For LangChainGo, the system prompt is often handled as part of the model's initialization
	// or via specific options in the Call/GenerateContent methods.
	// Here, we'll pass it as part of the prompt itself if using a simple llms.Call.
	// If using a chat model, it would be a SystemChatMessage.
	// This assumes the model can take a combined system + user prompt.
	// A more sophisticated implementation would use llms.GenerateContent with specific message types.

	// According to LangchainGo docs, for models that support SystemPrompt
	// it should be passed as an option if the specific LLM wrapper supports it
	// or as the first message in a chat sequence.
	// For a generic llms.Call, we might prepend it to the user prompt.
	// However, the `prompts.SystemPrompt` is quite large and might be better handled
	// by specific model capabilities (e.g. `llms.WithSystemPrompt` if available or by using `llms.ChatMessage` types).

	// For now, we will use llms.GenerateContent which allows specifying a slice of llms.MessageContent.
	// We will create two messages: one for the system prompt and one for the user prompt.

	messages := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, prompts.SystemPrompt),
		llms.TextParts(llms.ChatMessageTypeHuman, userPromptForPromQL),
	}

	options := []llms.CallOption{} // Add temperature, max tokens etc. here if needed.

	// Using GenerateContent for better control over message types (system vs user)
	// Note: Not all models in LangchainGo might support the System message type in the same way.
	// This part might need adjustment based on the specific llms.Model being used.
	// For example, some models might expect the system prompt as a specific field during initialization or call.
	// Corrected: llms.GenerateContent is a method on the model instance: c.llmModel.GenerateContent
	contentResponse, err := c.llmModel.GenerateContent(context.Background(), messages, options...) // Removed c.llmModel from args
	if err != nil {
		return nil, fmt.Errorf("LangChain LLM GenerateContent call failed: %w", err)
	}

	if len(contentResponse.Choices) == 0 {
		return nil, errors.New("LLM returned no choices")
	}

	response := contentResponse.Choices[0].Content

	var promqlOptions []struct {
		PromQL string  `json:"promql"`
		Score  float64 `json:"score"`
		// metric_label_pairs is ignored for now as we only need PromQL strings
	}

	if err := json.Unmarshal([]byte(response), &promqlOptions); err != nil {
		return nil, fmt.Errorf("error unmarshalling LLM response for PromQL: %w. Raw response: %s", err, response)
	}

	// Sort by score (descending) - already handled by prompt, but good practice
	// The prompt asks the LLM to sort, but we can re-sort if needed.
	// For now, we trust the LLM's sorting based on the prompt.

	var sortedPromqlStrings []string
	for _, option := range promqlOptions {
		sortedPromqlStrings = append(sortedPromqlStrings, option.PromQL)
	}

	return sortedPromqlStrings, nil
}

// Ensure LangChainClient implements the llm.LLMClient interface.
var _ llm.LLMClient = (*LangChainClient)(nil)
