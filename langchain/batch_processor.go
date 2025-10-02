package langchain

import (
	"context"
	"fmt"
	"log"
	"sync"
)

// BatchProcessor handles parallel batch processing of LLM requests
type BatchProcessor struct {
	batchSize int
	maxWorkers int
}

// NewBatchProcessor creates a new batch processor with specified batch size and max workers
func NewBatchProcessor(batchSize, maxWorkers int) *BatchProcessor {
	if batchSize <= 0 {
		batchSize = 10 // default batch size
	}
	if maxWorkers <= 0 {
		maxWorkers = 5 // default max workers
	}
	return &BatchProcessor{
		batchSize:  batchSize,
		maxWorkers: maxWorkers,
	}
}

// ProcessMetricBatches processes metric synonym requests in parallel batches
func (bp *BatchProcessor) ProcessMetricBatches(
	ctx context.Context,
	metricMap map[string]string,
	processFunc func(batch map[string]string) (map[string][]string, error),
) (map[string][]string, error) {
	
	// Create batches
	batches := bp.createMetricBatches(metricMap)
	log.Printf("Created %d metric batches (batch size: %d, total metrics: %d)", 
		len(batches), bp.batchSize, len(metricMap))
	
	// Process batches in parallel
	log.Printf("Processing metric batches with %d parallel workers", bp.maxWorkers)
	results, err := bp.processBatchesInParallel(ctx, batches, processFunc)
	if err != nil {
		return nil, err
	}
	
	// Merge results
	merged := make(map[string][]string)
	for _, result := range results {
		for k, v := range result {
			merged[k] = v
		}
	}
	
	log.Printf("Successfully processed all metric batches, total results: %d", len(merged))
	return merged, nil
}

// ProcessLabelBatches processes label synonym requests in parallel batches
func (bp *BatchProcessor) ProcessLabelBatches(
	ctx context.Context,
	labels []string,
	processFunc func(batch []string) (map[string][]string, error),
) (map[string][]string, error) {
	
	// Create batches
	batches := bp.createLabelBatches(labels)
	log.Printf("Created %d label batches (batch size: %d, total labels: %d)", 
		len(batches), bp.batchSize, len(labels))
	
	// Process batches in parallel
	log.Printf("Processing label batches with %d parallel workers", bp.maxWorkers)
	results, err := bp.processLabelBatchesInParallel(ctx, batches, processFunc)
	if err != nil {
		return nil, err
	}
	
	// Merge results
	merged := make(map[string][]string)
	for _, result := range results {
		for k, v := range result {
			merged[k] = v
		}
	}
	
	log.Printf("Successfully processed all label batches, total results: %d", len(merged))
	return merged, nil
}

// createMetricBatches splits the metric map into smaller batches
func (bp *BatchProcessor) createMetricBatches(metricMap map[string]string) []map[string]string {
	var batches []map[string]string
	currentBatch := make(map[string]string)
	count := 0
	
	for metric, desc := range metricMap {
		currentBatch[metric] = desc
		count++
		
		if count >= bp.batchSize {
			batches = append(batches, currentBatch)
			currentBatch = make(map[string]string)
			count = 0
		}
	}
	
	// Add remaining items
	if len(currentBatch) > 0 {
		batches = append(batches, currentBatch)
	}
	
	return batches
}

// createLabelBatches splits the label slice into smaller batches
func (bp *BatchProcessor) createLabelBatches(labels []string) [][]string {
	var batches [][]string
	
	for i := 0; i < len(labels); i += bp.batchSize {
		end := i + bp.batchSize
		if end > len(labels) {
			end = len(labels)
		}
		batches = append(batches, labels[i:end])
	}
	
	return batches
}

// processBatchesInParallel processes metric batches concurrently
func (bp *BatchProcessor) processBatchesInParallel(
	ctx context.Context,
	batches []map[string]string,
	processFunc func(batch map[string]string) (map[string][]string, error),
) ([]map[string][]string, error) {
	
	// Create channels for work distribution
	workCh := make(chan map[string]string, len(batches))
	resultCh := make(chan map[string][]string, len(batches))
	errorCh := make(chan error, len(batches))
	
	// Start workers
	var wg sync.WaitGroup
	workerCount := bp.maxWorkers
	if len(batches) < workerCount {
		workerCount = len(batches)
	}
	
	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range workCh {
				select {
				case <-ctx.Done():
					errorCh <- ctx.Err()
					return
				default:
					result, err := processFunc(batch)
					if err != nil {
						errorCh <- fmt.Errorf("batch processing failed: %w", err)
						return
					}
					resultCh <- result
				}
			}
		}()
	}
	
	// Send work to workers
	go func() {
		for _, batch := range batches {
			select {
			case <-ctx.Done():
				close(workCh)
				return
			case workCh <- batch:
			}
		}
		close(workCh)
	}()
	
	// Wait for all workers to complete
	wg.Wait()
	close(resultCh)
	close(errorCh)
	
	// Collect errors and results
	var firstError error
	var results []map[string][]string
	
	// Collect all results first
	for result := range resultCh {
		results = append(results, result)
	}
	
	// Then check for errors (non-blocking read from closed channel)
	select {
	case err := <-errorCh:
		if err != nil {
			firstError = err
		}
	default:
		// No errors
	}
	
	// Return error if any
	if firstError != nil {
		return nil, firstError
	}
	
	return results, nil
}

// processLabelBatchesInParallel processes label batches concurrently
func (bp *BatchProcessor) processLabelBatchesInParallel(
	ctx context.Context,
	batches [][]string,
	processFunc func(batch []string) (map[string][]string, error),
) ([]map[string][]string, error) {
	
	// Create channels for work distribution
	workCh := make(chan []string, len(batches))
	resultCh := make(chan map[string][]string, len(batches))
	errorCh := make(chan error, len(batches))
	
	// Start workers
	var wg sync.WaitGroup
	workerCount := bp.maxWorkers
	if len(batches) < workerCount {
		workerCount = len(batches)
	}
	
	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range workCh {
				select {
				case <-ctx.Done():
					errorCh <- ctx.Err()
					return
				default:
					result, err := processFunc(batch)
					if err != nil {
						errorCh <- fmt.Errorf("batch processing failed: %w", err)
						return
					}
					resultCh <- result
				}
			}
		}()
	}
	
	// Send work to workers
	go func() {
		for _, batch := range batches {
			select {
			case <-ctx.Done():
				close(workCh)
				return
			case workCh <- batch:
			}
		}
		close(workCh)
	}()
	
	// Wait for all workers to complete
	wg.Wait()
	close(resultCh)
	close(errorCh)
	
	// Collect errors and results
	var firstError error
	var results []map[string][]string
	
	// Collect all results first
	for result := range resultCh {
		results = append(results, result)
	}
	
	// Then check for errors (non-blocking read from closed channel)
	select {
	case err := <-errorCh:
		if err != nil {
			firstError = err
		}
	default:
		// No errors
	}
	
	// Return error if any
	if firstError != nil {
		return nil, firstError
	}
	
	return results, nil
}