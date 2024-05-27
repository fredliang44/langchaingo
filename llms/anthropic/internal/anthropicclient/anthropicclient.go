package anthropicclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmc/langchaingo/llms"
	"io"
	"net/http"
	"strings"
)

const (
	DefaultBaseURL          = "https://api.anthropic.com/v1"
	DefaultAnthropicVersion = "2023-06-01"
	defaultModel            = "claude-1.3"
)

// ErrEmptyResponse is returned when the Anthropic API returns an empty response.
var ErrEmptyResponse = errors.New("empty response")

// Client is a client for the Anthropic API.
type Client struct {
	token   string
	Model   string
	baseURL string

	vertexProjectID  string
	vertexLocation   string
	httpClient       Doer
	anthropicVersion string

	// UseLegacyTextCompletionsAPI is a flag to use the legacy text completions API.
	UseLegacyTextCompletionsAPI bool
}

// Option is an option for the Anthropic client.
type Option func(*Client) error

// Doer performs a HTTP request.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// WithVertexProjectID sets the Vertex project ID.
func WithVertexProjectID(projectID string) Option {
	return func(c *Client) error {
		c.vertexProjectID = projectID
		return nil
	}
}

// WithVertexLocation sets the Vertex AI location.
func WithVertexLocation(location string) Option {
	return func(c *Client) error {
		c.vertexLocation = location
		return nil
	}
}

// WithAnthropicVersion sets the Anthropic version.
func WithAnthropicVersion(version string) Option {
	return func(c *Client) error {
		c.anthropicVersion = version
		return nil
	}
}

// WithHTTPClient allows setting a custom HTTP client.
func WithHTTPClient(client Doer) Option {
	return func(c *Client) error {
		c.httpClient = client

		return nil
	}
}

// WithLegacyTextCompletionsAPI enables the use of the legacy text completions API.
func WithLegacyTextCompletionsAPI(val bool) Option {
	return func(opts *Client) error {
		opts.UseLegacyTextCompletionsAPI = val
		return nil
	}
}

// New returns a new Anthropic client.
func New(token string, model string, baseURL string, opts ...Option) (*Client, error) {
	c := &Client{
		Model:   model,
		token:   token,
		baseURL: strings.TrimSuffix(baseURL, "/"),
	}

	for _, opt := range opts {
		if err := opt(c); err != nil {
			return nil, err
		}
	}

	return c, nil
}

// CompletionRequest is a request to create a completion.
type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Temperature float64  `json:"temperature"`
	MaxTokens   int      `json:"max_tokens_to_sample,omitempty"`
	StopWords   []string `json:"stop_sequences,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	Stream      bool     `json:"stream,omitempty"`

	// StreamingFunc is a function to be called for each chunk of a streaming response.
	// Return an error to stop streaming early.
	StreamingFunc func(ctx context.Context, chunk []byte) error `json:"-"`
}

// Completion is a completion.
type Completion struct {
	Text string `json:"text"`
}

// CreateCompletion creates a completion.
func (c *Client) CreateCompletion(ctx context.Context, r *CompletionRequest) (*Completion, error) {
	resp, err := c.createCompletion(ctx, &completionPayload{
		Model:         r.Model,
		Prompt:        r.Prompt,
		Temperature:   r.Temperature,
		MaxTokens:     r.MaxTokens,
		StopWords:     r.StopWords,
		TopP:          r.TopP,
		Stream:        r.Stream,
		StreamingFunc: r.StreamingFunc,
	})
	if err != nil {
		return nil, err
	}
	return &Completion{
		Text: resp.Completion,
	}, nil
}

type MessageRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	System      string        `json:"system,omitempty"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	TopK        int           `json:"top_k,omitempty"`
	Tools       []llms.Tool   `json:"tools,omitempty"`

	// ToolChoice is the choice of tool to use, it can either be "none", "auto" (the default behavior), or a specific tool as described in the ToolChoice type.
	ToolChoice any      `json:"tool_choice,omitempty"`
	StopWords  []string `json:"stop_sequences,omitempty"`
	Stream     bool     `json:"stream,omitempty"`

	StreamingFunc func(ctx context.Context, chunk []byte) error `json:"-"`
}

func handleToolChoice(toolChoice any) (*ToolChoice, error) {
	switch toolChoice.(type) {
	case string:
		return &ToolChoice{
			Type: toolChoice.(string),
		}, nil
	case llms.Tool:
		if toolChoice.(llms.Tool).Function == nil {
			return nil, errors.New("tool choice function is nil")
		}
		return &ToolChoice{
			Type: toolChoice.(llms.Tool).Type,
			Name: toolChoice.(llms.Tool).Function.Name,
		}, nil
	default:
		return nil, nil
	}
}

func handleTools(tools []llms.Tool) ([]Tool, error) {
	var resultTools []Tool
	for _, tool := range tools {
		if tool.Function == nil {
			return nil, errors.New("tool choice function is nil")
		}
		resultTools = append(resultTools, Tool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
		})
	}
	return resultTools, nil
}

// CreateMessage creates message for the messages api.
func (c *Client) CreateMessage(ctx context.Context, r *MessageRequest) (*MessageResponsePayload, error) {
	toolChoice, err := handleToolChoice(r.ToolChoice)
	if err != nil {
		return nil, err
	}
	tools, err := handleTools(r.Tools)
	if err != nil {
		return nil, err
	}
	resp, err := c.createMessage(ctx, &messagePayload{
		Model:         r.Model,
		Messages:      r.Messages,
		System:        r.System,
		Temperature:   r.Temperature,
		MaxTokens:     r.MaxTokens,
		StopWords:     r.StopWords,
		TopP:          r.TopP,
		Stream:        r.Stream,
		StreamingFunc: r.StreamingFunc,
		TopK:          r.TopK,
		Tools:         tools,
		ToolChoice:    toolChoice,
	})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")

	if c.vertexProjectID != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	} else {
		req.Header.Set("x-api-key", c.token)
	}

	if c.anthropicVersion != "" {
		req.Header.Set("anthropic-version", c.anthropicVersion)
	} else {
		// adjust version based on the vertex project ID
		if c.vertexProjectID != "" {
			req.Header.Set("anthropic-version", "vertex-2023-10-16")
		} else {
			req.Header.Set("anthropic-version", "2023-06-01")
		}
	}

}

func (c *Client) do(ctx context.Context, path string, payloadBytes []byte) (*http.Response, error) {
	var url string

	if c.vertexProjectID == "" {
		if c.baseURL == "" {
			c.baseURL = DefaultBaseURL
		}

		url = c.baseURL + path
	} else {
		url = fmt.Sprintf("https://%s-aiplatform.googleapis."+
			"com/v1/projects/%s/locations/%s/publishers/anthropic/models/%s:streamRawPredict",
			c.vertexLocation, c.vertexProjectID, c.vertexLocation, c.Model)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	c.setHeaders(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	return resp, nil
}

type errorMessage struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func (c *Client) decodeError(resp *http.Response) error {
	msg := fmt.Sprintf("API returned unexpected status code: %d", resp.StatusCode)

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("%s: %w", msg, err)
	}

	var errResp errorMessage
	if err := json.Unmarshal(respBody, &errResp); err != nil {
		return errors.New(msg) // nolint:goerr113
	}

	// nolint:goerr113
	return &llms.LLMError{
		Message:      fmt.Sprintf("%s: %s", msg, errResp.Error.Message),
		StatusCode:   resp.StatusCode,
		ErrorType:    errResp.Error.Type,
		ErrorMessage: errResp.Error.Message,
		RawResponse:  respBody,
	}
}
