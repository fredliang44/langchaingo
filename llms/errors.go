package llms

type LLMError struct {
	Message      string `json:"message,omitempty"`
	ErrorMessage string `json:"error_message,omitempty"`
	StatusCode   int    `json:"status_code,omitempty"`
	ErrorType    string `json:"error_code,omitempty"`
	RawResponse  []byte `json:"raw_response,omitempty"`
}

func (e *LLMError) Error() string {
	return e.Message
}
