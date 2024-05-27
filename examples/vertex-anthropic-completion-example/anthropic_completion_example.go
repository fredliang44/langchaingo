package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/tmc/langchaingo/llms"
    "github.com/tmc/langchaingo/llms/anthropic"
)

func main() {
    llm, err := anthropic.New(
        anthropic.WithModel("claude-3-haiku@20240307"),
        anthropic.WithToken(os.Getenv("ANTHROPIC_API_KEY")),
        anthropic.WithAnthropicVersion("vertex-2023-10-16"),
        anthropic.WithVertexProjectID(os.Getenv("ANTHROPIC_PROJECT_ID")),
        anthropic.WithVertexLocation("us-central1"),
    )
    if err != nil {
        log.Fatal(err)
    }
    ctx := context.Background()
    completion, err := llms.GenerateFromSinglePrompt(ctx, llm, "Hi claude, write a poem about golang powered AI systems",
        llms.WithTemperature(0.2),
        llms.WithStopWords([]string{"STOP"}),
        llms.WithMaxTokens(10),
        llms.WithTopP(0.9),
        llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
            fmt.Print(string(chunk))
            return nil
        }),
    )
    if err != nil {
        log.Fatal(err)
    }

    _ = completion
}
