package main

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/thewh1teagle/sona/internal/audio"
	"github.com/thewh1teagle/sona/internal/server"
	"github.com/thewh1teagle/sona/internal/whisper"
)

type app struct {
	verbose bool
}

func newRootCommand() *cobra.Command {
	a := &app{}
	rootCmd := &cobra.Command{
		Use:     "sona",
		Short:   "Speech-to-text powered by whisper.cpp",
		Version: version,
	}
	rootCmd.PersistentFlags().BoolVarP(&a.verbose, "verbose", "v", false, "show ffmpeg and whisper/ggml logs")
	rootCmd.AddCommand(a.newTranscribeCommand(), a.newServeCommand(), newPullCommand())
	return rootCmd
}

func (a *app) newTranscribeCommand() *cobra.Command {
	var language, prompt string
	var translate bool
	var threads int

	cmd := &cobra.Command{
		Use:   "transcribe <model.bin> <audio.wav>",
		Short: "Transcribe an audio file",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			modelPath := args[0]
			audioPath := args[1]
			audio.SetVerbose(a.verbose)
			whisper.SetVerbose(a.verbose)

			samples, err := audio.ReadFile(audioPath)
			if err != nil {
				return fmt.Errorf("error reading audio: %w", err)
			}

			ctx, err := whisper.New(modelPath)
			if err != nil {
				return fmt.Errorf("error loading model: %w", err)
			}
			defer ctx.Close()

			text, err := ctx.Transcribe(samples, whisper.TranscribeOptions{
				Language:  language,
				Translate: translate,
				Threads:   threads,
				Prompt:    prompt,
				Verbose:   a.verbose,
			})
			if err != nil {
				return fmt.Errorf("error transcribing: %w", err)
			}
			fmt.Println(text)
			return nil
		},
	}

	cmd.Flags().StringVarP(&language, "language", "l", "", "language code (e.g. en, he, auto)")
	cmd.Flags().BoolVar(&translate, "translate", false, "translate to English")
	cmd.Flags().IntVar(&threads, "threads", 0, "CPU threads (0 = default)")
	cmd.Flags().StringVar(&prompt, "prompt", "", "initial prompt / vocabulary hint")
	return cmd
}

func (a *app) newServeCommand() *cobra.Command {
	var port int

	cmd := &cobra.Command{
		Use:   "serve <model.bin>",
		Short: "Start an OpenAI-compatible transcription server",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			modelPath := args[0]
			audio.SetVerbose(a.verbose)
			whisper.SetVerbose(a.verbose)

			ctx, err := whisper.New(modelPath)
			if err != nil {
				return fmt.Errorf("error loading model: %w", err)
			}
			defer ctx.Close()

			addr := fmt.Sprintf(":%d", port)
			return server.ListenAndServe(addr, server.New(ctx, modelPath, a.verbose))
		},
	}

	cmd.Flags().IntVarP(&port, "port", "p", 11531, "port to listen on")
	return cmd
}
