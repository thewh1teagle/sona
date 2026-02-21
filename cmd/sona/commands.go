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
	var translate, detectLanguage bool
	var enhanceAudio, wordTimestamps bool
	var threads, maxTextCtx, maxSegmentLen, bestOf, beamSize, gpuDevice int
	var temperature float32

	cmd := &cobra.Command{
		Use:   "transcribe <model.bin> <audio.wav>",
		Short: "Transcribe an audio file",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			modelPath := args[0]
			audioPath := args[1]
			audio.SetVerbose(a.verbose)
			whisper.SetVerbose(a.verbose)

			samples, err := audio.ReadFileWithOptions(audioPath, audio.ReadOptions{
				EnhanceAudio: enhanceAudio,
			})
			if err != nil {
				return fmt.Errorf("error reading audio: %w", err)
			}

			ctx, err := whisper.New(modelPath, gpuDevice, false)
			if err != nil {
				return fmt.Errorf("error loading model: %w", err)
			}
			defer ctx.Close()

			result, err := ctx.Transcribe(samples, whisper.TranscribeOptions{
				Language:       language,
				DetectLanguage: detectLanguage,
				Translate:      translate,
				Threads:        threads,
				Prompt:         prompt,
				Verbose:        a.verbose,
				Temperature:    temperature,
				MaxTextCtx:     maxTextCtx,
				WordTimestamps: wordTimestamps,
				MaxSegmentLen:  maxSegmentLen,
				BestOf:         bestOf,
				BeamSize:       beamSize,
			})
			if err != nil {
				return fmt.Errorf("error transcribing: %w", err)
			}
			fmt.Println(result.Text())
			return nil
		},
	}

	cmd.Flags().StringVarP(&language, "language", "l", "", "language code (e.g. en, he); empty uses whisper.cpp default (en)")
	cmd.Flags().BoolVar(&detectLanguage, "detect-language", false, "auto-detect language")
	cmd.Flags().BoolVar(&enhanceAudio, "enhance-audio", false, "clean audio with ffmpeg before transcription (slower, can reduce repeats)")
	cmd.Flags().BoolVar(&translate, "translate", false, "translate to English")
	cmd.Flags().IntVar(&threads, "threads", 0, "CPU threads (0 = default)")
	cmd.Flags().StringVar(&prompt, "prompt", "", "initial prompt / vocabulary hint")
	cmd.Flags().Float32Var(&temperature, "temperature", 0, "initial decoding temperature (0 = default)")
	cmd.Flags().IntVar(&maxTextCtx, "max-text-ctx", 0, "max tokens from past text as context (0 = default)")
	cmd.Flags().BoolVar(&wordTimestamps, "word-timestamps", false, "enable token-level timestamps")
	cmd.Flags().IntVar(&maxSegmentLen, "max-segment-len", 0, "max segment length in characters (0 = no limit)")
	cmd.Flags().IntVar(&bestOf, "best-of", 0, "greedy sampling: top candidates (0 = default)")
	cmd.Flags().IntVar(&beamSize, "beam-size", 0, "beam search: beam width (0 = default)")
	cmd.Flags().IntVar(&gpuDevice, "gpu-device", -1, "GPU device index (-1 = whisper default)")
	return cmd
}

func (a *app) newServeCommand() *cobra.Command {
	var host string
	var port int

	cmd := &cobra.Command{
		Use:   "serve [model.bin]",
		Short: "Start a transcription runner",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			audio.SetVerbose(a.verbose)
			whisper.SetVerbose(a.verbose)

			s := server.New(a.verbose)
			s.Version = version
			s.Commit = commit

			// Load initial model if provided.
			if len(args) > 0 {
				if err := s.LoadModel(args[0], -1, false); err != nil {
					return fmt.Errorf("error loading model: %w", err)
				}
			}

			return server.ListenAndServe(host, port, s)
		},
	}

	cmd.Flags().StringVar(&host, "host", "127.0.0.1", "host to bind to")
	cmd.Flags().IntVarP(&port, "port", "p", 0, "port to listen on (0 = auto-assign)")
	return cmd
}
