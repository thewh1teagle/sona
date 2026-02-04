package main

import (
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

func newPullCommand() *cobra.Command {
	var outputPath string

	cmd := &cobra.Command{
		Use:   "pull <url>",
		Short: "Download a model file",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return pullFile(args[0], outputPath)
		},
	}
	cmd.Flags().StringVarP(&outputPath, "output", "o", "", "output file path or directory")
	return cmd
}

func pullFile(rawURL, outputPath string) error {
	req, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil {
		return fmt.Errorf("invalid url: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}

	filename := resolveFilename(resp, rawURL)
	dst := resolveOutputPath(outputPath, filename)
	if dir := filepath.Dir(dst); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create output dir: %w", err)
		}
	}

	tmp := dst + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}

	total := resp.ContentLength
	var written int64
	start := time.Now()
	lastPrint := time.Time{}
	buf := make([]byte, 64*1024)

	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := out.Write(buf[:n]); err != nil {
				out.Close()
				return fmt.Errorf("write output file: %w", err)
			}
			written += int64(n)
			if time.Since(lastPrint) >= 200*time.Millisecond {
				printProgress(written, total, start)
				lastPrint = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			out.Close()
			return fmt.Errorf("download interrupted: %w", readErr)
		}
	}

	if err := out.Close(); err != nil {
		return fmt.Errorf("close output file: %w", err)
	}
	if err := os.Rename(tmp, dst); err != nil {
		return fmt.Errorf("finalize output file: %w", err)
	}

	printProgress(written, total, start)
	fmt.Printf("\nsaved %s\n", dst)
	return nil
}

func resolveFilename(resp *http.Response, rawURL string) string {
	if cd := resp.Header.Get("Content-Disposition"); cd != "" {
		if _, params, err := mime.ParseMediaType(cd); err == nil {
			if name := strings.TrimSpace(params["filename"]); name != "" {
				return filepath.Base(name)
			}
			if star := strings.TrimSpace(params["filename*"]); star != "" {
				if parts := strings.SplitN(star, "''", 2); len(parts) == 2 {
					if decoded, err := url.QueryUnescape(parts[1]); err == nil && decoded != "" {
						return filepath.Base(decoded)
					}
				}
			}
		}
	}

	if resp.Request != nil && resp.Request.URL != nil {
		if base := path.Base(resp.Request.URL.Path); base != "." && base != "/" && base != "" {
			return base
		}
	}

	if parsed, err := url.Parse(rawURL); err == nil {
		if base := path.Base(parsed.Path); base != "." && base != "/" && base != "" {
			return base
		}
	}

	return "model.bin"
}

func resolveOutputPath(outputPath, filename string) string {
	if outputPath == "" || outputPath == "." {
		return filename
	}
	if info, err := os.Stat(outputPath); err == nil && info.IsDir() {
		return filepath.Join(outputPath, filename)
	}
	if strings.HasSuffix(outputPath, string(os.PathSeparator)) || strings.HasSuffix(outputPath, "/") || strings.HasSuffix(outputPath, "\\") {
		return filepath.Join(outputPath, filename)
	}
	return outputPath
}

func printProgress(written, total int64, start time.Time) {
	mb := float64(written) / (1024 * 1024)
	seconds := time.Since(start).Seconds()
	if seconds < 0.001 {
		seconds = 0.001
	}
	speed := mb / seconds

	if total > 0 {
		pct := float64(written) * 100 / float64(total)
		remainingMB := float64(total-written) / (1024 * 1024)
		etaSeconds := 0.0
		if speed > 0 {
			etaSeconds = remainingMB / speed
		}
		fmt.Printf("\rdownloading %.1f%% (%.1f/%.1f MB) %.2f MB/s eta %.0fs", pct, mb, float64(total)/(1024*1024), speed, etaSeconds)
		return
	}
	fmt.Printf("\rdownloading %.1f MB %.2f MB/s", mb, speed)
}
