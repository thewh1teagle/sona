package main

import (
	"os"
	"time"
)

func watchParent() {
	ppid := os.Getppid()
	for range time.Tick(time.Second) {
		if os.Getppid() != ppid {
			os.Exit(0)
		}
	}
}
