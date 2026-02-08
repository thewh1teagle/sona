package server

import "strconv"

func parseBoolFormValue(v string) bool {
	b, err := strconv.ParseBool(v)
	return err == nil && b
}

func parseIntFormValue(v string) int {
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0
	}
	return n
}

func parseFloatFormValue(v string) float32 {
	f, err := strconv.ParseFloat(v, 32)
	if err != nil {
		return 0
	}
	return float32(f)
}
