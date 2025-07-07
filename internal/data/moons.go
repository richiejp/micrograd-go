package data

import (
	"math"
	"math/rand"
	"time"
)

// Converted from scikit-learn with GitHub Co-pilot

// makeMoons generates a toy dataset of two interleaving half circles.
// nSamples: total number of points
// noise: standard deviation of Gaussian noise (set to 0.0 for no noise)
// shuffle: whether to shuffle the samples
func MakeMoons(nSamples int, noise float64, shuffle bool) ([][]float64, []int) {
	nSamplesOut := nSamples / 2
	nSamplesIn := nSamples - nSamplesOut

	X := make([][]float64, nSamples)
	y := make([]int, nSamples)

	// First half-circle
	for i := range nSamplesOut {
		theta := math.Pi * float64(i) / float64(nSamplesOut-1)
		x := math.Cos(theta)
		yval := math.Sin(theta)
		X[i] = []float64{x, yval}
		y[i] = 0
	}
	// Second half-circle
	for i := range nSamplesIn {
		theta := math.Pi * float64(i) / float64(nSamplesIn-1)
		x := 1.0 - math.Cos(theta)
		yval := 1.0 - math.Sin(theta) - 0.5
		X[nSamplesOut+i] = []float64{x, yval}
		y[nSamplesOut+i] = 1
	}

	// Add Gaussian noise if needed
	if noise > 0.0 {
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		for i := range nSamples {
			X[i][0] += noise * randNorm(rng)
			X[i][1] += noise * randNorm(rng)
		}
	}

	// Shuffle samples if needed
	if shuffle {
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		for i := nSamples - 1; i > 0; i-- {
			j := rng.Intn(i + 1)
			X[i], X[j] = X[j], X[i]
			y[i], y[j] = y[j], y[i]
		}
	}

	return X, y
}

// randNorm generates a standard normal random value using Box-Muller
func randNorm(rng *rand.Rand) float64 {
	u1 := rng.Float64()
	u2 := rng.Float64()
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}
