package main

import (
	"github.com/richiejp/micrograd/internal/grad"
	"github.com/richiejp/micrograd/internal/viz"
)

func main() {
	gc := &grad.Context[float64]{}

	x1 := gc.NewValue(2, gc.WithLabel("x1"))
	x2 := gc.NewValue(0, gc.WithLabel("x2"))

	w1 := gc.NewValue(-3.0, gc.WithLabel("w1"))
	w2 := gc.NewValue(1.0, gc.WithLabel("w2"))

	b := gc.NewValue(6.881373587019543, gc.WithLabel("b"))

	x1w1 := x1.Mul(&w1, gc.WithLabel("x1*w1"))
	x2w2 := x2.Mul(&w2, gc.WithLabel("x2*w2"))
	x1w1x2w2 := x1w1.Add(&x2w2, gc.WithLabel("x1w1*x2w2"))
	n := x1w1x2w2.Add(&b, gc.WithLabel("n"))
	o := n.Tanh(gc.WithLabel("o"))
	gc.Backward(&o)
	
	if err := viz.Render(&o); err != nil {
		panic(err)
	}
}
