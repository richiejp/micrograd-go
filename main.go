package main

import (
	"fmt"

	"github.com/richiejp/micrograd/internal/grad"
	"github.com/richiejp/micrograd/internal/viz"
)

func simpleGraph() {
	gc := &grad.Context[float64]{}

	x1 := gc.Val(2, gc.WithLabel("x1"))
	x2 := gc.Val(0, gc.WithLabel("x2"))

	w1 := gc.Val(-3.0, gc.WithLabel("w1"))
	w2 := gc.Val(1.0, gc.WithLabel("w2"))

	b := gc.Val(6.881373587019543, gc.WithLabel("b"))

	x1w1 := x1.Mul(w1, gc.WithLabel("x1*w1"))
	x2w2 := x2.Mul(w2, gc.WithLabel("x2*w2"))
	x1w1x2w2 := x1w1.Add(x2w2, gc.WithLabel("x1w1*x2w2"))
	n := x1w1x2w2.Add(b, gc.WithLabel("n"))
	e := n.Mul(gc.Val(2), gc.WithLabel("2n")).Exp(gc.WithLabel("e^2n"))
	o := e.Add(gc.Val(-1), gc.WithLabel("e-1")).Div(e.Add(gc.Val(1), gc.WithLabel("e+1")), gc.WithLabel("o"))
	gc.Backward(o)

	if err := viz.Render("./out/graph.gv", o); err != nil {
		panic(err)
	}

}

func trainNet() {
	gc := grad.Context[float64]{}

	fmt.Printf("Training...\n")

	n := gc.MLP(3, 4, 4, 1)
	xsf := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	xs := make([][]*grad.Value[float64], len(xsf))
	for i, row := range xsf {
		xs[i] = make([]*grad.Value[float64], len(row))
		for j, v := range row {
			xs[i][j] = gc.Val(v, gc.WithLabel(fmt.Sprintf("xs%d,%d", i, j)))
		}
	}

	ys := gc.Vals(1, -1, -1, 1)
	ypred := make([]*grad.Value[float64], len(ys))

	fmt.Printf("Predictions: ")
	for i, x := range xs {
		ypred[i] = n.Forward(x)[0]
		if i < len(xs)-1 {
			fmt.Printf("%v, ", ypred[i])
		} else {
			fmt.Printf("%v\n", ypred[i])
		}
	}

	if err := viz.Render("./out/mlp.gv", ypred[0]); err != nil {
		panic(err)
	}

	loss := gc.Val(0)
	for i, ygt := range ys {
		yout := ypred[i]
		l := yout.Sub(ygt).Pow(2)
		loss = loss.Add(l, gc.WithLabel(fmt.Sprintf("loss%d", i)))
	}

	fmt.Printf("Loss: %v\n", loss)

	if err := viz.Render("./out/loss.gv", loss); err != nil {
		panic(err)
	}
}

func main() {
	simpleGraph()
	trainNet()
}
