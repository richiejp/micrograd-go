package main

import (
	"fmt"

	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"github.com/richiejp/micrograd/internal/data"
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

// The NN trained in the video
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

	for t := range 1000 {
		prnt := t%100 == 0
		if prnt {
			fmt.Printf("Predictions: ")
		}
		for i, x := range xs {
			ypred[i] = n.Forward(x)[0]
			if prnt {
				if i < len(xs)-1 {
					fmt.Printf("%v, ", ypred[i])
				} else {
					fmt.Printf("%v\n", ypred[i])
				}
			}
		}

		loss := gc.Val(0)
		for i, ygt := range ys {
			yout := ypred[i]
			l := yout.Sub(ygt).Pow(2)
			loss = loss.Add(l, gc.WithLabel(fmt.Sprintf("loss%d", i)))
		}

		if prnt {
			fmt.Printf("Loss: %v\n", loss)
		}

		if loss.Data() < 0.00001 {
			fmt.Printf("Loss < 0.00001; stopping early")
			break
		}

		gc.Backward(loss)

		ps := n.Parameters()

		if prnt {
			fmt.Printf("Parameters:\n")
		}
		for x := range ps {
			d := x.Descend(-0.05)
			if prnt {
				fmt.Printf("\t%v", x)
				fmt.Printf(" -> %v\n", d)
			}
		}
	}

	if err := viz.Render("./out/mlp.gv", ypred[0]); err != nil {
		panic(err)
	}
}

// Similar to demo.ipynb in the repo
func demo() {
	X, y := data.MakeMoons(100, 0.1, true)

	for i, yi := range y {
		y[i] = 2*yi - 1
	}

	var class0, class1 plotter.XYs
	for i := range X {
		point := plotter.XY{X: X[i][0], Y: X[i][1]}
		if y[i] == 0 {
			class0 = append(class0, point)
		} else {
			class1 = append(class1, point)
		}
	}

	p := plot.New()
	p.Title.Text = "makeMoons Dataset"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	s0, _ := plotter.NewScatter(class0)
	s1, _ := plotter.NewScatter(class1)
	s0.GlyphStyle.Color = color.RGBA{R: 255, A: 255}
	s1.GlyphStyle.Color = color.RGBA{B: 255, A: 255}

	p.Add(s0, s1)
	p.Legend.Add("Class 0", s0)
	p.Legend.Add("Class 1", s1)

	if err := p.Save(6*vg.Inch, 6*vg.Inch, "./out/moons.png"); err != nil {
		panic(err)
	}

	gc := &grad.Context[float64]{}

	model := gc.MLP(2, 16, 16, 1)

	// The activation functions differ between the video and the repository
	for l := range model.Depth() {
		model.SetActFn(l, grad.ReluActFn)
	}
	model.SetActFn(model.Depth() - 1, grad.LinearActFn)

	inputs := make([][]*grad.Value[float64], len(X))
	for i, xrow := range X {
		inputs[i] = gc.Vals(xrow...)
	}

	expected := make([]*grad.Value[float64], len(y))
	for i, yi := range y {
		expected[i] = gc.Val(float64(yi))
	}

	zero := gc.Val(0)
	one := gc.Val(1)
	losses_len := gc.Val(float64(len(inputs)))
	alpha := gc.Val(1e-4)

	if err := viz.Render("./out/demo.gv", model.Forward(inputs[0])[0]); err != nil {
		panic(err)
	}

	for k := range 100 {
		scores := make([]*grad.Value[float64], len(inputs))
		for i, input := range inputs {
			scores[i] = model.Forward(input)[0]
		}

		// SVM "max-margin" loss
		losses := make([]*grad.Value[float64], len(scores))
		for i, yi := range scores {
			losses[i] = one.Sub(expected[i].Mul(yi)).Relu(gc.WithLabel("loss")) 
		}
		data_loss := gc.Sum(losses).Div(losses_len)

		// L2 regularization
		square_params := zero
		for p := range model.Parameters() {
			square_params = square_params.Add(p.Mul(p))
		}
		reg_loss := square_params.Mul(alpha)
		total_loss := data_loss.Add(reg_loss)

		correct := 0
		for i, yi := range y {
			if (yi > 0) == (scores[i].Data() > 0) {
				correct += 1
			}
		}
		accuracy := float64(correct) / float64(len(y))
		
		gc.Backward(total_loss)

		learning_rate := 0.5 // 1.0 - 0.9*float64(k)/100.0
		for p := range model.Parameters() {
			p.Descend(-learning_rate)
		}

		fmt.Printf("Step %v loss %v, accuracy %.1f%%\n", k, total_loss.Data(), 100*accuracy)
	}
}

func main() {
	simpleGraph()
	trainNet()
	demo()
}
