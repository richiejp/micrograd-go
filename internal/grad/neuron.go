package grad

import (
	"fmt"
	"iter"
	"math/rand/v2"

	"golang.org/x/exp/constraints"
)

type ActFn string

const (
	LinearActFn ActFn = "linear"
	TanhActFn   ActFn = ActFn(OpTanh)
	ReluActFn   ActFn = ActFn(OpRelu)
)

type Neuron[T constraints.Float] struct {
	w     []*Value[T]
	b     *Value[T]
	actFn ActFn
}

func (c *Context[T]) Neu(nin uint) *Neuron[T] {
	n := Neuron[T]{
		w:     make([]*Value[T], nin),
		b:     c.Val(0, c.WithLabel("b")),
		actFn: TanhActFn,
	}

	// For Tanh activation and the smaller NN example a starting value of 0.5 allowed for successful training
	// For Relu and the moon fitting demo however it would get stuck
	for i := range n.w {
		n.w[i] = c.Val(T(rand.NormFloat64()), c.WithLabel(fmt.Sprintf("w%d", i)))
	}

	return &n
}

func (n *Neuron[T]) Forward(inputs []*Value[T]) *Value[T] {
	act := n.b

	for i, xi := range inputs {
		act = act.Add(xi.Mul(n.w[i]))
	}

	switch n.actFn {
	case LinearActFn:
		return act
	case TanhActFn:
		return act.Tanh(act.ctx.WithLabel("out"))
	case ReluActFn:
		return act.Relu(act.ctx.WithLabel("out"))
	}

	panic("Unhandled activation")
}

func (n *Neuron[T]) Parameters() iter.Seq[*Value[T]] {
	return func(yield func(*Value[T]) bool) {
		if !yield(n.b) {
			return
		}
		for _, v := range n.w {
			if !yield(v) {
				return
			}
		}
	}
}

type Layer[T constraints.Float] struct {
	neurons []*Neuron[T]
}

func (c *Context[T]) Lay(nin uint, nout uint) *Layer[T] {
	l := Layer[T]{
		neurons: make([]*Neuron[T], nout),
	}

	for i := range l.neurons {
		l.neurons[i] = c.Neu(nin)
	}

	return &l
}

func (l *Layer[T]) Forward(inputs []*Value[T]) []*Value[T] {
	outs := make([]*Value[T], len(l.neurons))

	for i, n := range l.neurons {
		outs[i] = n.Forward(inputs)
	}

	return outs
}

func (l *Layer[T]) Parameters() iter.Seq[*Value[T]] {
	return func(yield func(*Value[T]) bool) {
		for _, n := range l.neurons {
			for v := range n.Parameters() {
				if !yield(v) {
					return
				}
			}
		}
	}
}

type MLP[T constraints.Float] struct {
	layers []*Layer[T]
}

func (c *Context[T]) MLP(nin uint, nout uint, nouts ...uint) *MLP[T] {
	sz := []uint{nin, nout}
	sz = append(sz, nouts...)

	mlp := MLP[T]{
		layers: make([]*Layer[T], len(sz)-1),
	}

	for i := range mlp.layers {
		mlp.layers[i] = c.Lay(sz[i], sz[i+1])
	}

	return &mlp
}

func (mlp *MLP[T]) SetActFn(layer int, fn ActFn) {
	for _, n := range mlp.layers[layer].neurons {
		n.actFn = fn
	}
}

func (mlp *MLP[T]) Forward(inputs []*Value[T]) []*Value[T] {
	out := mlp.layers[0].Forward(inputs)

	for _, l := range mlp.layers[1:] {
		out = l.Forward(out)
	}

	return out
}

func (mlp *MLP[T]) Parameters() iter.Seq[*Value[T]] {
	return func(yield func(*Value[T]) bool) {
		for _, l := range mlp.layers {
			for v := range l.Parameters() {
				if !yield(v) {
					return
				}
			}
		}
	}
}

func (mlp *MLP[T]) Depth() int {
	return len(mlp.layers)
}
