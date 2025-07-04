package grad

import (
	"fmt"
	"iter"

	"golang.org/x/exp/constraints"
)

type Neuron[T constraints.Float] struct {
	w []*Value[T]
	b *Value[T]
}

func (c *Context[T]) Neu(nin uint) *Neuron[T] {
	n := Neuron[T]{
		w: make([]*Value[T], nin),
		b: c.Val(0.5, c.WithLabel("b")),
	}

	for i := range n.w {
		n.w[i] = c.Val(0.5, c.WithLabel(fmt.Sprintf("w%d", i)))
	}

	return &n
}

func (n *Neuron[T]) Forward(inputs []*Value[T]) *Value[T] {
	act := n.b

	for i, xi := range inputs {
		act = act.Add(xi.Mul(n.w[i]))
	}

	return act.Tanh(act.ctx.WithLabel("out"))
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

	for i := range(mlp.layers) {
		mlp.layers[i] = c.Lay(sz[i], sz[i+1])
	}

	return &mlp
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
