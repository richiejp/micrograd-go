package grad

import (
	"fmt"
	"math"
	"sync/atomic"

	"golang.org/x/exp/constraints"
	"golang.org/x/exp/slices"
)

type Op string

const (
	OpNil  Op = ""
	OpAdd  Op = "+"
	OpMul  Op = "*"
	OpTanh Op = "tanh"
	OpRelu Op = "relu"
	OpExp  Op = "exp"
	OpPow  Op = "pow"
	OpDiv  Op = "/"
)

type Value[T constraints.Float] struct {
	ctx      *Context[T]
	data     T
	grad     T
	prev     []*Value[T]
	op       Op
	label    string
	id       uint64
	param    T
}

type ValueArg[T constraints.Float] func(args *ValueArgs[T])

type ValueArgs[T constraints.Float] struct {
	prev     []*Value[T]
	op       Op
	label    string
	grad     T
	param    T
}

type Context[T constraints.Float] struct {
	maxId atomic.Uint64
	topoSorted []*Value[T]
}

func (c *Context[T]) WithPrev(children ...*Value[T]) ValueArg[T] {
	return func(args *ValueArgs[T]) {
		args.prev = children
	}
}

func (c *Context[T]) WithOp(op Op) ValueArg[T] {
	return func(args *ValueArgs[T]) {
		args.op = op
	}
}

func (c *Context[T]) WithLabel(label string) ValueArg[T] {
	return func(args *ValueArgs[T]) {
		args.label = label
	}
}

func (c *Context[T]) WithGrad(grad T) ValueArg[T] {
	return func(args *ValueArgs[T]) {
		args.grad = grad
	}
}

func (c *Context[T]) WithParam(param T) ValueArg[T] {
	return func(args *ValueArgs[T]) {
		args.param = param
	}
}

func (c *Context[T]) Val(d T, withArgs ...ValueArg[T]) *Value[T] {
	var args ValueArgs[T]

	for _, fn := range withArgs {
		fn(&args)
	}

	return &Value[T]{
		ctx:      c,
		data:     d,
		grad:     args.grad,
		prev:     args.prev,
		op:       args.op,
		label:    args.label,
		id:       c.maxId.Add(1),
		param:    args.param,
	}
}

func (c *Context[T]) Vals(ds ...T) []*Value[T] {
	vs := make([]*Value[T], len(ds))

	for i, d := range ds {
		vs[i] = c.Val(d)
	}

	return vs
}

func (c *Context[T]) topoSort(root *Value[T]) []*Value[T] {
	visited := make(map[uint64]struct{})
	var topoSorted []*Value[T]

	var build func (v *Value[T])
	build = func (v *Value[T]) {
		if  _, ok := visited[v.ID()]; ok {
			return
		}

		visited[v.ID()] = struct{}{}
		for _, child := range v.Prev() {
			build(child)
		}
		topoSorted = append(topoSorted, v)
	}
	build(root)
	slices.Reverse(topoSorted)

	return topoSorted 
}

func (c *Context[T]) Backward(root *Value[T]) {
	if len(c.topoSorted) < 1 || c.topoSorted[0].id != root.id {
		c.topoSorted = c.topoSort(root)
	}

	for _, v := range c.topoSorted {
		v.grad = 0
	}

	root.grad = 1

	for _, v := range c.topoSorted {
		v.backward()
	}
}

func (c *Context[T]) Sum(vs []*Value[T]) *Value[T] {
	sum := vs[0]
	for _, l := range vs[1:] {
		sum = sum.Add(l)
	}

	return sum
}

func (v Value[T]) String() string {
	return fmt.Sprintf("Value(data=%v, grad=%v)", v.data, v.grad)
}

func (v *Value[T]) backward() {
	switch v.op {
	case OpNil:
		if len(v.prev) > 0 {
			panic("Constant has children")
		}
	case OpAdd:
		for _, c := range v.prev {
			c.grad += v.grad
		}
	case OpMul:
		a := v.prev[0]
		b := v.prev[1]

		a.grad += v.grad * b.data
		b.grad += v.grad * a.data
	case OpPow:
		a := v.prev[0]

		a.grad += v.grad * v.param * T(math.Pow(float64(a.data), float64(v.param - 1)))
	case OpTanh:
		a := v.prev[0]

		a.grad += (1 - v.data*v.data) * v.grad
	case OpRelu:
		a := v.prev[0]

		if v.data > 0 {
			a.grad += v.grad
		}
	case OpExp:
		a := v.prev[0]

		a.grad += v.data * v.grad	
	}
}

func (v *Value[T]) Add(o *Value[T], withArgs ...ValueArg[T]) *Value[T] {
	c := v.ctx
	args := []ValueArg[T]{c.WithPrev(v, o), c.WithOp(OpAdd)}
	args = append(args, withArgs...)

	return c.Val(v.data+o.data, args...)
}

func (v *Value[T]) Mul(o *Value[T], withArgs ...ValueArg[T]) *Value[T] {
	c := v.ctx
	args := []ValueArg[T]{c.WithPrev(v, o), c.WithOp(OpMul)}
	args = append(args, withArgs...)

	return c.Val(v.data*o.data, args...)
}

func (v *Value[T]) Pow(n T, withArgs ...ValueArg[T]) *Value[T] {
	c := v.ctx
	args := []ValueArg[T]{c.WithPrev(v), c.WithOp(OpPow), c.WithParam(n)}
	args = append(args, withArgs...)

	return c.Val(T(math.Pow(float64(v.data), float64(n))), args...)
}

func (v *Value[T]) Div(o *Value[T], withArgs ...ValueArg[T]) *Value[T] {
	frac := o.Pow(-1)

	return v.Mul(frac, withArgs...)
}

func (v *Value[T]) Sub(o *Value[T], withArgs ...ValueArg[T]) *Value[T] {
	inv := v.ctx.Val(-1)
  neg := o.Mul(inv)

	return v.Add(neg, withArgs...)
}

func (v *Value[T]) Tanh(withArgs ...ValueArg[T]) *Value[T] {
	c := v.ctx
	args := []ValueArg[T]{c.WithPrev(v), c.WithOp(OpTanh)}
	args = append(args, withArgs...)

	return c.Val(T(math.Tanh(float64(v.data))), args...)
}

func (v *Value[T]) Relu(withArgs ...ValueArg[T]) *Value[T] {
	c := v.ctx
	args := []ValueArg[T]{c.WithPrev(v), c.WithOp(OpRelu)}
	args = append(args, withArgs...)
	var x T

	if v.data > 0 {
		x = v.data
	}

	return c.Val(x, args...)
}

func (v *Value[T]) Exp(withArgs ...ValueArg[T]) *Value[T] {
	c := v.ctx
	args := []ValueArg[T]{c.WithPrev(v), c.WithOp(OpExp)}
	args = append(args, withArgs...)

	return c.Val(T(math.Exp(float64(v.data))), args...)
}

func (v Value[T]) Data() T {
	return v.data
}

func (v Value[T]) Grad() T {
	return v.grad
}

func (v Value[T]) Prev() []*Value[T] {
	return v.prev
}

func (v Value[T]) Op() Op {
	return v.op
}

func (v Value[T]) Label() string {
	return v.label
}

func (v Value[T]) ID() uint64 {
	return v.id
}

 func (v *Value[T]) Descend(update T) T {
	v.data += update * v.grad

	return v.data
}
