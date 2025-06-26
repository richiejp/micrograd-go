package main_test

import (
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/richiejp/micrograd/internal/grad"
)

var _ = Describe("Main", func() {
	var gc *grad.Context[float64]
	BeforeEach(func() {
		gc = &grad.Context[float64]{}
	})

	It("Init Value", func(){
		v := gc.NewValue(2.1)

		Expect(fmt.Sprintf("%v", v)).To(ContainSubstring("2.1"))
	})

	It("Add Values", func(){
		a := gc.NewValue(0.1)
		b := gc.NewValue(0.2)
		c := a.Add(&b)

		Expect(c.Data()).To(BeNumerically("~", 0.3))
	})

	It("Mul Values", func(){
		a := gc.NewValue(0.1)
		b := gc.NewValue(0.2)
		c := a.Mul(&b)

		Expect(c.Data()).To(BeNumerically("~", 0.02))
	})

	It("Has previous Values", func(){
		a := gc.NewValue(0.1)
		b := gc.NewValue(0.2)
		c := a.Mul(&b)
		d := c.Add(&a)

		Expect(d.Prev()).To(Equal([]*grad.Value[float64]{&c, &a}))
		Expect(c.Prev()).To(Equal([]*grad.Value[float64]{&a, &b}))
	})

	It("Has Ops", func(){
		a := gc.NewValue(0.1)
		b := gc.NewValue(0.2)
		c := a.Mul(&b)
		d := c.Add(&a)

		Expect(a.Op()).To(Equal(grad.OpNil))
		Expect(c.Op()).To(Equal(grad.OpMul))
		Expect(d.Op()).To(Equal(grad.OpAdd))
	})

	It("Can label values", func(){
		a := gc.NewValue(0.1, gc.WithLabel("a"))

		Expect(a.Label()).To(Equal("a"))
	})

	It("Can backpropogate", func(){
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

		Expect(x1w1.Grad()).To(BeNumerically("~", 0.5))
		Expect(x2w2.Grad()).To(BeNumerically("~", 0.5))

		Expect(w1.Grad()).To(BeNumerically("~", 1.0))
		Expect(x1.Grad()).To(BeNumerically("~", -1.5))
		Expect(w2.Grad()).To(BeNumerically("~", 0.0))
		Expect(x2.Grad()).To(BeNumerically("~", 0.5))
	})

	It("Can backpropogate to variables with multiple parents", func(){
		gc := &grad.Context[float64]{}

		a := gc.NewValue(-2) // , -6 + 3 = -3
		b := gc.NewValue(3) //  , -6 - 2 = -8
		d := a.Mul(&b) // -6, 1
		e := a.Add(&b) // 1, -6
		f := d.Mul(&e) // -6, 1
		gc.Backward(&f)

		Expect(a.Grad()).To(BeNumerically("~", -3))
		Expect(b.Grad()).To(BeNumerically("~", -8))
	})
})


