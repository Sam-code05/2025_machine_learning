# Written assignment
## 一、 Constructing Polynomials with Tanh Neural Networks

### Background and Motivation

The main goal of this paper is to prove that a seemingly simple `tanh` neural network with only two hidden layers is sufficient to efficiently approximate various complex functions.  
However, directly jumping from the `tanh` function to approximating any arbitrary complex function (e.g., $\sin(x)$ or even more complicated solutions) would be too large a leap.

The authors adopt a classic mathematical strategy: **reduce complexity step by step**.  
From calculus, Taylor expansion tells us that any sufficiently *smooth* function can be well approximated by a polynomial within a small neighborhood.  
If we can approximate polynomials, we can then lay the foundation for approximating broader classes of functions.  
For example, $\cos(x)$ near $x=0$ behaves very much like ($1 - \frac{x^2}{2}$).

Thus, the overall reasoning can be simplified into two steps:
1. Prove that `tanh` networks can precisely construct the most fundamental functional units: **polynomials**.  
2. Use these “polynomial building blocks” to approximate more complex functions.

Lemmas 3.1 and 3.2 are the key to achieving the first step.  
They demonstrate how, like LEGO blocks, we can use `tanh` as the basic component to build $y, y^2, y^3, \dots$, the foundation of all polynomials — monomials.

---

### Lemma 3.1: Constructing Odd-Powered Monomials ($y, y^3, y^5, \dots$)

**1. Statement**

The lemma states that we can design a simple `tanh` neural network (with only one hidden layer) that can **simultaneously** approximate a series of odd-powered monomials, such as $y, y^3, y^5, \dots, y^s$ (where $s$ is an odd number).

Here, “precisely” has a strong meaning. It not only means that the function values are close (e.g., the network output is approximately $y^3$), but also that their **derivatives of all orders** are close as well.  
This implies that the two functions’ graphs not only overlap in position, but also match in slope (first derivative), curvature (second derivative), and so on.  
We can make this error arbitrarily small, less than $\epsilon$.

**2. Core Idea**

The authors employ a clever tool from numerical analysis: the **central finite difference operator**, to derive $y^3$ from the `tanh` function.

Although the operator sounds complicated, the idea is straightforward. In calculus, we approximate derivatives using $\frac{f(x+h) - f(x)}{h}$.  
This operator follows the same principle: it takes the values of a function at different positions (such as $x+h, x, x-h$), applies weighted linear combinations, and isolates specific behaviors of the function.

The `tanh` function is an odd function ($\tanh(-x) = -\tanh(x)$), and its Taylor expansion at $x=0$ contains only odd-powered terms:

$$
\tanh(x) = x - \frac{1}{3}x^3 + \frac{2}{15}x^5 - \dots
$$

The designed finite difference operator acts like a “filter.”  
When applied to `tanh`, it cancels out all the unwanted terms in the Taylor expansion, leaving only the desired odd-powered term (e.g., $x^p$) as the main part.

The remaining terms form the approximation error, which can be made arbitrarily small by adjusting the step size $h$ in the difference operator.  
Finally, the mathematical form of this operator can itself be implemented by a shallow `tanh` network, thereby completing the approximation of odd-powered monomials.

---

### Lemma 3.2: Extending to Even-Powered Monomials ($y^2, y^4, y^6, \dots$)

**1. Statement**

Lemma 3.2 is essentially an upgraded version of Lemma 3.1.  
It shows that we can also design a shallow `tanh` network to approximate **all** monomials of degree less than or equal to $s$, including both odd and even powers.  
The structure of this network is still simple (only one hidden layer), but slightly *wider* than in Lemma 3.1, since it must handle more complex tasks.

**2. Core Idea: Recursive Algebraic Technique**

Directly applying the method of Lemma 3.1 to construct even powers does not work.  
This is because `tanh` is an odd function, and all its even-order derivatives vanish at $x=0$, making the finite difference trick fail.

To resolve this, the authors adopt a purely **algebraic identity**, shown as equation (25) in the paper.  
The key idea is **recursion**:

$$
y^{2n} = \frac{1}{2\alpha(2n+1)} \left( (y+\alpha)^{2n+1} - (y-\alpha)^{2n+1} - 2\sum_{k=0}^{n-1} \dots y^{2k} \right)
$$

Interpretation of the formula:
* It shows that any **even-powered** term $y^{2n}$ can be expressed as a combination of:
  1. Two **odd-powered terms**: $(y+\alpha)^{2n+1}$ and $(y-\alpha)^{2n+1}$.  
  2. Some **lower-order even-powered terms** ($y^{2k}$, where $k < n$).

This gives us a clear construction blueprint:
* **Base case:** From Lemma 3.1, we already know how to approximate all odd-powered terms.  
* **Step 1:** To construct $y^2$, use equation (25). This requires $(y+\alpha)^3$ and $(y-\alpha)^3$. Perfect — both are odd powers, which we can already approximate!  
* **Step 2:** To construct $y^4$, use equation (25) again. This time, it involves $(y+\alpha)^5$, $(y-\alpha)^5$, and the previously constructed $y^2$.  
* **And so on...**

Through this recursive approach, starting from known odd powers, we can systematically construct all even powers.  
Finally, by integrating all these subnetworks for odd and even terms, we obtain a slightly wider shallow network capable of handling all monomials.

---

### Conclusion

Lemma 3.1 and Lemma 3.2 together form the cornerstone of this paper.  
They rigorously and creatively prove that `tanh`, despite being a seemingly simple activation function, possesses sufficient expressive power to construct any polynomial.  
This conclusion is not only interesting in its own right but also paves the way for subsequent sections of the paper, which show that `tanh` networks can approximate broader and more complex classes of functions.

---

## Questions

* When we talk about approximation, which error measure is actually used (e.g., $L^2$ norm, sup norm)?
Does the choice of metric change the conclusion?
* The proofs often use tanh or sigmoid as activation functions.
Does ReLU also have a similar approximation theorem, and are there differences in efficiency?
* The theory assumes the target function is continuous.
What happens if the target function is discontinuous or noisy — can the same approximation results still hold?