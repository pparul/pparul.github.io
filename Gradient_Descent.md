```python
# import automatic differentiator to compute gradient module
from autograd import grad 
import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```


```python
plt.rcParams['axes.grid'] = True
```


```python
# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha,max_its,w):
    # compute gradient module using autograd
    gradient = grad(g)
    #print(gradient)

    # run the gradient descent loop
    weight_history = [w]           # container for weight history
    cost_history = [g(w)]          # container for corresponding cost function history
    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)
        #print(grad_eval)
        

        # take gradient descent step
        w = w - alpha*grad_eval
        
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history,cost_history
```

## Trying a convex function


```python
J = lambda w: 1/float(50)*(w**4 + w**2 + 10*w)
plt.figure()
N = 100
w = np.round(np.random.randn(N,1),2)
sns.pointplot(x = w.ravel(), y = J(w).ravel())
#print(g(w))
```




    <Axes: >




    
![png](Gradient_Descent_files/Gradient_Descent_4_1.png)
    



```python


# run gradient descent 
w = 2.5; alpha = 1; max_its = 100
weight_history,cost_history = gradient_descent(J,alpha,max_its,w)
# plot the cost function history
sns.pointplot(x = np.round(weight_history,2), y = cost_history)
plt.xlabel('weight: w'); plt.ylabel('Cost Function: J(w)'); 
plt.xticks(rotation=45)
plt.show()
```


    
![png](Gradient_Descent_files/Gradient_Descent_5_0.png)
    



```python
## A non-convex function
##  step length divergent
## step length osciiation in 1D
```


```python

```


```python
## step-length/Learning rate alpha is fixed
# Learning starts at different places and stops at the local minimum 
# Here learning rate is very low so it stays at the same place, does not diverge. If learning rate increases then it will diverge.
```


```python
J = lambda w: np.sin(3*w) + 0.1*w**2

# run gradient descent 
w = 4.5; alpha = 0.05; max_its = 50
weight_history,cost_history = gradient_descent(J,alpha,max_its,w)

w = -2.0; alpha = 0.05; max_its = 10
weight_history_2,cost_history_2 = gradient_descent(J,alpha,max_its,w)

# plot the cost function history
w1 = np.linspace(-4,6,100)
sns.lineplot(x = w1,  y = J(w1))
sns.lineplot(x = np.round(weight_history,2), y = cost_history, marker ='o', linestyle='None')
sns.lineplot(x = np.round(weight_history_2,2), y = cost_history_2, marker ='o', linestyle='None')
```




    <Axes: >




    
![png](Gradient_Descent_files/Gradient_Descent_9_1.png)
    



```python
## Alpha is large causing learning to diverge and overshoot the minimum
```


```python
## Two-variable example
```
