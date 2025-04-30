# blob_pong
Generalization of the classic arcade-style game Pong for arbitrary ball viscosities.

### Requirements:

#### Linux:

If you need to install SuiteSparse (to install scikit-sparse), the headers are in these packages:
```
> sudo apt install liblapack-dev libopenblas-dev libgmp3-dev libmpfr-dev libsuitesparse-dev
```
Otherwise, just run:
```
> pip install jax scikit-sparse opencv-contrib-python
```

#### Windows + Anaconda:
Instructions from:[https://anaconda.org/conda-forge/scikit-sparse]().
```
> conda install conda-forge::scikit-sparse
> conda install conda-forge::jax
```
In your Venv:
```
> pip install opencv numpy matplotlib
```







### Running the fluid solver

`> python fluid_solver.py`

Right now, it's just smoke being wafted through a random, static velocity field with nonzero divergence (watch the mass creep up/down).


![static advection](/static_advection.png)

The vector field is interpolated over a random 15-cell velocity grid.  
The fluid density resolution is 10x that. 


#### Roadmap / To Do:

For V1 (gas phase only):
  * Interaction with static objects other than 4 walls.
  * Interaction with moving objects.
  * Multi-colored smoke.
  * Sources / Sinks / Fans

For V2:
  * liquid phase
  * gravity
  * lquid sources / sinks

Technical:
  * move $A$ matrix and $b$ vector construction to c/cython for speed.
  * Synchronize simulation speed with real-time.