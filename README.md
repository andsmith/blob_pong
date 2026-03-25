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

For Version 0.1 (gas phase only):
  * Multi-colored smoke/liquid  (advecting 2 fields for smoke, can blend).

  * liquid phase
  * gravity

For Version 0.2:
  * gaseous sources / sinks / fans (start a fire)
  * liquid sources / sinks / pumps (pour some water)

For Version 0.3:
  * Interaction with static rigid bodies (arbitrary shapes) other than walls.

For Version 0.4:
  * Interaction with dynamic rigid bodies (arbitrary shapes).
    * Mouse drags/rotates an I-shape (fan) to waft smoke, or
    * Mouse drags/rotates a U-shape (bucket) to scoop/pour liquid.  


Technical:
  * Move interpolation to jax
  * move $A$ matrix and $b$ vector construction to c/cython for speed.
  * Synchronize simulation speed with real-time.