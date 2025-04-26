# blob_pong
Generalization of the classic arcade-style game Pong for arbitrary ball viscosities.

### Requirements:




### Running the fluid solver

`> python fluid_solver.py`

Right now, it's just smoke being wafted through a random, static velocity field with nonzero divergence (watch the mass creep up/down).


![static advection](/static_advection.png)

The vector field is interpolated over a random 15-cell velocity grid.  The fluid density resolution is 10x that. 
