#!/usr/bin/env python
"""Render the target-time panel under several camera configs to choose a more
frontal screen orientation.  jf1uids env (pyvista)."""
import numpy as np
import pyvista as pv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
pv.OFF_SCREEN = True

d = np.load("panel_snaps.npz")
proj_axis = 2
F = np.moveaxis(d["target"].astype(np.float32), proj_axis, 0).copy()
allv = np.concatenate([np.moveaxis(d[k], proj_axis, 0).ravel() for k in ["init","pre","target","post"]])
vclim = (float(np.percentile(allv,50)), float(np.percentile(allv,99.5)))


def render(F, az, el, gap=0.6, zoom=1.15, window=(900,900)):
    N = F.shape
    grid = pv.ImageData(); grid.dimensions = np.array(F.shape)+1
    grid.cell_data["density"] = F.ravel(order="F")
    pl = pv.Plotter(off_screen=True, window_size=window); pl.set_background("white")
    pl.add_volume(grid, scalars="density", cmap="viridis", clim=vclim,
                  opacity="sigmoid", blending="composite", shade=True, show_scalar_bar=False)
    ax=0; proj=F.sum(axis=ax); gpix=gap*N[ax]
    others=[i for i in range(3) if i!=ax]
    a=np.linspace(0,N[others[0]],N[others[0]]+1); b=np.linspace(0,N[others[1]],N[others[1]]+1)
    A,B=np.meshgrid(a,b,indexing="ij")
    coords={others[0]:A, others[1]:B, ax:np.full_like(A,-gpix)}
    sg=pv.StructuredGrid(coords[0],coords[1],coords[2])
    sg.cell_data["proj"]=proj.ravel(order="F")
    lo,hi=np.percentile(proj,2),np.percentile(proj,98)
    pl.add_mesh(sg,scalars="proj",cmap="magma",clim=(lo,hi),show_scalar_bar=False,lighting=False)
    pl.add_mesh(sg.extract_feature_edges(),color=[0.35,0.35,0.35],line_width=1.5)
    for ca in (0,N[others[0]]):
        for cb in (0,N[others[1]]):
            p0={others[0]:ca,others[1]:cb,ax:0.0}; p1={others[0]:ca,others[1]:cb,ax:-gpix}
            pl.add_mesh(pv.Line([p0[0],p0[1],p0[2]],[p1[0],p1[1],p1[2]]),color=[0.5,0.5,0.5],line_width=1.0)
    pl.camera_position="iso"; pl.camera.azimuth=az; pl.camera.elevation=el; pl.camera.zoom(zoom)
    img=pl.screenshot(return_img=True); pl.close(); return img


# candidate (azimuth, elevation) — more frontal screen = view direction closer
# to the screen normal (scene x).  Lower azimuth swings the screen toward us.
cands = [(20,12),(0,8),(-15,8),(-30,6),(-45,5),(-60,8)]
fig,axes=plt.subplots(2,3,figsize=(15,10))
for axp,(az,el) in zip(axes.ravel(),cands):
    axp.imshow(render(F,az,el)); axp.set_title(f"az={az} el={el}"); axp.set_xticks([]); axp.set_yticks([])
fig.tight_layout(); fig.savefig("_cam_tune.png",dpi=100); print("wrote _cam_tune.png")
