from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def add_inset_box(ax, x1, x2, y1, y2, loc='lower left', connect_loc1=2, connect_loc2=4):
    """
    Adds an inset box to the given axis `ax` positioned at loc
    that zooms into the region defined by (x1, x2) and (y1, y2).

    Connect_loc1 and connect_loc2 specify the corners 
    of the box to connect to the inset plot, 
    with 1=upper right, 2=upper left, 
    3=lower left, 4=lower right.
    """

    # Create inset axes in the lower left corner of the density plot
    axins = inset_axes(ax, width="40%", height="40%", loc=loc)

    # Re-plot all data from the main density plot onto the inset axes
    for line in ax.get_lines():
        axins.plot(line.get_xdata(), line.get_ydata(),
                linestyle=line.get_linestyle(),
                color=line.get_color())

    # Set the limits of the zoom-in box
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.tick_params(labelleft=False, labelbottom=False)
    # no axis ticks in the inset plot
    axins.xaxis.set_ticks([])
    axins.yaxis.set_ticks([])

    # Draw a box around the region of interest on the main plot
    # and connect it to the inset plot for clarity
    mark_inset(ax, axins, loc1=connect_loc1, loc2=connect_loc2, fc="none", ec="0.5")