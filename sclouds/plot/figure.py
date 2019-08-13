from matplotlib.pyplot import Figure


class MSfigure(Figure):
  """

    num : integer or string, optional, default: None

        If not provided, a new figure will be created, and the figure number will be incremented. The figure objects holds this number in a number attribute. If num is provided, and a figure with this id already exists, make it active, and returns a reference to it. If this figure does not exists, create it and returns it. If num is a string, the window title will be set to this figure's num.
    figsize : (float, float), optional, default: None

        width, height in inches. If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8].
    dpi : integer, optional, default: None

        resolution of the figure. If not provided, defaults to rcParams["figure.dpi"] = 100.
    facecolor : color spec

        the background color. If not provided, defaults to rcParams["figure.facecolor"] = 'w'.
    edgecolor : color spec

        the border color. If not provided, defaults to rcParams["figure.edgecolor"] = 'w'.
    frameon : bool, optional, default: True

        If False, suppress drawing the figure frame.
    FigureClass : subclass of Figure

        Optionally use a custom Figure instance.
    clear : bool, optional, default: False

        If True and the figure already exists, then it is cleared.

  """
  def __init__():
    # call super
    rc('figure', figsize=(11.69,8.27))



    pass


  def savefig():
    pass
