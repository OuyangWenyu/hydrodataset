import matplotlib.pyplot as plt
import pydaymet as daymet
from pynhd import NLDI


class DaymetVx(object):
    def __init__(self):
        print("the xth version daymet")


geometry = NLDI().get_basins("01013500").geometry[0]
var = ["prcp", "tmin"]
dates = ("2000-01-01", "2000-03-31")
# daily = daymet.get_bygeom(geometry, dates, variables=var, pet=True)
monthly = daymet.get_bygeom(geometry, dates, variables=var, time_scale="monthly")
ax = monthly.prcp.plot(x="x", y="y", row="time", col_wrap=3)
print("finish")
plt.show()
