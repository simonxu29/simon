#---------------------------------------------------------
# File:   MIT18_05S22_shading_curve_polygon.r
# Author: Jeremy Orloff
#
# MIT OpenCourseWare: https://ocw.mit.edu
# 18.05 Introduction to Probability and Statistics
# Spring 2022
# For information about citing these materials or our Terms of Use, visit:
# https://ocw.mit.edu/terms}.
#
#---------------------------------------------------------

# Tutorial:
# Plotting with curve(),
# Shading with lines() and polygon()

do_plot1 = TRUE
do_plot2 = TRUE
do_plot3 = TRUE
do_plot4 = TRUE

if (do_plot1) {
   x = seq(-3, 3, 0.01)  #finely spaced for shading
   y = dnorm(x)

   # Draw curve
   plot(x, y, type='l', xlab='', ylab='', axes=FALSE)
   ind = (x>-1) & (x<1.5)

   # Draw shading --finely spaced vertical lines
   # type='h' says to draw histogram like vertical lines
   col = 'gray80'
   lines(x[ind], y[ind], col=col, type='h')  #shading
   title('Plot 1: Using lines() to shade', cex=2)
}

if (do_plot2) {
   x = seq(-3, 3, 0.01)
   y = dnorm(x)

   # Draw curve
   plot(x, y, type='l')
   title('Plot 2: Using lines() to make bars', cex=2)

   # Draw boxes
   x = seq(-1, 1.5, 0.5)
   y = dnorm(x)
   n = length(x)
   for (j in 1:(n-1)) {
      xb = x[j]
      yb = y[j+1]
      lines(c(xb,xb,x[j+1]), c(0,yb,yb))
      if (yb < y[j]) {
         lines(c(xb,xb), c(y[j],yb))
      }
   }
   lines(c(x[n],x[n]), c(0,y[n]))
}

if (do_plot3) {
   # Same as Plot 1 using curve() and polygon()

   # Use curve() to make a plot
   # Note curve() replaces the use of plot()
   curve(dnorm, from=-3, to=3, xlim=c(-5,5), xlab='', ylab='', axes=T)
   title('Plot 3: curve() for graph, polygon() for shading', cex=2)

   # Use polygon() for shading
   x = seq(-3, 3, 0.01)  #finely spaced
   y = dnorm(x)
   ind = (x>-1) & (x<1.5)
   polygon(c(-1,x[ind],1.5,-1), c(0,y[ind],0,0), col='orange')
}

if (do_plot4) {
   # Same as Plot 2 using curve()
   curve(dnorm, from=-3, to=3, xlim=c(-5,5), xlab='', ylab='', axes=T)
   title('Plot 4: curve()', cex=2)

   # Draw boxes
   x = seq(-1, 1.5, 0.5)
   y = dnorm(x)
   n = length(x)
   for (j in 1:(n-1)) {
      xb = x[j]
      yb = y[j+1]
      lines(c(xb,xb,x[j+1]), c(0,yb,yb))
      if (yb < y[j])
         lines(c(xb,xb), c(y[j],yb))
   }
   lines(c(x[n],x[n]), c(0,y[n]))
}
