set terminal pngcairo enhanced font "Arial,12" size 800,600

set output "individual_hulls.png"

set title "Individual Convex Hulls"
set xlabel "x-coordinate"
set ylabel "y-coordinate"
set grid
set key outside
set style fill transparent solid 0.15 noborder

set size ratio -1

plot "points.txt" using 1:2 with points pt 7 ps 0.5 lc rgb "grey" title "Points", \
  "left_hull.txt" using 1:2 with lines lw 2 lc rgb "red" title "Left Hull", \
  "mid_hull.txt" using 1:2 with lines lw 2 lc rgb "green" title "Middle Hull", \
  "right_hull.txt" using 1:2 with lines lw 2 lc rgb "blue" title "Right Hull"

set output "merged_hulls.png"

set title "Merged Convex Hulls"
set xlabel "x-coordinate"
set ylabel "y-coordinate"
set grid
set key outside
set style fill transparent solid 0.15 noborder

set size ratio -1

plot "points.txt" using 1:2 with points pt 7 ps 0.5 lc rgb "grey" title "Points", \
  "after_first_merge.txt" using 1:2 with lines lw 2 lc rgb "orange" title "First Merge", \
  "after_second_merge.txt" using 1:2 with lines lw 2 lc rgb "purple" title "Second Merge"

set output "serial_hull.png"

set title "Serial Implementation"
set xlabel "x-coordinate"
set ylabel "y-coordinate"
set grid
set key outside
set style fill transparent solid 0.15 noborder

set size ratio -1

plot "points.txt" using 1:2 with points pt 7 ps 0.5 lc rgb "grey" title "Points", \
  "serial.txt" using 1:2 with lines lw 2 lc rgb "dark-cyan" title "Serial Hull"
