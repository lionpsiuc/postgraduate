set terminal pngcairo enhanced font "Arial,12" size 800,600
set output "individual_hulls.png"

set title "Individual Convex Hulls with All Points"
set xlabel "X Coordinate"
set ylabel "Y Coordinate"
set grid
set key outside
set style fill transparent solid 0.15 noborder

set size ratio -1 # Equal aspect ratio

# Plot all points first
plot "points.txt" using 1:2 with points pt 7 ps 0.5 lc rgb "grey" title "All Points", \
  "left_hull.txt" using 1:2 with lines lw 2 lc rgb "red" title "Left Hull", \
  "mid_hull.txt" using 1:2 with lines lw 2 lc rgb "green" title "Middle Hull", \
  "right_hull.txt" using 1:2 with lines lw 2 lc rgb "blue" title "Right Hull"

set terminal pngcairo enhanced font "Arial,12" size 800,600
set output "merged_hulls.png"

set title "Merged Convex Hulls with All Points"
set xlabel "X Coordinate"
set ylabel "Y Coordinate"
set grid
set key outside
set style fill transparent solid 0.15 noborder

set size ratio -1 # Equal aspect ratio

# Plot all points first
plot "points.txt" using 1:2 with points pt 7 ps 0.5 lc rgb "grey" title "All Points", \
  "after_first_merge.txt" using 1:2 with lines lw 2 lc rgb "orange" title "After 1st Merge", \
  "after_second_merge.txt" using 1:2 with lines lw 2 lc rgb "purple" title "After 2nd Merge (Final)"

set terminal pngcairo enhanced font "Arial,12" size 800,600
set output "serial_hull.png"

set title "Serial Convex Hull with All Points"
set xlabel "X Coordinate"
set ylabel "Y Coordinate"
set grid
set key outside
set style fill transparent solid 0.15 noborder

set size ratio -1 # Equal aspect ratio

# Plot all points first
plot "points.txt" using 1:2 with points pt 7 ps 0.5 lc rgb "grey" title "All Points", \
  "serial.txt" using 1:2 with lines lw 2 lc rgb "dark-cyan" title "Serial Hull"
