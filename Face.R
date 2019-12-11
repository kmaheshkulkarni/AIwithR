# devtools::install_github("ropensci/opencv")

#Libraries
library(opencv)
library(psych)

# Live face detection:
ocv_video(ocv_face)

# Various options
ocv_video(ocv_edges)
ocv_video(ocv_knn)
ocv_video(ocv_facemask)
ocv_video(ocv_mog2)
ocv_video(ocv_stylize)
ocv_video(ocv_sketch)

# Overlay face filter  
test <- ocv_camera()   
bitmap <- ocv_bitmap(test)
width <- dim(bitmap)[2]
height <- dim(bitmap)[3]

png('bg.png', width = width, height = height)
data('iris')
print(pairs.panels(iris[1:4], 
                   gap=0,
                   pch=21,
                   bg = c("red", "green", "blue")[iris$Species]))
dev.off()  
bg <- ocv_read('C:/Users/v0mg1uf/Documents/Github/AIwithR/st1.jpg')

ocv_video(function(input){
  mask <- ocv_facemask(input)
  ocv_copyto(input, bg, mask) })

# Face recognition
ccb <- ocv_read('C:/Users/v0mg1uf/Documents/Github/AIwithR/st1.jpg')
faces <- ocv_face(ccb)
ocv_write(bg, 'C:/Users/v0mg1uf/Documents/Github/AIwithR/st1.jpg')

# Various options
ocv_sketch(ccb, color = T)
ocv_blur(ccb, ksize = 15)
ocv_hog(ccb)
ocv_markers(ccb)
ocv_stylize(ccb)

# get the face location data:
facemask <- ocv_facemask(ccb)
attr(facemask, 'faces')
